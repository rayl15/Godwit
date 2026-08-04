import Foundation
import Metal

/// Selects which experts a token needs.
///
/// Everything downstream waits on this. The router decides which ~12.6 MiB
/// blobs get read from disk, so no fetch can start until it has run — which is
/// why its weights stay resident and BF16 while comparable tensors are
/// quantised, and why it is the one place where latency matters more than
/// throughput.
public struct Router {
    /// One token's routing decision.
    public struct Decision: Sendable, Equatable {
        /// Chosen expert ids, highest logit first.
        public let experts: [Int]
        /// Softmax over the selected logits only — not over all experts.
        public let weights: [Float]

        public init(experts: [Int], weights: [Float]) {
            self.experts = experts
            self.weights = weights
        }
    }

    public let context: MetalContext
    public let expertCount: Int
    public let hiddenSize: Int
    public let topK: Int

    public init(context: MetalContext, expertCount: Int, hiddenSize: Int, topK: Int) {
        self.context = context
        self.expertCount = expertCount
        self.hiddenSize = hiddenSize
        self.topK = topK
    }

    /// Computes router logits for one token.
    public func logits(
        for hidden: [Float16], weight: MTLBuffer, bias: MTLBuffer
    ) throws -> [Float] {
        precondition(hidden.count == hiddenSize, "expected \(hiddenSize) inputs")

        let xBuffer = try context.buffer(hidden)
        let output = try context.emptyBuffer(of: Float.self, count: expertCount)
        var cols = UInt32(hiddenSize)

        let pipeline = try context.pipeline(shader: "router", function: "bf16_gemv_bias")
        guard let commands = context.queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder()
        else { throw MetalError.encoderCreationFailed }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: 0, index: 1)
        encoder.setBuffer(xBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: expertCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()

        let produced = output.contents().bindMemory(to: Float.self, capacity: expertCount)
        return Array(UnsafeBufferPointer(start: produced, count: expertCount))
    }

    /// Turns logits into a routing decision.
    ///
    /// **Order matters**: top-k is taken first and the softmax runs over only
    /// those `k` logits. Normalising over all 128 first and then selecting gives
    /// different weights, and the difference is not subtle.
    ///
    /// Selection runs on the CPU deliberately. The expert ids are needed on the
    /// CPU anyway — to plan against the slot cache and to issue reads — so
    /// computing them on the GPU would only add a round trip on the one path
    /// where latency is not hideable.
    public func select(logits: [Float]) -> Decision {
        precondition(logits.count == expertCount, "expected \(expertCount) logits")

        var order = Array(logits.indices)
        // Partial selection: k is 4 against 128, so a full sort would be waste.
        for position in 0..<topK {
            var best = position
            for candidate in (position + 1)..<order.count
            where logits[order[candidate]] > logits[order[best]] {
                best = candidate
            }
            order.swapAt(position, best)
        }
        let experts = Array(order.prefix(topK))

        let selected = experts.map { logits[$0] }
        let peak = selected.max() ?? 0
        let exponentials = selected.map { Foundation.exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        return Decision(experts: experts, weights: exponentials.map { $0 / total })
    }

    public func route(
        hidden: [Float16], weight: MTLBuffer, bias: MTLBuffer
    ) throws -> Decision {
        select(logits: try logits(for: hidden, weight: weight, bias: bias))
    }
}
