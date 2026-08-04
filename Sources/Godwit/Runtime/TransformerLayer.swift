import Foundation
import Metal

/// One complete transformer layer.
///
/// ```
/// h += attention(norm(h))          <- resident weights
/// h += moe(norm(h))                <- streamed weights
/// ```
///
/// The asymmetry between those two lines is the entire project. The first
/// touches ~26M resident parameters; the second selects 4 experts out of 128 and
/// pulls ~50 MiB off disk to do it.
public struct TransformerLayer {
    public let context: MetalContext
    public let reader: ModelReader
    public let spec: ArchitectureSpec
    public let index: Int

    private let attention: Attention
    private let router: Router
    private let experts: ExpertRunner

    public init(context: MetalContext, reader: ModelReader, index: Int, rope: RoPE) {
        self.context = context
        self.reader = reader
        self.spec = reader.manifest.spec
        self.index = index
        self.attention = Attention(context: context, spec: spec, rope: rope)
        self.router = Router(context: context,
                             expertCount: spec.layers[index].routedExpertCount,
                             hiddenSize: spec.hiddenSize,
                             topK: spec.layers[index].expertsPerToken)
        self.experts = ExpertRunner(context: context, reader: reader)
    }

    /// Resident weights for this layer. Attention plus the two norms and the
    /// router — everything except the experts.
    public struct Weights {
        public let attention: Attention.Weights
        public let inputNorm: MTLBuffer
        public let postNorm: MTLBuffer
        public let routerWeight: MTLBuffer
        public let routerBias: MTLBuffer
    }

    public func loadWeights() throws -> Weights {
        let device = context.device
        return Weights(
            attention: try Attention.loadWeights(reader: reader, layer: index, device: device),
            inputNorm: try reader.loadTrunk(section: "layer\(index).input_norm", device: device),
            postNorm: try reader.loadTrunk(section: "layer\(index).post_norm", device: device),
            routerWeight: try reader.loadTrunk(section: "layer\(index).router_w", device: device),
            routerBias: try reader.loadTrunk(section: "layer\(index).router_b", device: device))
    }

    /// What the layer did, for tracing and cache accounting.
    public struct Trace: Sendable {
        public let routing: [Router.Decision]
    }

    /// Runs the layer over `tokenCount` tokens starting at `positionBase`.
    ///
    /// `hidden` is the residual stream, `[tokenCount, hiddenSize]`, and is
    /// returned updated.
    public func forward(
        hidden: [Float16], tokenCount: Int, positionBase: Int,
        weights: Weights, cache: KVCache
    ) throws -> (hidden: [Float16], trace: Trace) {
        let width = spec.hiddenSize
        var stream = hidden

        // --- Attention block ---
        var normed = try normalise(stream, tokenCount: tokenCount, weight: weights.inputNorm)
        let attended = try attention.forward(hidden: normed, tokenCount: tokenCount,
                                             positionBase: positionBase,
                                             weights: weights.attention, layer: index,
                                             cache: cache)
        for i in 0..<(tokenCount * width) {
            stream[i] = Float16(Float(stream[i]) + Float(attended[i]))
        }

        // --- Mixture-of-experts block ---
        normed = try normalise(stream, tokenCount: tokenCount, weight: weights.postNorm)

        var decisions: [Router.Decision] = []
        decisions.reserveCapacity(tokenCount)
        // One expert's weights are ~12.6 MiB, so reuse within a batch matters
        // even before a real slot cache exists.
        var loaded: [Int: ExpertRunner.Weights] = [:]

        for token in 0..<tokenCount {
            let slice = Array(normed[(token * width)..<((token + 1) * width)])
            let decision = try router.route(hidden: slice,
                                            weight: weights.routerWeight,
                                            bias: weights.routerBias)
            decisions.append(decision)

            var combined = [Float](repeating: 0, count: width)
            for (position, expert) in decision.experts.enumerated() {
                let expertWeights: ExpertRunner.Weights
                if let cached = loaded[expert] {
                    expertWeights = cached
                } else {
                    expertWeights = try experts.loadWeights(layer: index, expert: expert)
                    loaded[expert] = expertWeights
                }
                let output = try experts.apply(slice, weights: expertWeights)
                let scale = decision.weights[position]
                for i in 0..<width { combined[i] += output[i] * scale }
            }

            for i in 0..<width {
                stream[token * width + i] = Float16(
                    Float(stream[token * width + i]) + combined[i])
            }
        }

        return (stream, Trace(routing: decisions))
    }

    /// RMSNorm on the CPU.
    ///
    /// One row per token over 2,880 values is trivial next to the reads this
    /// layer performs; the GPU kernel exists for when the rest of the loop stops
    /// round-tripping through host memory.
    private func normalise(
        _ values: [Float16], tokenCount: Int, weight: MTLBuffer
    ) throws -> [Float16] {
        let width = spec.hiddenSize
        let weights = weight.contents().bindMemory(to: UInt16.self, capacity: width)
        var out = [Float16](repeating: 0, count: values.count)

        for token in 0..<tokenCount {
            let base = token * width
            var sumSquares: Float = 0
            for i in 0..<width {
                let value = Float(values[base + i])
                sumSquares += value * value
            }
            let inverse = 1 / (sumSquares / Float(width) + spec.rmsNormEpsilon).squareRoot()
            for i in 0..<width {
                out[base + i] = Float16(
                    Float(values[base + i]) * inverse * BFloat16.toFloat(weights[i]))
            }
        }
        return out
    }
}
