import Foundation
import Metal

/// The whole model: embeddings, 36 layers, final norm, output head.
///
/// This is the first code that produces logits rather than intermediate
/// tensors, and the first that touches the entire 58.9 GiB installation in a
/// single call. Every layer streams four experts, so one token pulls roughly
/// 1.8 GiB off disk before the cache has anything useful in it.
public struct ModelRunner {
    public let context: MetalContext
    public let reader: ModelReader
    public let spec: ArchitectureSpec
    private let rope: RoPE
    private let layers: [TransformerLayer]

    public init(context: MetalContext, reader: ModelReader) {
        self.context = context
        self.reader = reader
        self.spec = reader.manifest.spec
        let rope = RoPE(configuration: .forSpec(reader.manifest.spec))
        self.rope = rope
        self.layers = (0..<reader.manifest.layerCount).map {
            TransformerLayer(context: context, reader: reader, index: $0, rope: rope)
        }
    }

    /// Resident weights held across tokens.
    ///
    /// Loading these is the one genuinely large allocation the runtime makes —
    /// about 2.1 GiB — and it happens once rather than per token.
    public final class Weights {
        public let layers: [TransformerLayer.Weights]
        public let embed: MTLBuffer
        public let finalNorm: MTLBuffer
        public let head: MTLBuffer
        public let embedShape: [Int]
        public let headShape: [Int]

        init(layers: [TransformerLayer.Weights], embed: MTLBuffer, finalNorm: MTLBuffer,
             head: MTLBuffer, embedShape: [Int], headShape: [Int]) {
            self.layers = layers
            self.embed = embed
            self.finalNorm = finalNorm
            self.head = head
            self.embedShape = embedShape
            self.headShape = headShape
        }
    }

    public func loadWeights() throws -> Weights {
        let device = context.device
        return Weights(
            layers: try layers.map { try $0.loadWeights() },
            embed: try reader.loadTrunk(section: "embed", device: device),
            finalNorm: try reader.loadTrunk(section: "final_norm", device: device),
            head: try reader.loadTrunk(section: "head", device: device),
            embedShape: try reader.trunkShape("embed"),
            headShape: try reader.trunkShape("head"))
    }

    /// Allocates the resident expert slot cache.
    ///
    /// Eight slots per layer is the measured sweet spot: 73.6% hit rate for
    /// 3.6 GiB, with returns flattening above twelve.
    public func makeExpertCache(slots: Int = 8, bypassPageCache: Bool = true) throws -> ExpertCache {
        try ExpertCache(context: context, reader: reader, slotCount: slots,
                        bypassPageCache: bypassPageCache)
    }

    /// Allocates a KV cache sized for `maxContext` tokens.
    public func makeCache(maxContext: Int) throws -> KVCache {
        try KVCache(context: context, spec: spec, maxContext: maxContext,
                    layerCount: reader.manifest.layerCount)
    }

    /// Runs a token sequence and returns logits for the final position.
    ///
    /// `positionBase` is where this run sits in the conversation, so a prompt
    /// passes 0 and each subsequent decode step passes the length so far. The
    /// cache holds every previous token's keys and values, which is what keeps
    /// decode linear rather than re-running the whole sequence per token.
    ///
    /// Only the last position's logits are produced: during generation the
    /// earlier ones are never read, and the head is a 201,088-row projection
    /// that is not worth computing for rows nobody looks at.
    public func logits(
        tokens: [Int], positionBase: Int = 0, cache: KVCache, weights: Weights,
        expertCache: ExpertCache? = nil,
        progress: (Int, Int) -> Void = { _, _ in },
        routing: ((Int, [Router.Decision]) -> Void)? = nil
    ) throws -> [Float] {
        precondition(!tokens.isEmpty, "need at least one token")
        var stream = try embed(tokens: tokens, weights: weights)

        for (index, layer) in layers.enumerated() {
            let (next, trace) = try layer.forward(
                hidden: stream, tokenCount: tokens.count, positionBase: positionBase,
                weights: weights.layers[index], cache: cache, expertCache: expertCache)
            stream = next
            routing?(index, trace.routing)
            progress(index + 1, layers.count)
        }
        cache.advance(by: tokens.count)

        // Only the last row matters from here on.
        let width = spec.hiddenSize
        let last = Array(stream[((tokens.count - 1) * width)..<(tokens.count * width)])
        let normed = try normalise(last, weight: weights.finalNorm)
        return try project(normed, codes: weights.head,
                           rows: weights.headShape[0], cols: weights.headShape[1])
    }

    /// Dequantises embedding rows for a token sequence.
    private func embed(tokens: [Int], weights: Weights) throws -> [Float] {
        let width = spec.hiddenSize
        let rows = weights.embedShape[0]
        let groups = width / Int8Affine.groupSize
        let base = weights.embed.contents()
        let meta = base.advanced(by: rows * width)

        var out = [Float](repeating: 0, count: tokens.count * width)
        for (position, token) in tokens.enumerated() {
            precondition(token >= 0 && token < rows, "token \(token) out of range")
            let codeRow = base.advanced(by: token * width)
                .bindMemory(to: UInt8.self, capacity: width)
            let metaRow = meta.advanced(by: token * groups * 4)
                .bindMemory(to: UInt16.self, capacity: groups * 2)
            for group in 0..<groups {
                let scale = BFloat16.toFloat(metaRow[group * 2])
                let zero = BFloat16.toFloat(metaRow[group * 2 + 1])
                for i in 0..<Int8Affine.groupSize {
                    let index = group * Int8Affine.groupSize + i
                    out[position * width + index] = Float(codeRow[index]) * scale + zero
                }
            }
        }
        return out
    }

    private func normalise(_ values: [Float], weight: MTLBuffer) throws -> [Float16] {
        let width = spec.hiddenSize
        let weights = weight.contents().bindMemory(to: UInt16.self, capacity: width)
        var sumSquares: Float = 0
        for value in values { sumSquares += value * value }
        let inverse = 1 / (sumSquares / Float(width) + spec.rmsNormEpsilon).squareRoot()
        return (0..<width).map {
            Float16(values[$0] * inverse * BFloat16.toFloat(weights[$0]))
        }
    }

    /// int8 affine projection, used for the output head.
    private func project(_ input: [Float16], codes: MTLBuffer, rows: Int, cols: Int) throws -> [Float] {
        let xBuffer = try context.buffer(input)
        let output = try context.emptyBuffer(of: Float.self, count: rows)
        var colsValue = UInt32(cols)

        let pipeline = try context.pipeline(shader: "router", function: "int8_affine_gemv")
        guard let commands = context.queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder()
        else { throw MetalError.encoderCreationFailed }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(codes, offset: 0, index: 0)
        encoder.setBuffer(codes, offset: rows * cols, index: 1)
        encoder.setBuffer(xBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&colsValue, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        context.run("head", commands)

        let produced = output.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: produced, count: rows))
    }
}
