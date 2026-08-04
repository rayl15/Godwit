import Foundation
import Metal

/// Grouped-query attention with sinks, for one layer.
///
/// Everything here is resident: projections, sinks, and the KV cache. Attention
/// is the part of the model that does *not* stream, which is why it stays int8
/// rather than being pushed to 4 bits — the memory saved would be small and the
/// quality cost is not.
public struct Attention {
    public let context: MetalContext
    public let spec: ArchitectureSpec
    public let rope: RoPE

    public init(context: MetalContext, spec: ArchitectureSpec, rope: RoPE) {
        self.context = context
        self.spec = spec
        self.rope = rope
    }

    /// Resident weights for one layer's attention block.
    public struct Weights {
        public let qCodes: MTLBuffer
        public let kCodes: MTLBuffer
        public let vCodes: MTLBuffer
        public let oCodes: MTLBuffer
        public let sinks: MTLBuffer
        /// BF16 projection biases. GPT-OSS sets attention_bias, unlike most
        /// contemporary models; omitting these runs and is simply wrong.
        public let qBias: MTLBuffer
        public let kBias: MTLBuffer
        public let vBias: MTLBuffer
        public let oBias: MTLBuffer

        public init(qCodes: MTLBuffer, kCodes: MTLBuffer, vCodes: MTLBuffer,
                    oCodes: MTLBuffer, sinks: MTLBuffer,
                    qBias: MTLBuffer, kBias: MTLBuffer,
                    vBias: MTLBuffer, oBias: MTLBuffer) {
            self.qCodes = qCodes
            self.kCodes = kCodes
            self.vCodes = vCodes
            self.oCodes = oCodes
            self.sinks = sinks
            self.qBias = qBias
            self.kBias = kBias
            self.vBias = vBias
            self.oBias = oBias
        }
    }

    public static func loadWeights(
        reader: ModelReader, layer: Int, device: MTLDevice
    ) throws -> Weights {
        Weights(
            qCodes: try reader.loadTrunk(section: "layer\(layer).q_proj", device: device),
            kCodes: try reader.loadTrunk(section: "layer\(layer).k_proj", device: device),
            vCodes: try reader.loadTrunk(section: "layer\(layer).v_proj", device: device),
            oCodes: try reader.loadTrunk(section: "layer\(layer).o_proj", device: device),
            sinks: try reader.loadTrunk(section: "layer\(layer).sinks", device: device),
            qBias: try reader.loadTrunk(section: "layer\(layer).q_bias", device: device),
            kBias: try reader.loadTrunk(section: "layer\(layer).k_bias", device: device),
            vBias: try reader.loadTrunk(section: "layer\(layer).v_bias", device: device),
            oBias: try reader.loadTrunk(section: "layer\(layer).o_bias", device: device))
    }

    /// Projects, rotates, attends, and projects back for a run of tokens.
    ///
    /// `hidden` is `[tokenCount, hiddenSize]` already normalised. Returns the
    /// same shape. Positions start at `positionBase`, so a decode step passes
    /// one token and its absolute position.
    public func forward(
        hidden: [Float16], tokenCount: Int, positionBase: Int,
        weights: Weights, layer: Int, cache: KVCache
    ) throws -> [Float16] {
        let heads = spec.attentionHeads
        let kvHeads = spec.keyValueHeads
        let headDim = spec.headDimension
        let qWidth = heads * headDim
        let kvWidth = kvHeads * headDim

        // Buffers are allocated through `context`; no direct device use here.
        let hiddenBuffer = try context.buffer(hidden)

        // Everything below goes into a single command buffer. Compute encoders
        // within one buffer run in order, so the data dependencies hold without
        // a CPU round trip between each step — and it is those round trips,
        // not the arithmetic, that dominate decode.
        guard let commands = context.queue.makeCommandBuffer() else {
            throw MetalError.encoderCreationFailed
        }

        let q = try context.emptyBuffer(of: Float16.self, count: tokenCount * qWidth)
        let k = try context.emptyBuffer(of: Float16.self, count: tokenCount * kvWidth)
        let v = try context.emptyBuffer(of: Float16.self, count: tokenCount * kvWidth)

        try encodeProjection(commands, input: hiddenBuffer, codes: weights.qCodes,
                             bias: weights.qBias, output: q,
                             rows: qWidth, cols: spec.hiddenSize, tokens: tokenCount)
        try encodeProjection(commands, input: hiddenBuffer, codes: weights.kCodes,
                             bias: weights.kBias, output: k,
                             rows: kvWidth, cols: spec.hiddenSize, tokens: tokenCount)
        try encodeProjection(commands, input: hiddenBuffer, codes: weights.vCodes,
                             bias: weights.vBias, output: v,
                             rows: kvWidth, cols: spec.hiddenSize, tokens: tokenCount)

        // Rotate Q and K. V is deliberately untouched: position information
        // belongs in the similarity, not in the payload.
        let (cosBuffer, sinBuffer) = try ropeTables(tokenCount: tokenCount,
                                                    positionBase: positionBase)
        try encodeRoPE(commands, vectors: q, tokens: tokenCount, heads: heads,
                       cosines: cosBuffer, sines: sinBuffer)
        try encodeRoPE(commands, vectors: k, tokens: tokenCount, heads: kvHeads,
                       cosines: cosBuffer, sines: sinBuffer)

        try cache.encodeWrite(commands, keys: k, values: v, layer: layer,
                              position: positionBase, tokenCount: tokenCount)
        let layerCache = cache.layers[layer]

        let attended = try context.emptyBuffer(of: Float16.self, count: tokenCount * qWidth)
        let pipeline = try context.pipeline(shader: "attention", function: "gqa_attention_sinks",
                                            constants: [0: spec.headDimension])
        guard let encoder = commands.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }

        var keyCount = UInt32(positionBase + tokenCount)
        var headCount = UInt32(heads)
        var kvCount = UInt32(kvHeads)
        var window = UInt32(spec.layers[layer].attention == .sliding
                            ? spec.layers[layer].window : 0)
        var base = UInt32(positionBase)
        var scale = 1 / Float(headDim).squareRoot()
        var ring = UInt32(layerCache.ring)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(layerCache.keys, offset: 0, index: 1)
        encoder.setBuffer(layerCache.values, offset: 0, index: 2)
        encoder.setBuffer(weights.sinks, offset: 0, index: 3)
        encoder.setBuffer(attended, offset: 0, index: 4)
        encoder.setBytes(&keyCount, length: 4, index: 5)
        encoder.setBytes(&headCount, length: 4, index: 6)
        encoder.setBytes(&kvCount, length: 4, index: 7)
        encoder.setBytes(&window, length: 4, index: 8)
        encoder.setBytes(&base, length: 4, index: 9)
        encoder.setBytes(&scale, length: 4, index: 10)
        encoder.setBytes(&ring, length: 4, index: 11)
        encoder.dispatchThreadgroups(
            MTLSize(width: tokenCount, height: heads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()

        let projected = try context.emptyBuffer(of: Float16.self,
                                                count: tokenCount * spec.hiddenSize)
        try encodeProjection(commands, input: attended, codes: weights.oCodes,
                             bias: weights.oBias, output: projected,
                             rows: spec.hiddenSize, cols: qWidth, tokens: tokenCount)

        commands.commit()
        commands.waitUntilCompleted()

        let produced = projected.contents().bindMemory(
            to: Float16.self, capacity: tokenCount * spec.hiddenSize)
        return Array(UnsafeBufferPointer(start: produced, count: tokenCount * spec.hiddenSize))
    }

    /// Encodes an int8 projection over all tokens into an existing buffer.
    private func encodeProjection(
        _ commands: MTLCommandBuffer, input: MTLBuffer, codes: MTLBuffer,
        bias: MTLBuffer, output: MTLBuffer, rows: Int, cols: Int, tokens: Int
    ) throws {
        guard let encoder = commands.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        var colsValue = UInt32(cols)
        var rowsValue = UInt32(rows)
        let pipeline = try context.pipeline(shader: "router",
                                            function: "int8_affine_gemv_bias_batched")
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(codes, offset: 0, index: 0)
        encoder.setBuffer(codes, offset: rows * cols, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBytes(&colsValue, length: 4, index: 5)
        encoder.setBytes(&rowsValue, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeRoPE(
        _ commands: MTLCommandBuffer, vectors: MTLBuffer, tokens: Int, heads: Int,
        cosines: MTLBuffer, sines: MTLBuffer
    ) throws {
        guard let encoder = commands.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        let pipeline = try context.pipeline(shader: "attention", function: "apply_rope",
                                            constants: [0: spec.headDimension])
        var headCount = UInt32(heads)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(vectors, offset: 0, index: 0)
        encoder.setBuffer(cosines, offset: 0, index: 1)
        encoder.setBuffer(sines, offset: 0, index: 2)
        encoder.setBytes(&headCount, length: 4, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: tokens, height: heads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func ropeTables(tokenCount: Int, positionBase: Int) throws -> (MTLBuffer, MTLBuffer) {
        let half = spec.headDimension / 2
        var cosines = [Float](repeating: 0, count: tokenCount * half)
        var sines = [Float](repeating: 0, count: tokenCount * half)
        for token in 0..<tokenCount {
            let (c, s) = rope.table(position: positionBase + token)
            cosines.replaceSubrange((token * half)..<((token + 1) * half), with: c)
            sines.replaceSubrange((token * half)..<((token + 1) * half), with: s)
        }
        return (try context.buffer(cosines), try context.buffer(sines))
    }

}
