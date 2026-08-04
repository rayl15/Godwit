import Foundation
import Metal

/// Runs one GPT-OSS expert's feed-forward branch on the GPU.
///
/// gate_up projection → interleaved gated activation → down projection. This is
/// the work every streamed expert performs, and the reason the whole streaming
/// design exists: it is ~25M quantised weights that are needed for one token and
/// then likely not again for a while.
public struct ExpertRunner {
    public let context: MetalContext
    public let reader: ModelReader
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let swigluLimit: Float
    public let swigluAlpha: Float

    public init(context: MetalContext, reader: ModelReader,
                swigluLimit: Float? = nil, swigluAlpha: Float? = nil) {
        self.context = context
        self.reader = reader
        let spec = reader.manifest.spec
        self.hiddenSize = spec.hiddenSize
        self.intermediateSize = spec.intermediateSize
        // Both are model properties, so the spec is the default source.
        self.swigluLimit = swigluLimit ?? spec.activationLimit
        self.swigluAlpha = swigluAlpha ?? spec.activationAlpha
    }

    /// Weights for one expert, already resident in GPU memory.
    ///
    /// Held separately from the computation so a caller can keep these across
    /// tokens — which is exactly what the slot cache will do.
    public struct Weights {
        public let gateUpBlocks: MTLBuffer
        public let gateUpScales: MTLBuffer
        public let gateUpBias: MTLBuffer
        public let downBlocks: MTLBuffer
        public let downScales: MTLBuffer
        public let downBias: MTLBuffer
    }

    public func loadWeights(layer: Int, expert: Int) throws -> Weights {
        let device = context.device
        return Weights(
            gateUpBlocks: try reader.loadSection(layer: layer, expert: expert,
                                                 section: .gateUpBlocks, device: device),
            gateUpScales: try reader.loadSection(layer: layer, expert: expert,
                                                 section: .gateUpScales, device: device),
            gateUpBias: try reader.loadSection(layer: layer, expert: expert,
                                               section: .gateUpBias, device: device),
            downBlocks: try reader.loadSection(layer: layer, expert: expert,
                                               section: .downBlocks, device: device),
            downScales: try reader.loadSection(layer: layer, expert: expert,
                                               section: .downScales, device: device),
            downBias: try reader.loadSection(layer: layer, expert: expert,
                                             section: .downBias, device: device))
    }

    /// Applies one expert to a single token's hidden state.
    ///
    /// Returns the expert's contribution before routing-weight scaling; the
    /// caller decides how to combine several experts.
    public func apply(_ input: [Float16], weights: Weights) throws -> [Float] {
        precondition(input.count == hiddenSize, "expected \(hiddenSize) inputs")

        let gateUpRows = 2 * intermediateSize
        let xBuffer = try context.buffer(input)
        let gateUpBuffer = try context.emptyBuffer(of: Float.self, count: gateUpRows)
        let activated = try context.emptyBuffer(of: Float16.self, count: intermediateSize)
        let output = try context.emptyBuffer(of: Float.self, count: hiddenSize)

        let gemv = try context.pipeline(shader: "expert", function: "mxfp4_gemv_bias")
        let activation = try context.pipeline(shader: "expert",
                                              function: "gptoss_expert_activation")

        guard let commands = context.queue.makeCommandBuffer() else {
            throw MetalError.encoderCreationFailed
        }

        // gate_up: [2F, H] @ [H] -> [2F]
        try encodeGEMV(commands, pipeline: gemv,
                       blocks: weights.gateUpBlocks, scales: weights.gateUpScales,
                       bias: weights.gateUpBias, x: xBuffer, y: gateUpBuffer,
                       rows: gateUpRows, cols: hiddenSize)

        // Interleaved split, asymmetric clamp, sigmoid gate. See expert.metal.
        guard let encoder = commands.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        var width = UInt32(intermediateSize)
        var limit = swigluLimit
        var alpha = swigluAlpha
        encoder.setComputePipelineState(activation)
        encoder.setBuffer(gateUpBuffer, offset: 0, index: 0)
        encoder.setBuffer(activated, offset: 0, index: 1)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&limit, length: MemoryLayout<Float>.size, index: 3)
        encoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 4)
        encoder.dispatchThreads(MTLSize(width: intermediateSize, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()

        // down: [H, F] @ [F] -> [H]
        try encodeGEMV(commands, pipeline: gemv,
                       blocks: weights.downBlocks, scales: weights.downScales,
                       bias: weights.downBias, x: activated, y: output,
                       rows: hiddenSize, cols: intermediateSize)

        commands.commit()
        commands.waitUntilCompleted()

        let produced = output.contents().bindMemory(to: Float.self, capacity: hiddenSize)
        return Array(UnsafeBufferPointer(start: produced, count: hiddenSize))
    }

    private func encodeGEMV(
        _ commands: MTLCommandBuffer, pipeline: MTLComputePipelineState,
        blocks: MTLBuffer, scales: MTLBuffer, bias: MTLBuffer,
        x: MTLBuffer, y: MTLBuffer, rows: Int, cols: Int
    ) throws {
        guard let encoder = commands.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        var colsValue = UInt32(cols)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(blocks, offset: 0, index: 0)
        encoder.setBuffer(scales, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(x, offset: 0, index: 3)
        encoder.setBuffer(y, offset: 0, index: 4)
        encoder.setBytes(&colsValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
