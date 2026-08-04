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
    /// Use the persistent-workgroup projection kernel.
    ///
    /// Kept switchable so the two can be compared on identical inputs — the
    /// naive kernel is the correctness reference the fast one is checked
    /// against.
    public var usePersistentKernel = true

    /// Threadgroups launched by the persistent kernel.
    ///
    /// Enough to fill the GPU with several waves so stragglers overlap, but not
    /// so many that the grid-stride loop degenerates back into one row per
    /// threadgroup. The M4 has 10 cores.
    public static let persistentThreadgroups = 64

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

    /// One expert as a single buffer plus the offsets of its six sub-tensors.
    ///
    /// A slot holds the whole expert stride, so the sections are addressed by
    /// offset rather than being separate allocations. That is what lets a cache
    /// hit cost nothing: the bytes are already where the GPU expects them.
    public struct Weights {
        public let buffer: MTLBuffer
        public let offsets: [ExpertLayout.Section: Int]

        public init(buffer: MTLBuffer, offsets: [ExpertLayout.Section: Int]) {
            self.buffer = buffer
            self.offsets = offsets
        }

        func offset(_ section: ExpertLayout.Section) -> Int { offsets[section] ?? 0 }
    }

    /// Section offsets within an expert stride, from the manifest.
    public var sectionOffsets: [ExpertLayout.Section: Int] {
        var result: [ExpertLayout.Section: Int] = [:]
        for section in ExpertLayout.Section.allCases {
            if let span = reader.manifest.expertSections[section.rawValue] {
                result[section] = span.offset
            }
        }
        return result
    }

    /// Reads one expert into fresh memory, bypassing the cache.
    ///
    /// Used by the verification commands, which want to read a specific expert
    /// without disturbing cache state.
    public func loadWeights(layer: Int, expert: Int) throws -> Weights {
        let buffer = try reader.loadExpertStride(layer: layer, expert: expert,
                                                 device: context.device)
        return Weights(buffer: buffer, offsets: sectionOffsets)
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

        let gemv = try context.pipeline(
            shader: "expert",
            function: usePersistentKernel ? "mxfp4_gemv_bias_multirow" : "mxfp4_gemv_bias")
        let activation = try context.pipeline(shader: "expert",
                                              function: "gptoss_expert_activation")

        guard let commands = context.queue.makeCommandBuffer() else {
            throw MetalError.encoderCreationFailed
        }

        // gate_up: [2F, H] @ [H] -> [2F]
        try encodeGEMV(commands, pipeline: gemv, source: weights.buffer,
                       blocksOffset: weights.offset(.gateUpBlocks),
                       scalesOffset: weights.offset(.gateUpScales),
                       biasOffset: weights.offset(.gateUpBias),
                       x: xBuffer, y: gateUpBuffer,
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
        try encodeGEMV(commands, pipeline: gemv, source: weights.buffer,
                       blocksOffset: weights.offset(.downBlocks),
                       scalesOffset: weights.offset(.downScales),
                       biasOffset: weights.offset(.downBias),
                       x: activated, y: output,
                       rows: hiddenSize, cols: intermediateSize)

        commands.commit()
        commands.waitUntilCompleted()

        let produced = output.contents().bindMemory(to: Float.self, capacity: hiddenSize)
        return Array(UnsafeBufferPointer(start: produced, count: hiddenSize))
    }

    private func encodeGEMV(
        _ commands: MTLCommandBuffer, pipeline: MTLComputePipelineState,
        source: MTLBuffer, blocksOffset: Int, scalesOffset: Int, biasOffset: Int,
        x: MTLBuffer, y: MTLBuffer, rows: Int, cols: Int
    ) throws {
        guard let encoder = commands.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        var colsValue = UInt32(cols)
        var rowsValue = UInt32(rows)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: blocksOffset, index: 0)
        encoder.setBuffer(source, offset: scalesOffset, index: 1)
        encoder.setBuffer(source, offset: biasOffset, index: 2)
        encoder.setBuffer(x, offset: 0, index: 3)
        encoder.setBuffer(y, offset: 0, index: 4)
        encoder.setBytes(&colsValue, length: MemoryLayout<UInt32>.size, index: 5)

        if usePersistentKernel {
            encoder.setBytes(&rowsValue, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: Self.persistentThreadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            encoder.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        encoder.endEncoding()
    }
}
