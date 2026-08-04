import Foundation
import Metal

/// End-to-end check against real GPT-OSS-120B weights.
///
/// Reads an expert off disk with `pread`, runs the fused MXFP4 GEMV on it, and
/// compares against a NumPy reference produced from the same bytes. This is the
/// first point where our understanding of the format meets OpenAI's actual
/// data: nibble order, scale alignment, row-major layout, and the E8M0 exponent
/// are all assumptions until this passes.
public struct ExpertVerification {
    public struct Report: Sendable {
        public let rows: Int
        public let cols: Int
        public let bytesRead: Int
        public let readSeconds: Double
        public let computeSeconds: Double
        public let maxRelativeError: Float
        public let meanRelativeError: Float
        public let worstRow: Int
        public let referenceMagnitude: Float

        public var passed: Bool { maxRelativeError < 1e-2 }
    }

    public let context: MetalContext

    public init(context: MetalContext) {
        self.context = context
    }

    public func run(directory: URL) throws -> Report {
        let reader = try ExpertBlobReader(directory: directory)
        let manifest = reader.manifest

        let readStart = CFAbsoluteTimeGetCurrent()
        let blocks = try reader.load(section: "gate_up_blocks", device: context.device)
        let scales = try reader.load(section: "gate_up_scales", device: context.device)
        let readSeconds = CFAbsoluteTimeGetCurrent() - readStart

        let rows = manifest.rows
        let cols = manifest.cols

        let x = try loadFloat16(directory.appendingPathComponent(manifest.reference?.x ?? "x.f16"),
                                count: cols)
        let expected = try loadFloat32(directory.appendingPathComponent(manifest.reference?.y ?? "y.f32"),
                                       count: rows)

        let xBuffer = try context.buffer(x)
        let yBuffer = try context.emptyBuffer(of: Float.self, count: rows)
        var colsValue = UInt32(cols)

        let pipeline = try context.pipeline(shader: "dequant_gemv", function: "mxfp4_gemv")
        guard let commands = context.queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder()
        else { throw MetalError.encoderCreationFailed }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(blocks, offset: 0, index: 0)
        encoder.setBuffer(scales, offset: 0, index: 1)
        encoder.setBuffer(xBuffer, offset: 0, index: 2)
        encoder.setBuffer(yBuffer, offset: 0, index: 3)
        encoder.setBytes(&colsValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()

        let computeStart = CFAbsoluteTimeGetCurrent()
        commands.commit()
        commands.waitUntilCompleted()
        let computeSeconds = CFAbsoluteTimeGetCurrent() - computeStart

        let produced = yBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        // Normalise by the reference's typical magnitude rather than per-element:
        // a near-zero output row would otherwise report enormous relative error
        // for a numerically irrelevant difference.
        var magnitude: Float = 0
        for row in 0..<rows { magnitude += abs(expected[row]) }
        magnitude = max(magnitude / Float(rows), 1e-6)

        var worst: Float = 0
        var worstRow = 0
        var total: Float = 0
        for row in 0..<rows {
            let error = abs(produced[row] - expected[row]) / magnitude
            total += error
            if error > worst { worst = error; worstRow = row }
        }

        let bytesRead = (manifest.sections["gate_up_blocks"]?.length ?? 0)
            + (manifest.sections["gate_up_scales"]?.length ?? 0)

        return Report(rows: rows, cols: cols,
                      bytesRead: bytesRead,
                      readSeconds: readSeconds,
                      computeSeconds: computeSeconds,
                      maxRelativeError: worst,
                      meanRelativeError: total / Float(rows),
                      worstRow: worstRow,
                      referenceMagnitude: magnitude)
    }

    private func loadFloat16(_ url: URL, count: Int) throws -> [Float16] {
        let data = try Data(contentsOf: url)
        precondition(data.count >= count * 2, "\(url.lastPathComponent) is short")
        return data.withUnsafeBytes { raw in
            Array(UnsafeBufferPointer(start: raw.bindMemory(to: Float16.self).baseAddress!,
                                      count: count))
        }
    }

    private func loadFloat32(_ url: URL, count: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        precondition(data.count >= count * 4, "\(url.lastPathComponent) is short")
        return data.withUnsafeBytes { raw in
            Array(UnsafeBufferPointer(start: raw.bindMemory(to: Float.self).baseAddress!,
                                      count: count))
        }
    }
}
