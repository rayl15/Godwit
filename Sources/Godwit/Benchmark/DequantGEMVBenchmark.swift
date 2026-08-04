import Foundation
import Metal

/// Measures fused dequantise-and-multiply throughput for MXFP4 against
/// MLX-style affine int4 at identical shape.
///
/// This answers the project's go/no-go question. Decode has a per-token budget:
/// every token multiplies `topK * layers * paramsPerExpert` quantised weights,
/// and that must finish inside the time expert reads take. If MXFP4 cannot hit
/// the required rate, the target model family is off the table.
public struct DequantGEMVBenchmark {
    public struct Result: Sendable {
        public let label: String
        public let rows: Int
        public let cols: Int
        public let iterations: Int
        public let seconds: Double
        /// Quantised weights consumed per second.
        public let weightsPerSecond: Double
        /// Effective read bandwidth over the packed representation.
        public let gibPerSecond: Double
        /// Largest absolute difference from the CPU reference, or nil if unchecked.
        public let maxAbsoluteError: Float?
    }

    public let context: MetalContext

    public init(context: MetalContext) {
        self.context = context
    }

    // MARK: - Deterministic inputs

    /// A small xorshift so runs are reproducible without a seeded RNG dependency.
    private static func pseudoRandomBytes(count: Int, seed: UInt32) -> [UInt8] {
        var state = seed | 1
        return (0..<count).map { _ in
            state ^= state << 13
            state ^= state >> 17
            state ^= state << 5
            return UInt8(truncatingIfNeeded: state >> 8)
        }
    }

    private static func activations(count: Int, seed: UInt32) -> [Float16] {
        var state = seed | 1
        return (0..<count).map { _ in
            state ^= state << 13
            state ^= state >> 17
            state ^= state << 5
            // Keep values small so FP16 accumulation stays well conditioned.
            return Float16(Float(Int32(bitPattern: state) % 200) / 200.0)
        }
    }

    // MARK: - MXFP4

    public func runMXFP4(rows: Int, cols: Int, iterations: Int, validate: Bool) throws -> Result {
        precondition(cols % 32 == 0, "MXFP4 needs cols to be a multiple of 32")
        let blocksPerRow = cols / 32
        let packed = Self.pseudoRandomBytes(count: rows * cols / 2, seed: 0xA5A5)
        // Keep exponents near 1.0 so the reference and GPU agree to FP32 precision.
        let scales = (0..<(rows * blocksPerRow)).map { UInt8(120 + ($0 % 14)) }
        let x = Self.activations(count: cols, seed: 0x1234)

        let packedBuffer = try context.buffer(packed)
        let scaleBuffer = try context.buffer(scales)
        let xBuffer = try context.buffer(x)
        let yBuffer = try context.emptyBuffer(of: Float.self, count: rows)
        var colsValue = UInt32(cols)

        let pipeline = try context.pipeline(shader: "dequant_gemv", function: "mxfp4_gemv")

        let seconds = try time(iterations: iterations) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(packedBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(xBuffer, offset: 0, index: 2)
            encoder.setBuffer(yBuffer, offset: 0, index: 3)
            encoder.setBytes(&colsValue, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }

        var error: Float?
        if validate {
            error = Self.compareToReference(
                gpu: yBuffer, rows: rows, cols: cols,
                reference: Self.referenceMXFP4(packed: packed, scales: scales, x: x,
                                               rows: rows, cols: cols))
        }

        return makeResult(label: "MXFP4 (block 32, E8M0)", rows: rows, cols: cols,
                          iterations: iterations, seconds: seconds,
                          bitsPerWeight: 4.25, error: error)
    }

    /// CPU ground truth for `mxfp4_gemv`.
    static func referenceMXFP4(
        packed: [UInt8], scales: [UInt8], x: [Float16], rows: Int, cols: Int
    ) -> [Float] {
        let blocksPerRow = cols / 32
        return (0..<rows).map { row in
            var acc = 0.0 as Float
            for block in 0..<blocksPerRow {
                let scale = MXFP4.decodeScale(scales[row * blocksPerRow + block])
                let base = row * (cols / 2) + block * 16
                let col0 = block * 32
                var partial = 0.0 as Float
                for i in 0..<16 {
                    let byte = packed[base + i]
                    partial += MXFP4.codebook[Int(byte & 0x0F)] * Float(x[col0 + i * 2])
                    partial += MXFP4.codebook[Int(byte >> 4)] * Float(x[col0 + i * 2 + 1])
                }
                acc += partial * scale
            }
            return acc
        }
    }

    // MARK: - Affine int4 baseline

    public func runAffineInt4(rows: Int, cols: Int, iterations: Int) throws -> Result {
        precondition(cols % 64 == 0, "affine int4 needs cols to be a multiple of 64")
        let groupsPerRow = cols / 64
        let packed = Self.pseudoRandomBytes(count: rows * cols / 2, seed: 0xA5A5)
        let scales = (0..<(rows * groupsPerRow)).map { _ in Float16(0.05) }
        let biases = (0..<(rows * groupsPerRow)).map { _ in Float16(-0.4) }
        let x = Self.activations(count: cols, seed: 0x1234)

        let packedBuffer = try context.buffer(packed)
        // Metal reads these as `bfloat`; Float16 shares the 2-byte stride, and the
        // benchmark only needs representative bit patterns, not exact values.
        let scaleBuffer = try context.buffer(scales)
        let biasBuffer = try context.buffer(biases)
        let xBuffer = try context.buffer(x)
        let yBuffer = try context.emptyBuffer(of: Float.self, count: rows)
        var colsValue = UInt32(cols)

        let pipeline = try context.pipeline(shader: "dequant_gemv", function: "affine_int4_gemv")

        let seconds = try time(iterations: iterations) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(packedBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(biasBuffer, offset: 0, index: 2)
            encoder.setBuffer(xBuffer, offset: 0, index: 3)
            encoder.setBuffer(yBuffer, offset: 0, index: 4)
            encoder.setBytes(&colsValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }

        return makeResult(label: "affine int4 (group 64, BF16 s+b)", rows: rows, cols: cols,
                          iterations: iterations, seconds: seconds,
                          bitsPerWeight: 4.5, error: nil)
    }

    // MARK: - Plumbing

    private func time(
        iterations: Int,
        encode: (MTLComputeCommandEncoder) -> Void
    ) throws -> Double {
        func dispatchOnce() throws {
            guard let commands = context.queue.makeCommandBuffer(),
                  let encoder = commands.makeComputeCommandEncoder()
            else { throw MetalError.encoderCreationFailed }
            encode(encoder)
            encoder.endEncoding()
            commands.commit()
            commands.waitUntilCompleted()
        }

        // One untimed warm-up: the first dispatch pays pipeline warm-up and
        // first-touch page costs that would otherwise land in the measurement.
        try dispatchOnce()

        // A fanless Air throttles and dispatch overhead is lumpy, so a single
        // timing round has enough spread to invert a comparison. Report the
        // median of several rounds instead.
        var rounds: [Double] = []
        for _ in 0..<Self.timingRounds {
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations { try dispatchOnce() }
            rounds.append(CFAbsoluteTimeGetCurrent() - start)
        }
        rounds.sort()
        return rounds[rounds.count / 2]
    }

    /// Timing rounds per measurement; the median is reported.
    static let timingRounds = 7

    private func makeResult(
        label: String, rows: Int, cols: Int, iterations: Int,
        seconds: Double, bitsPerWeight: Double, error: Float?
    ) -> Result {
        let weights = Double(rows) * Double(cols) * Double(iterations)
        let bytes = weights * bitsPerWeight / 8
        return Result(
            label: label, rows: rows, cols: cols, iterations: iterations,
            seconds: seconds,
            weightsPerSecond: weights / seconds,
            gibPerSecond: bytes / seconds / 1_073_741_824,
            maxAbsoluteError: error)
    }

    private static func compareToReference(
        gpu: MTLBuffer, rows: Int, cols: Int, reference: [Float]
    ) -> Float {
        let produced = gpu.contents().bindMemory(to: Float.self, capacity: rows)
        var worst: Float = 0
        for row in 0..<rows {
            // Relative to row magnitude: absolute error scales with `cols`.
            let scale = max(abs(reference[row]), 1)
            worst = max(worst, abs(produced[row] - reference[row]) / scale)
        }
        return worst
    }
}
