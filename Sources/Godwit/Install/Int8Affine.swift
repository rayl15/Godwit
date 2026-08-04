import Foundation

/// Affine int8 quantisation with per-group scale and zero point.
///
/// Used for the trunk — embeddings, output head, attention — which GPT-OSS
/// ships in BF16 and which we cannot afford to keep at that width.
///
/// 8 bits rather than 4 is a measured decision, not a conservative default. At
/// 4 bits the output head changes its top-1 prediction 19% of the time; at 8
/// bits it agrees 98.4% of the time. See `Scripts/analysis/requant_quality.py`
/// and docs/ESTIMATE.md.
public enum Int8Affine {
    /// Weights per scale/zero pair. Matches the MXFP4 side closely enough that
    /// kernels can share their addressing patterns.
    public static let groupSize = 64

    public static let levels: Float = 255

    /// Bits per weight including metadata: 8 for the code, plus a BF16 scale and
    /// BF16 zero shared across 64 weights.
    public static var bitsPerWeight: Double { 8 + 32.0 / Double(groupSize) }

    public struct Group: Sendable, Equatable {
        public let scale: Float
        public let zero: Float
    }

    /// Quantises one row. `row.count` must be a multiple of `groupSize`.
    ///
    /// Returns codes alongside the per-group scale and zero needed to invert it.
    public static func quantize(row: [Float]) -> (codes: [UInt8], groups: [Group]) {
        precondition(row.count % groupSize == 0,
                     "row width \(row.count) is not a multiple of \(groupSize)")
        let groupCount = row.count / groupSize
        var codes = [UInt8](repeating: 0, count: row.count)
        var groups = [Group]()
        groups.reserveCapacity(groupCount)

        for group in 0..<groupCount {
            let base = group * groupSize
            var low = row[base]
            var high = row[base]
            for i in 1..<groupSize {
                low = min(low, row[base + i])
                high = max(high, row[base + i])
            }
            // A constant group would give a zero scale and divide by zero on the
            // way back out; clamping keeps the inverse well defined.
            let scale = max((high - low) / levels, .leastNormalMagnitude)
            groups.append(Group(scale: scale, zero: low))

            for i in 0..<groupSize {
                let level = ((row[base + i] - low) / scale).rounded()
                codes[base + i] = UInt8(max(0, min(levels, level)))
            }
        }
        return (codes, groups)
    }

    /// Inverts `quantize`, for tests and CPU reference paths.
    public static func dequantize(codes: [UInt8], groups: [Group]) -> [Float] {
        var out = [Float](repeating: 0, count: codes.count)
        for group in groups.indices {
            let base = group * groupSize
            let scale = groups[group].scale
            let zero = groups[group].zero
            for i in 0..<groupSize {
                out[base + i] = Float(codes[base + i]) * scale + zero
            }
        }
        return out
    }
}

/// BF16 is the top 16 bits of an IEEE float, so conversion is a shift in both
/// directions. Round-to-nearest-even on the way down matters: truncation biases
/// every value toward zero, and that bias accumulates across a 2880-wide dot
/// product.
public enum BFloat16 {
    @inlinable
    public static func toFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    @inlinable
    public static func fromFloat(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
            return UInt16(truncatingIfNeeded: bits >> 16) | 0x0040   // keep NaN a NaN
        }
        let rounding = ((bits >> 16) & 1) &+ 0x7FFF
        return UInt16(truncatingIfNeeded: (bits &+ rounding) >> 16)
    }

    /// Widens a little-endian BF16 buffer to `Float`.
    public static func decode(_ raw: UnsafeRawBufferPointer, count: Int) -> [Float] {
        precondition(raw.count >= count * 2, "buffer holds fewer than \(count) bf16 values")
        return (0..<count).map { index in
            toFloat(raw.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self))
        }
    }
}
