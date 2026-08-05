import Foundation

/// Reference decoder for the OCP Microscaling `MXFP4` block format.
///
/// A block is 32 weights sharing one 8-bit exponent:
///
/// - each weight is a 4-bit `E2M1` float (1 sign, 2 exponent, 1 mantissa),
///   packed two-per-byte, low nibble first;
/// - the block scale is `E8M0`, a raw power of two with a 127 bias, so the
///   decoded weight is `e2m1Value * pow(2, scaleByte - 127)`.
///
/// `E8M0` reserves `0xFF` for NaN and has no zero encoding, which is the one
/// place this format bites: a scale byte of 0 means `2^-127`, not 0.
///
/// This is the CPU ground truth. It exists to validate the Metal kernel, not to
/// be fast — every kernel change is checked against it bit for bit.
public enum MXFP4 {
    /// Weights encodable in `E2M1`, indexed by the 4-bit code.
    ///
    /// The exponent field is 2 bits with a bias of 1, and the format has no
    /// infinities, so the representable magnitudes are exactly these eight.
    public static let codebook: [Float] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]

    /// Weights per block, fixed by the OCP specification.
    public static let blockSize = 32

    /// Packed bytes per block: 32 weights at two per byte.
    public static let packedBytesPerBlock = blockSize / 2

    /// Decodes an `E8M0` scale byte to its multiplier.
    ///
    /// Returns NaN for `0xFF`, which the specification reserves.
    @inlinable
    public static func decodeScale(_ byte: UInt8) -> Float {
        guard byte != 0xFF else { return .nan }
        return exp2(Float(Int(byte) - 127))
    }

    /// Decodes one block of 32 weights.
    ///
    /// - Parameters:
    ///   - packed: exactly `packedBytesPerBlock` bytes of nibble-packed codes.
    ///   - scaleByte: the block's `E8M0` shared exponent.
    ///   - destination: receives `blockSize` decoded weights.
    public static func decodeBlock(
        packed: UnsafeRawBufferPointer,
        scaleByte: UInt8,
        into destination: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(packed.count >= packedBytesPerBlock, "block needs \(packedBytesPerBlock) packed bytes")
        precondition(destination.count >= blockSize, "block decodes to \(blockSize) weights")

        let scale = decodeScale(scaleByte)
        for byteIndex in 0..<packedBytesPerBlock {
            let byte = packed[byteIndex]
            // Low nibble holds the even-indexed weight.
            destination[byteIndex * 2] = codebook[Int(byte & 0x0F)] * scale
            destination[byteIndex * 2 + 1] = codebook[Int(byte >> 4)] * scale
        }
    }

    /// Decodes a contiguous run of blocks into a newly allocated array.
    ///
    /// `packed` and `scales` are the two parallel streams the format defines:
    /// one packed-nibble stream and one scale byte per block.
    public static func decode(
        packed: UnsafeRawBufferPointer,
        scales: UnsafeRawBufferPointer,
        blockCount: Int
    ) -> [Float] {
        precondition(
            packed.count >= blockCount * packedBytesPerBlock,
            "packed stream is short for \(blockCount) blocks")
        precondition(scales.count >= blockCount, "need one scale byte per block")

        var weights = [Float](repeating: 0, count: blockCount * blockSize)
        weights.withUnsafeMutableBufferPointer { output in
            for block in 0..<blockCount {
                let packedStart = block * packedBytesPerBlock
                let slice = UnsafeRawBufferPointer(
                    rebasing: packed[packedStart..<(packedStart + packedBytesPerBlock)])
                let outputStart = block * blockSize
                let destination = UnsafeMutableBufferPointer(
                    rebasing: output[outputStart..<(outputStart + blockSize)])
                decodeBlock(packed: slice, scaleByte: scales[block], into: destination)
            }
        }
        return weights
    }
}

// MARK: - Encoding

extension MXFP4 {
    /// The eight representable magnitudes, in code order. Sign lives in bit 3.
    static let magnitudes: [Float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    /// Midpoints between adjacent magnitudes, for round-to-nearest.
    ///
    /// A lookup against these is a branchless alternative to searching the
    /// codebook, and makes the tie behaviour explicit rather than a property of
    /// whichever comparison happened to run first.
    @usableFromInline static let thresholds: [Float] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

    /// Chooses a block's `E8M0` scale byte.
    ///
    /// The exponent is the smallest power of two that brings the block's
    /// largest magnitude to 6.0 or below — `ceil(log2(peak / 6))`, not `floor`.
    /// Flooring puts the scale under `peak/6`, so the largest weight lands above
    /// 6.0 and clips to it, up to half its true size, on precisely the element
    /// the scale exists to protect.
    ///
    /// This matches what OpenAI's own MXFP4 weights use: decoding GPT-OSS's
    /// experts and re-encoding them here reproduces every scale byte exactly.
    public static func scaleByte(forPeak peak: Float) -> UInt8 {
        guard peak > 0, peak.isFinite else { return 127 }   // 2^0
        let exponent = (peak / 6.0).binaryLogarithm.rounded(.up)
        // 0xFF is reserved for NaN, so the usable exponent tops out at 126.
        return UInt8(clamping: Int(exponent) + 127 == 255 ? 254 : Int(exponent) + 127)
    }

    /// Encodes one magnitude to its 3-bit code, rounding to nearest.
    @inlinable
    public static func code(forMagnitude value: Float) -> UInt8 {
        var code: UInt8 = 0
        for threshold in thresholds {
            if value < threshold { return code }
            code += 1
        }
        return code
    }

    /// Encodes one block of at most 32 weights.
    ///
    /// A short final block is zero-filled, which is what the reader expects:
    /// rows are padded to a whole number of blocks rather than truncated.
    ///
    /// - Returns: the block's `E8M0` scale byte.
    @discardableResult
    public static func encodeBlock(
        weights: UnsafeBufferPointer<Float>,
        into packed: UnsafeMutableRawBufferPointer
    ) -> UInt8 {
        precondition(packed.count >= packedBytesPerBlock,
                     "block needs \(packedBytesPerBlock) packed bytes")

        var peak: Float = 0
        for weight in weights where weight.isFinite {
            peak = max(peak, abs(weight))
        }
        let scaleByte = scaleByte(forPeak: peak)
        let inverse = 1.0 / decodeScale(scaleByte)

        for byteIndex in 0..<packedBytesPerBlock {
            var byte: UInt8 = 0
            for half in 0..<2 {
                let index = byteIndex * 2 + half
                guard index < weights.count else { continue }
                let weight = weights[index]
                guard weight.isFinite else { continue }
                let scaled = abs(weight) * inverse
                var nibble = code(forMagnitude: scaled)
                // Sign is bit 3, and it is kept even when the magnitude rounds
                // to zero. A small negative weight becomes code 8, negative
                // zero — which is what the shipped GPT-OSS weights contain, and
                // dropping it made 15% of re-encoded nibbles disagree with
                // OpenAI's own bytes. Arithmetically -0.0 is harmless: it
                // multiplies to -0.0 and adds as 0.
                //
                // `weight < 0` is false for -0.0, so the sign bit is read
                // directly.
                if weight.sign == .minus { nibble |= 0x08 }
                // Low nibble holds the even-indexed weight, matching decode.
                byte |= half == 0 ? nibble : (nibble << 4)
            }
            packed[byteIndex] = byte
        }
        return scaleByte
    }

    /// Encodes a contiguous run of weights into the format's two streams.
    ///
    /// The count need not be a multiple of `blockSize`; the last block is
    /// padded with zeros.
    public static func encode(_ weights: [Float]) -> (packed: [UInt8], scales: [UInt8]) {
        let blockCount = (weights.count + blockSize - 1) / blockSize
        var packed = [UInt8](repeating: 0, count: blockCount * packedBytesPerBlock)
        var scales = [UInt8](repeating: 127, count: blockCount)

        weights.withUnsafeBufferPointer { source in
            packed.withUnsafeMutableBytes { destination in
                for block in 0..<blockCount {
                    let start = block * blockSize
                    let end = min(start + blockSize, weights.count)
                    let slice = UnsafeBufferPointer(rebasing: source[start..<end])
                    let byteStart = block * packedBytesPerBlock
                    let target = UnsafeMutableRawBufferPointer(
                        rebasing: destination[byteStart..<(byteStart + packedBytesPerBlock)])
                    scales[block] = encodeBlock(weights: slice, into: target)
                }
            }
        }
        return (packed, scales)
    }
}

private extension Float {
    /// `log2`, named to avoid colliding with the global function inside a type
    /// that also has a `scale` concept.
    var binaryLogarithm: Float { Foundation.log2(self) }
}
