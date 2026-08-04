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
