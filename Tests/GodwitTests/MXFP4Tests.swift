import Testing

@testable import Godwit

@Suite("MXFP4 reference decode")
struct MXFP4Tests {
    @Test("E2M1 codebook covers the representable magnitudes")
    func codebookShape() {
        #expect(MXFP4.codebook.count == 16)
        #expect(MXFP4.codebook[0] == 0.0)
        #expect(MXFP4.codebook[7] == 6.0, "largest E2M1 magnitude is 6")
        // Codes 8...15 mirror 0...7 with the sign bit set.
        for code in 0..<8 {
            #expect(MXFP4.codebook[code + 8] == -MXFP4.codebook[code])
        }
    }

    @Test("E8M0 scale is a biased power of two")
    func scaleDecoding() {
        #expect(MXFP4.decodeScale(127) == 1.0)
        #expect(MXFP4.decodeScale(128) == 2.0)
        #expect(MXFP4.decodeScale(126) == 0.5)
        #expect(MXFP4.decodeScale(0xFF).isNaN, "0xFF is reserved for NaN")
        // There is no zero encoding: byte 0 is a very small scale, not zero.
        #expect(MXFP4.decodeScale(0) > 0)
    }

    @Test("Nibbles unpack low-first")
    func nibbleOrder() {
        // 0x71 -> low nibble 1 (0.5), high nibble 7 (6.0).
        var packed = [UInt8](repeating: 0, count: MXFP4.packedBytesPerBlock)
        packed[0] = 0x71
        var decoded = [Float](repeating: .nan, count: MXFP4.blockSize)

        packed.withUnsafeBytes { raw in
            decoded.withUnsafeMutableBufferPointer { output in
                MXFP4.decodeBlock(packed: raw, scaleByte: 127, into: output)
            }
        }

        #expect(decoded[0] == 0.5)
        #expect(decoded[1] == 6.0)
    }

    @Test("Scale multiplies the whole block")
    func scaleApplies() {
        var packed = [UInt8](repeating: 0x22, count: MXFP4.packedBytesPerBlock) // code 2 = 1.0
        packed[0] = 0x22
        var decoded = [Float](repeating: .nan, count: MXFP4.blockSize)

        packed.withUnsafeBytes { raw in
            decoded.withUnsafeMutableBufferPointer { output in
                MXFP4.decodeBlock(packed: raw, scaleByte: 130, into: output) // 2^3 = 8
            }
        }

        #expect(decoded.allSatisfy { $0 == 8.0 })
    }

    @Test("Multi-block decode keeps blocks independent")
    func multiBlockDecode() {
        let blocks = 3
        let packed = [UInt8](repeating: 0x22, count: blocks * MXFP4.packedBytesPerBlock)
        let scales: [UInt8] = [127, 128, 126] // 1, 2, 0.5

        let decoded = packed.withUnsafeBytes { packedRaw in
            scales.withUnsafeBytes { scaleRaw in
                MXFP4.decode(packed: packedRaw, scales: scaleRaw, blockCount: blocks)
            }
        }

        #expect(decoded.count == blocks * MXFP4.blockSize)
        #expect(decoded[0] == 1.0)
        #expect(decoded[MXFP4.blockSize] == 2.0)
        #expect(decoded[MXFP4.blockSize * 2] == 0.5)
    }
}
