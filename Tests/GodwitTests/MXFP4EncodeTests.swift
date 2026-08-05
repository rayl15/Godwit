import Foundation
import Testing

@testable import Godwit

@Suite("MXFP4 encoding")
struct MXFP4EncodeTests {
    /// Encode-then-decode must be the identity on values already on the grid.
    ///
    /// This is the property that makes the encoder checkable against OpenAI's
    /// own weights: GPT-OSS ships MXFP4, so decoding and re-encoding it has to
    /// come back bit-identical or the encoder disagrees with the format.
    @Test("Values already representable survive a round trip exactly")
    func gridValuesAreFixedPoints() {
        // One block per exponent. A block shares a single exponent, so mixing
        // scales within one is not a round-trip failure but the format working
        // as designed — a 0.03 next to a 6144 has nowhere to go.
        var weights: [Float] = []
        for exponent in [-4, -1, 0, 3, 10] {
            let scale = exp2(Float(exponent))
            var block: [Float] = []
            for magnitude in MXFP4.magnitudes {
                block.append(magnitude * scale)
                block.append(-magnitude * scale)
            }
            while block.count < MXFP4.blockSize { block.append(0) }
            weights += block
        }

        let (packed, scales) = MXFP4.encode(weights)
        let decoded = packed.withUnsafeBytes { packedBytes in
            scales.withUnsafeBytes { scaleBytes in
                MXFP4.decode(packed: packedBytes, scales: scaleBytes,
                             blockCount: scales.count)
            }
        }
        for (index, original) in weights.enumerated() {
            #expect(decoded[index] == original,
                    "index \(index): \(original) became \(decoded[index])")
        }
    }

    @Test("The scale ceils rather than floors, so the peak never clips")
    func scaleNeverClipsThePeak() {
        // A peak of 10 wants a scale of 2: 10/2 = 5, which is representable.
        // Flooring would pick 1, and 10 would clip to 6 — 40% low.
        #expect(MXFP4.decodeScale(MXFP4.scaleByte(forPeak: 10)) == 2.0)
        #expect(MXFP4.decodeScale(MXFP4.scaleByte(forPeak: 6)) == 1.0)
        #expect(MXFP4.decodeScale(MXFP4.scaleByte(forPeak: 3)) == 0.5)

        // Across a wide sweep of peaks, the block maximum must land inside the
        // representable range rather than on the clamp.
        for step in 0..<200 {
            let peak = exp2(Float(step) / 8 - 12)
            let scale = MXFP4.decodeScale(MXFP4.scaleByte(forPeak: peak))
            #expect(peak / scale <= 6.0 + 1e-6, "peak \(peak) clips at scale \(scale)")
        }
    }

    @Test("Rounding goes to the nearest representable magnitude")
    func roundsToNearest() {
        #expect(MXFP4.code(forMagnitude: 0.0) == 0)      // 0.0
        #expect(MXFP4.code(forMagnitude: 0.24) == 0)     // below the 0.25 midpoint
        #expect(MXFP4.code(forMagnitude: 0.26) == 1)     // 0.5
        #expect(MXFP4.code(forMagnitude: 0.9) == 2)      // 1.0
        #expect(MXFP4.code(forMagnitude: 1.4) == 3)      // 1.5
        #expect(MXFP4.code(forMagnitude: 2.6) == 5)      // 3.0
        #expect(MXFP4.code(forMagnitude: 4.9) == 6)      // 4.0
        #expect(MXFP4.code(forMagnitude: 5.1) == 7)      // 6.0
        #expect(MXFP4.code(forMagnitude: 99) == 7)       // saturates
    }

    @Test("Sign is carried, and zero stays positive")
    func signHandling() {
        let weights: [Float] = [1.0, -1.0, 0.0, -0.0] + [Float](repeating: 0, count: 28)
        let (packed, scales) = MXFP4.encode(weights)
        let decoded = packed.withUnsafeBytes { p in
            scales.withUnsafeBytes { s in
                MXFP4.decode(packed: p, scales: s, blockCount: 1)
            }
        }
        #expect(decoded[0] == 1.0)
        #expect(decoded[1] == -1.0)
        // The sign survives a magnitude that rounds to zero, because that is
        // what the format does and what OpenAI's shipped weights contain.
        // Both still compare equal to zero.
        #expect(decoded[2].sign == .plus)
        #expect(decoded[3].sign == .minus)
        #expect(decoded[2] == 0.0)
        #expect(decoded[3] == 0.0)
    }

    @Test("A short final block is padded, not truncated")
    func shortBlockPads() {
        let weights: [Float] = [1.0, 2.0, 3.0]
        let (packed, scales) = MXFP4.encode(weights)
        #expect(scales.count == 1)
        #expect(packed.count == MXFP4.packedBytesPerBlock)
        let decoded = packed.withUnsafeBytes { p in
            scales.withUnsafeBytes { s in
                MXFP4.decode(packed: p, scales: s, blockCount: 1)
            }
        }
        #expect(Array(decoded[0..<3]) == [1.0, 2.0, 3.0])
        #expect(decoded[3..<32].allSatisfy { $0 == 0 })
    }

    @Test("An all-zero block encodes to zeros rather than a denormal scale")
    func zeroBlock() {
        let (packed, scales) = MXFP4.encode([Float](repeating: 0, count: 32))
        #expect(packed.allSatisfy { $0 == 0 })
        // E8M0 has no zero, so a scale byte of 0 would mean 2^-127, not 0.
        #expect(scales[0] == 127)
    }

    @Test("Quantisation error stays within the format's bound")
    func errorBound() {
        // Within a block the worst relative step is between 4.0 and 6.0, so no
        // weight should move by more than half that gap times the scale.
        var generator = SystemRandomNumberGenerator()
        var weights: [Float] = []
        for _ in 0..<(MXFP4.blockSize * 64) {
            weights.append(Float.random(in: -3...3, using: &generator))
        }
        let (packed, scales) = MXFP4.encode(weights)
        let decoded = packed.withUnsafeBytes { p in
            scales.withUnsafeBytes { s in
                MXFP4.decode(packed: p, scales: s, blockCount: scales.count)
            }
        }
        for block in 0..<scales.count {
            let scale = MXFP4.decodeScale(scales[block])
            for offset in 0..<MXFP4.blockSize {
                let index = block * MXFP4.blockSize + offset
                let error = abs(decoded[index] - weights[index])
                #expect(error <= scale * 1.0 + 1e-6,
                        "index \(index) moved by \(error) at scale \(scale)")
            }
        }
    }
}
