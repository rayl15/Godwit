import Foundation
import Testing

@testable import Godwit

@Suite("RoPE with YaRN")
struct RoPETests {
    let rope = RoPE(configuration: .gptOSS)

    /// Reference values from transformers' ROPE_INIT_FUNCTIONS["yarn"] with
    /// GPT-OSS-120B's published rope_scaling block.
    /// Full float64 precision: rounding these to four significant figures makes
    /// the reference itself coarser than the tolerance, which reads as a code
    /// failure rather than a fixture problem.
    static let referenceInvFreq: [(index: Int, value: Float)] = [
        (0, 1.000000000e+00), (1, 6.890443059e-01),
        (2, 4.747820555e-01), (3, 3.271458719e-01),
        (14, 2.277271936e-03), (15, 1.206130969e-03),
        (16, 5.809475019e-04), (17, 2.279477958e-04),
        (28, 9.242089502e-07), (29, 6.368209146e-07),
        (30, 4.387978251e-07), (31, 3.023511428e-07),
    ]

    @Test("Corrected frequencies match transformers")
    func frequenciesMatchReference() {
        #expect(rope.inverseFrequencies.count == 32)
        for entry in Self.referenceInvFreq {
            let got = rope.inverseFrequencies[entry.index]
            // Float32 across a powf and a blend; 1e-5 relative is the floor here.
            let tolerance = max(abs(entry.value) * 1e-5, 1e-12)
            #expect(abs(got - entry.value) < tolerance,
                    "index \(entry.index): got \(got), want \(entry.value)")
        }
    }

    @Test("Correction range splits extrapolation from interpolation")
    func correctionRange() {
        // transformers computes low=8, high=18 for these parameters.
        let low = RoPE.correctionDimension(rotations: 32, configuration: .gptOSS)
        let high = RoPE.correctionDimension(rotations: 1, configuration: .gptOSS)
        #expect(Int(low.rounded(.down)) == 8, "got \(low)")
        #expect(Int(high.rounded(.up)) == 18, "got \(high)")
    }

    @Test("Attention factor compensates for stretched positions")
    func attentionFactor() {
        #expect(abs(rope.attentionScale - 1.346574) < 1e-5)

        // Unscaled RoPE must not pick up a gain.
        let plain = RoPE(configuration: .init(headDimension: 64, theta: 10000,
                                              scalingFactor: 1, originalContextLength: 4096,
                                              betaFast: 32, betaSlow: 1))
        #expect(plain.attentionScale == 1)
    }

    @Test("High frequencies are extrapolated, low ones interpolated")
    func rampDirection() {
        // Below the correction range: untouched, so still 1.0 at index 0.
        #expect(abs(rope.inverseFrequencies[0] - 1.0) < 1e-6)

        // Above it: divided by the full factor of 32.
        let base = 1 / powf(150_000, Float(31 * 2) / 64)
        #expect(abs(rope.inverseFrequencies[31] - base / 32) < base * 1e-4)
    }

    @Test("Rotation is NeoX half-split, not interleaved")
    func neoxLayout() {
        var vector = [Float](repeating: 0, count: 64)
        vector[0] = 1     // first half
        vector[32] = 0    // its partner, half a head away

        let rotated = rope.apply(to: vector, position: 1)
        let (cosines, sines) = rope.table(position: 1)

        // Interleaved layout would pair index 0 with index 1; NeoX pairs it
        // with index 32.
        #expect(abs(rotated[0] - cosines[0]) < 1e-6)
        #expect(abs(rotated[32] - sines[0]) < 1e-6)
        #expect(abs(rotated[1]) < 1e-6, "index 1 must be untouched by index 0")
    }

    @Test("Position zero is identity up to the attention factor")
    func positionZero() {
        let vector = (0..<64).map { Float($0) / 64 }
        let rotated = rope.apply(to: vector, position: 0)
        for i in vector.indices {
            #expect(abs(rotated[i] - vector[i] * rope.attentionScale) < 1e-5)
        }
    }
}
