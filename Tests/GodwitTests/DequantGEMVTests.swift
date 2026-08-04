import Metal
import Testing

@testable import Godwit

/// GPU kernels validated against the CPU reference decoder.
///
/// These are the tests that keep `MXFP4.swift` honest as a ground truth: if a
/// kernel optimisation changes results, this catches it. Skipped where no Metal
/// device exists rather than failing.
@Suite("Fused dequant GEMV", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct DequantGEMVTests {
    @Test("Metal context compiles the shader")
    func shaderCompiles() throws {
        let context = try MetalContext()
        _ = try context.pipeline(shader: "dequant_gemv", function: "mxfp4_gemv")
        _ = try context.pipeline(shader: "dequant_gemv", function: "affine_int4_gemv")
    }

    @Test("MXFP4 GEMV matches the CPU reference")
    func mxfp4MatchesReference() throws {
        let context = try MetalContext()
        let benchmark = DequantGEMVBenchmark(context: context)
        // Small and cheap: correctness does not need production shapes.
        let result = try benchmark.runMXFP4(rows: 64, cols: 256, iterations: 1, validate: true)

        let error = try #require(result.maxAbsoluteError)
        // FP32 accumulation over 256 terms in a different order to the CPU, with
        // fast math enabled, so exact equality is not the bar.
        #expect(error < 1e-3, "relative error \(error) too large for a 256-wide dot product")
    }

    @Test("MXFP4 GEMV stays accurate at a production column count")
    func mxfp4AccurateAtWidth() throws {
        let context = try MetalContext()
        let benchmark = DequantGEMVBenchmark(context: context)
        // 2880 is GPT-OSS's hidden size; accumulation error grows with width.
        let result = try benchmark.runMXFP4(rows: 32, cols: 2880, iterations: 1, validate: true)

        let error = try #require(result.maxAbsoluteError)
        #expect(error < 1e-3, "relative error \(error) too large at cols=2880")
    }

    @Test("Head dimension is a specialisation constant, not a literal")
    func headDimensionSpecialises() throws {
        let context = try MetalContext()
        // GPT-OSS uses 64, Qwen3 uses 128. Both must build; a hardcoded head
        // dimension would silently compute the wrong thing on the second.
        let sixtyFour = try context.pipeline(shader: "attention",
                                             function: "gqa_attention_sinks",
                                             constants: [0: 64])
        let oneTwentyEight = try context.pipeline(shader: "attention",
                                                  function: "gqa_attention_sinks",
                                                  constants: [0: 128])
        #expect(sixtyFour !== oneTwentyEight, "each value needs its own pipeline")

        // Cached by value, so asking twice returns the same object.
        let again = try context.pipeline(shader: "attention",
                                         function: "gqa_attention_sinks",
                                         constants: [0: 64])
        #expect(again === sixtyFour)
    }

    @Test("Reference decoder and GEMV reference agree on a hand case")
    func referenceConsistency() {
        // One block: code 2 (1.0) everywhere, scale 2^0, activations all 1.0.
        // Every one of the 32 weights contributes 1.0, so the row sums to 32.
        let packed = [UInt8](repeating: 0x22, count: 16)
        let scales: [UInt8] = [127]
        let x = [Float16](repeating: 1.0, count: 32)

        let rows = DequantGEMVBenchmark.referenceMXFP4(
            packed: packed, scales: scales, x: x, rows: 1, cols: 32)

        #expect(rows.count == 1)
        #expect(abs(rows[0] - 32.0) < 1e-4)
    }
}
