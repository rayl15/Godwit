import Metal
import Testing

@testable import Godwit

/// The expert activation kernels, against CPU references.
///
/// These exist because of a specific failure: `geluTanh` was declared in
/// `FeedForwardActivation` and routed to the SwiGLU kernel, so a model
/// declaring GeGLU would have run without complaint and computed the wrong
/// function. Nothing caught it — there was no model using it yet, and a wrong
/// activation produces plausible numbers rather than an error.
///
/// The same shape of bug has now cost this project time three times: missing
/// attention biases, a QK-norm weight read as `half` instead of `bfloat`, and
/// this. All three ran fine and were simply wrong.
@Suite("Expert activations", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct ActivationTests {
    /// gate and up interleaved, as the kernels and the .gwt layout expect.
    private func interleave(gate: [Float], up: [Float]) -> [Float] {
        var out = [Float]()
        for (g, u) in zip(gate, up) { out.append(g); out.append(u) }
        return out
    }

    private func run(_ function: String, gate: [Float], up: [Float]) throws -> [Float] {
        let context = try MetalContext()
        let pipeline = try context.pipeline(shader: "expert", function: function)
        let input = try context.buffer(interleave(gate: gate, up: up))
        let output = try context.emptyBuffer(of: Float16.self, count: gate.count)

        guard let commands = context.queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var width = UInt32(gate.count)
        encoder.setBytes(&width, length: 4, index: 2)
        // The GPT-OSS kernel reads two more constants; harmless for the others.
        var limit: Float = 7.0, alpha: Float = 1.702
        encoder.setBytes(&limit, length: 4, index: 3)
        encoder.setBytes(&alpha, length: 4, index: 4)
        encoder.dispatchThreads(MTLSize(width: gate.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()

        let raw = output.contents().bindMemory(to: Float16.self, capacity: gate.count)
        return (0..<gate.count).map { Float(raw[$0]) }
    }

    private func geluTanh(_ x: Float) -> Float {
        let k: Float = 0.7978845608028654          // sqrt(2/pi)
        return 0.5 * x * (1 + tanh(k * (x + 0.044715 * x * x * x)))
    }

    private func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }

    private let gate: [Float] = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, -8, 0.1]
    private let up: [Float] = [1, -1, 2, 0.5, 3, -2, 1, 0.25, -1, 2, 1, -3]

    @Test("GeGLU matches the tanh-approximated GELU, not SiLU")
    func gegluMatchesReference() throws {
        let got = try run("expert_activation_geglu", gate: gate, up: up)
        for (index, value) in got.enumerated() {
            let want = geluTanh(gate[index]) * up[index]
            #expect(abs(value - want) < 0.01,
                    "index \(index): got \(value), want \(want)")
        }
    }

    @Test("SwiGLU matches SiLU")
    func swigluMatchesReference() throws {
        let got = try run("expert_activation_swiglu", gate: gate, up: up)
        for (index, value) in got.enumerated() {
            let want = silu(gate[index]) * up[index]
            #expect(abs(value - want) < 0.01,
                    "index \(index): got \(value), want \(want)")
        }
    }

    /// The regression that motivated all of this. GeGLU and SwiGLU are close
    /// enough to look right and far enough apart to be wrong: at gate = 2 the
    /// gates differ by about 8%, which is invisible in generated text and
    /// compounds over thirty layers.
    @Test("GeGLU and SwiGLU are not the same function")
    func gegluIsNotSwiglu() throws {
        let geglu = try run("expert_activation_geglu", gate: gate, up: up)
        let swiglu = try run("expert_activation_swiglu", gate: gate, up: up)
        let differing = zip(geglu, swiglu).filter { abs($0 - $1) > 0.01 }.count
        #expect(differing >= gate.count / 2,
                "near-identical output: geluTanh is probably back on SwiGLU")
    }

    /// Each activation must name a distinct kernel. A new case added to the
    /// enum without a kernel would otherwise reuse whichever branch it fell to.
    @Test("Every activation maps to its own kernel, and that kernel exists")
    func activationsMapDistinctly() throws {
        let context = try MetalContext()
        let kernels = FeedForwardActivation.allCases.map(\.kernel)
        #expect(Set(kernels).count == kernels.count,
                "two activations share a kernel")
        for kernel in kernels {
            _ = try context.pipeline(shader: "expert", function: kernel)
        }
    }
}
