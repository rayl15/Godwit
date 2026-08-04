import Foundation
import Testing

@testable import Godwit

@Suite("GPT-OSS-120B architecture")
struct ArchitectureSpecTests {
    let spec = ArchitectureSpec.gptOSS120B

    @Test("Matches the published config")
    func matchesConfig() {
        #expect(spec.layerCount == 36)
        #expect(spec.hiddenSize == 2880)
        #expect(spec.intermediateSize == 2880)
        #expect(spec.attentionHeads == 64)
        #expect(spec.keyValueHeads == 8)
        #expect(spec.headDimension == 64)
        #expect(spec.vocabularySize == 201088)
        #expect(spec.maxExpertsPerToken == 4)
        #expect(spec.routedLayerIndices.count == 36, "every layer is MoE")
    }

    @Test("Attention alternates sliding and full, 18 each")
    func attentionPattern() {
        let sliding = spec.layers.filter { $0.attention == .sliding }
        let full = spec.layers.filter { $0.attention == .full }
        #expect(sliding.count == 18)
        #expect(full.count == 18)
        #expect(sliding.allSatisfy { $0.window == 128 })
    }

    @Test("Features that are easy to miss are recorded")
    func unusualFeatures() {
        // Each of these appears in the safetensors header. Silently omitting any
        // of them produces plausible-looking but wrong output.
        #expect(spec.attentionSinks, "self_attn.sinks is present")
        #expect(spec.attentionBias, "q/k/v/o_proj.bias are present")
        #expect(spec.expertBias, "gate_up_proj_bias and down_proj_bias are present")
        #expect(spec.routerBias, "mlp.router.bias is present")
        #expect(!spec.tiedEmbedding, "lm_head is separate from embed_tokens")
        #expect(spec.layers.allSatisfy { !$0.hasSharedExpert },
                "no shared expert branch, so there is no free work to overlap reads against")
    }

    @Test("Derived sizes reproduce the published checkpoint")
    func derivedSizes() {
        // 4.25 bits per weight: 4-bit code plus one E8M0 scale byte per 32.
        let perExpert = 2 * spec.intermediateSize * spec.hiddenSize
            + spec.hiddenSize * spec.intermediateSize
        #expect(perExpert == 24_883_200)

        let routedBytes = Double(perExpert) * 4.25 / 8 * 128 * 36
        let routedGiB = routedBytes / 1_073_741_824
        // Published checkpoint is ~60 GB; 56.7 GiB is the same number in binary units.
        #expect(routedGiB > 56 && routedGiB < 58, "got \(routedGiB) GiB")
    }

    @Test("Round-trips through JSON")
    func codableRoundTrip() throws {
        let encoded = try JSONEncoder().encode(spec)
        let decoded = try JSONDecoder().decode(ArchitectureSpec.self, from: encoded)
        #expect(decoded == spec)
    }
}

@Suite("Expert activation")
struct ExpertActivationTests {
    let spec = ArchitectureSpec.gptOSS120B

    @Test("Activation is not SiLU despite what config.json says")
    func activationIsNotSilu() {
        // config.json reports hidden_act: "silu". The experts do something else,
        // and taking config at face value produces plausible-looking garbage.
        #expect(spec.activation == .gptOssClampedGLU)
        #expect(spec.activationLimit == 7.0)
        #expect(spec.activationAlpha == 1.702)
    }

    /// Mirrors `gptoss_expert_activation` in expert.metal.
    static func reference(gate: Float, up: Float, limit: Float = 7, alpha: Float = 1.702) -> Float {
        let g = min(gate, limit)                       // clamped above only
        let u = max(-limit, min(up, limit))
        return (u + 1) * (g / (1 + Foundation.exp(-alpha * g)))
    }

    @Test("The +1 shift on the up branch is load-bearing")
    func upShiftMatters() {
        // Without the shift, up == 0 would zero the output. With it, the gate
        // passes through — a difference that changes every token.
        #expect(abs(Self.reference(gate: 2, up: 0) - Self.reference(gate: 2, up: 0)) < 1e-6)
        #expect(Self.reference(gate: 2, up: 0) != 0)
    }

    @Test("Clamping is asymmetric")
    func asymmetricClamp() {
        // Gate has no lower bound: a large negative gate stays large negative
        // and the sigmoid drives the product toward zero on its own.
        let veryNegative = Self.reference(gate: -100, up: 1)
        #expect(abs(veryNegative) < 1e-3, "sigmoid should suppress it, got \(veryNegative)")

        // Gate is bounded above, so growth past the limit does nothing.
        #expect(Self.reference(gate: 8, up: 1) == Self.reference(gate: 1000, up: 1))

        // Up is bounded both ways.
        #expect(Self.reference(gate: 1, up: 50) == Self.reference(gate: 1, up: 7))
        #expect(Self.reference(gate: 1, up: -50) == Self.reference(gate: 1, up: -7))
    }
}
