import Foundation
import Testing

@testable import Godwit

@Suite("Tensor naming")
struct TensorNamingTests {
    /// The failure this exists to prevent: a spec that says a model has
    /// biases while its naming has no name to fetch them from, or names a
    /// tensor the flags say does not exist. Either way the install runs for
    /// hours and produces something wrong, or dies partway through.
    @Test("Naming and flags agree, for every spec", arguments: [
        ArchitectureSpec.gptOSS120B,
        ArchitectureSpec.gptOSS20B,
        ArchitectureSpec.qwen3MoE30B,
    ])
    func namingMatchesFlags(spec: ArchitectureSpec) {
        let naming = spec.naming

        #expect((naming.queryBias != nil) == spec.attentionBias,
                "\(spec.name): attentionBias and queryBias disagree")
        #expect((naming.keyBias != nil) == spec.attentionBias)
        #expect((naming.valueBias != nil) == spec.attentionBias)
        #expect((naming.outputBias != nil) == spec.attentionBias)

        #expect((naming.attentionSinks != nil) == spec.attentionSinks,
                "\(spec.name): sinks flag and name disagree")
        #expect((naming.routerBias != nil) == spec.routerBias)
        #expect((naming.queryNorm != nil) == spec.queryKeyNorm,
                "\(spec.name): QK-norm flag and name disagree")
        #expect((naming.keyNorm != nil) == spec.queryKeyNorm)
        #expect((naming.outputHead != nil) == !spec.tiedEmbedding)
    }

    @Test("Placeholders resolve, and leave nothing behind")
    func placeholdersResolve() {
        let naming = TensorNaming.qwen3MoE
        #expect(naming.resolve(naming.expertGate, layer: 7, expert: 42)
                == "model.layers.7.mlp.experts.42.gate_proj.weight")
        #expect(naming.resolve(naming.inputNorm, layer: 3)
                == "model.layers.3.input_layernorm.weight")

        // Any unresolved placeholder would become a 404 three hours into an
        // install, so check every template survives substitution.
        for template in [naming.inputNorm, naming.postAttentionNorm,
                         naming.queryProjection, naming.keyProjection,
                         naming.valueProjection, naming.outputProjection,
                         naming.router, naming.expertGate, naming.expertUp,
                         naming.expertDown] {
            let resolved = naming.resolve(template, layer: 1, expert: 2)
            #expect(!resolved.contains("{"), "unresolved placeholder in \(resolved)")
        }
    }

    @Test("The two families really do differ where it matters")
    func familiesDiffer() {
        let oss = TensorNaming.gptOSS, qwen = TensorNaming.qwen3MoE
        #expect(oss.expertLayout == .stackedInterleaved)
        #expect(qwen.expertLayout == .perExpertTensors)
        #expect(oss.router != qwen.router, "router name is mlp.router vs mlp.gate")
        #expect(oss.attentionSinks != nil && qwen.attentionSinks == nil)
        #expect(oss.queryNorm == nil && qwen.queryNorm != nil)
    }

    @Test("Qwen3's shape is read from its own config, not GPT-OSS's habits")
    func qwen3Shape() {
        let spec = ArchitectureSpec.qwen3MoE30B
        #expect(spec.layerCount == 48)
        #expect(spec.hiddenSize == 2048)
        // moe_intermediate_size, not intermediate_size. The dense 6144 belongs
        // to no layer here — every layer is routed — and using it would size
        // every expert buffer eight times too large.
        #expect(spec.intermediateSize == 768)
        #expect(spec.attentionHeads == 32)
        #expect(spec.keyValueHeads == 4)
        #expect(spec.headDimension == 128)
        #expect(spec.maxExpertsPerToken == 8)
        #expect(spec.routedLayerIndices.count == 48)
        #expect(spec.layers.allSatisfy { $0.attention == .full },
                "Qwen3 has no sliding window")
        #expect(spec.activation == .silu)
        #expect(spec.queryKeyNorm)
    }

    /// A spec written before these fields existed must still load, because an
    /// install is three hours and adding a field must not invalidate one that
    /// is already on disk.
    @Test("An older manifest decodes with sensible defaults")
    func decodesWithoutNewFields() throws {
        let json = """
        {"name":"gpt-oss-120b","hiddenSize":2880,"intermediateSize":2880,
         "attentionHeads":64,"keyValueHeads":8,"headDimension":64,
         "vocabularySize":201088,"ropeTheta":150000,"rmsNormEpsilon":1e-5,
         "activation":"gptOssClampedGLU","tiedEmbedding":false,
         "attentionSinks":true,"attentionBias":true,"expertBias":true,
         "routerBias":true,"layers":[]}
        """
        let spec = try JSONDecoder().decode(ArchitectureSpec.self, from: Data(json.utf8))
        #expect(spec.queryKeyNorm == false)
        #expect(spec.naming == .gptOSS)
    }
}
