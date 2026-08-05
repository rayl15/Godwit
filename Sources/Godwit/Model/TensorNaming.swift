import Foundation

/// Where a checkpoint keeps each tensor, and how its experts are laid out.
///
/// Thirteen GPT-OSS tensor names used to be string literals inside the
/// installer. That was fine while there was one model family and actively
/// misleading once there were two: Qwen3 calls its router `mlp.gate` rather
/// than `mlp.router`, has no biases and no attention sinks, and stores each
/// expert as three separate tensors instead of one stacked pair.
///
/// Making this data rather than code means a new family is a value to write
/// down, and — more usefully — that the things it *lacks* are `nil` rather than
/// a lookup that throws halfway through a three-hour install.
public struct TensorNaming: Sendable, Codable, Equatable {
    /// How the checkpoint stores a layer's routed experts.
    public enum ExpertLayout: String, Sendable, Codable {
        /// GPT-OSS: one tensor holds every expert's gate and up projections,
        /// interleaved pairwise on the row axis, and a second holds down.
        /// An expert is a contiguous slice of both.
        case stackedInterleaved
        /// Qwen3: `mlp.experts.<n>.{gate,up,down}_proj`, three tensors per
        /// expert. Contiguity has to be built rather than inherited.
        case perExpertTensors
    }

    // MARK: Model-level

    public var embedding: String
    /// Absent when the model ties input and output embeddings.
    public var outputHead: String?
    public var finalNorm: String

    // MARK: Per-layer, formatted with the layer index

    public var inputNorm: String
    public var postAttentionNorm: String
    public var queryProjection: String
    public var keyProjection: String
    public var valueProjection: String
    public var outputProjection: String
    /// Present only when `ArchitectureSpec.attentionBias` is set.
    public var queryBias: String?
    public var keyBias: String?
    public var valueBias: String?
    public var outputBias: String?
    /// Qwen3 normalises q and k per head before RoPE. GPT-OSS does not.
    public var queryNorm: String?
    public var keyNorm: String?
    /// Present only when `ArchitectureSpec.attentionSinks` is set.
    public var attentionSinks: String?

    public var router: String
    public var routerBias: String?

    public var expertLayout: ExpertLayout
    /// Formatted with layer, then expert index, when `perExpertTensors`.
    public var expertGate: String
    public var expertUp: String
    public var expertDown: String
    public var expertGateBias: String?
    public var expertUpBias: String?
    public var expertDownBias: String?

    /// Substitutes `{layer}` and `{expert}` placeholders.
    public func resolve(_ template: String, layer: Int, expert: Int = 0) -> String {
        template
            .replacingOccurrences(of: "{layer}", with: String(layer))
            .replacingOccurrences(of: "{expert}", with: String(expert))
    }

    /// GPT-OSS-120B and -20B. Biases everywhere, sinks, stacked experts.
    public static let gptOSS = TensorNaming(
        embedding: "model.embed_tokens.weight",
        outputHead: "lm_head.weight",
        finalNorm: "model.norm.weight",
        inputNorm: "model.layers.{layer}.input_layernorm.weight",
        postAttentionNorm: "model.layers.{layer}.post_attention_layernorm.weight",
        queryProjection: "model.layers.{layer}.self_attn.q_proj.weight",
        keyProjection: "model.layers.{layer}.self_attn.k_proj.weight",
        valueProjection: "model.layers.{layer}.self_attn.v_proj.weight",
        outputProjection: "model.layers.{layer}.self_attn.o_proj.weight",
        queryBias: "model.layers.{layer}.self_attn.q_proj.bias",
        keyBias: "model.layers.{layer}.self_attn.k_proj.bias",
        valueBias: "model.layers.{layer}.self_attn.v_proj.bias",
        outputBias: "model.layers.{layer}.self_attn.o_proj.bias",
        queryNorm: nil,
        keyNorm: nil,
        attentionSinks: "model.layers.{layer}.self_attn.sinks",
        router: "model.layers.{layer}.mlp.router.weight",
        routerBias: "model.layers.{layer}.mlp.router.bias",
        expertLayout: .stackedInterleaved,
        expertGate: "model.layers.{layer}.mlp.experts.gate_up_proj_blocks",
        expertUp: "model.layers.{layer}.mlp.experts.gate_up_proj_scales",
        expertDown: "model.layers.{layer}.mlp.experts.down_proj_blocks",
        expertGateBias: "model.layers.{layer}.mlp.experts.gate_up_proj_bias",
        expertUpBias: nil,
        expertDownBias: "model.layers.{layer}.mlp.experts.down_proj_bias")

    /// Qwen3 MoE. No biases, no sinks, QK-norm, one tensor per expert per
    /// projection.
    public static let qwen3MoE = TensorNaming(
        embedding: "model.embed_tokens.weight",
        outputHead: "lm_head.weight",
        finalNorm: "model.norm.weight",
        inputNorm: "model.layers.{layer}.input_layernorm.weight",
        postAttentionNorm: "model.layers.{layer}.post_attention_layernorm.weight",
        queryProjection: "model.layers.{layer}.self_attn.q_proj.weight",
        keyProjection: "model.layers.{layer}.self_attn.k_proj.weight",
        valueProjection: "model.layers.{layer}.self_attn.v_proj.weight",
        outputProjection: "model.layers.{layer}.self_attn.o_proj.weight",
        queryBias: nil,
        keyBias: nil,
        valueBias: nil,
        outputBias: nil,
        queryNorm: "model.layers.{layer}.self_attn.q_norm.weight",
        keyNorm: "model.layers.{layer}.self_attn.k_norm.weight",
        attentionSinks: nil,
        router: "model.layers.{layer}.mlp.gate.weight",
        routerBias: nil,
        expertLayout: .perExpertTensors,
        expertGate: "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        expertUp: "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        expertDown: "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        expertGateBias: nil,
        expertUpBias: nil,
        expertDownBias: nil)
}
