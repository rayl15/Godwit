import Foundation

/// How a layer attends over the context.
public enum AttentionKind: String, Codable, Sendable {
    /// Attends to the full context. Storage grows with sequence length.
    case full
    /// Attends to the most recent `window` tokens. Storage is bounded.
    case sliding
}

/// The activation applied between the gate and down projections of an expert.
///
/// Model families disagree here and it is cheap to parameterise: Gemma uses
/// tanh-approximated GELU, most others use SiLU.
public enum FeedForwardActivation: String, Codable, Sendable, CaseIterable {
    case silu
    case geluTanh
    /// GPT-OSS's expert gate: `(up + 1) * gate * sigmoid(alpha * gate)`, with
    /// `gate` clamped above only and `up` clamped both ways.
    ///
    /// Named separately because `config.json` reports `hidden_act: "silu"`,
    /// which is not what the experts actually compute. The `+1` shift and the
    /// asymmetric clamp are both load-bearing.
    case gptOssClampedGLU

    /// The Metal kernel that computes this activation.
    ///
    /// Lives here rather than in the runtime so that adding a case forces an
    /// answer at the point of declaration. `geluTanh` was previously declared
    /// with no kernel of its own and quietly routed to SwiGLU, which runs and
    /// computes the wrong function.
    public var kernel: String {
        switch self {
        case .gptOssClampedGLU: return "gptoss_expert_activation"
        case .silu: return "expert_activation_swiglu"
        case .geluTanh: return "expert_activation_geglu"
        }
    }
}

/// One transformer layer's shape.
public struct LayerSpec: Codable, Sendable, Equatable {
    public let attention: AttentionKind
    /// Sliding-window span in tokens. Ignored when `attention == .full`.
    public let window: Int
    /// Number of routed experts available to this layer, or 0 for a dense layer.
    public let routedExpertCount: Int
    /// Experts activated per token. Ignored when `routedExpertCount == 0`.
    public let expertsPerToken: Int
    /// Whether a dense expert runs in parallel with the routed branch.
    public let hasSharedExpert: Bool

    public init(
        attention: AttentionKind,
        window: Int = 0,
        routedExpertCount: Int,
        expertsPerToken: Int,
        hasSharedExpert: Bool
    ) {
        self.attention = attention
        self.window = window
        self.routedExpertCount = routedExpertCount
        self.expertsPerToken = expertsPerToken
        self.hasSharedExpert = hasSharedExpert
    }

    public var isMixtureOfExperts: Bool { routedExpertCount > 0 }
}

/// A complete, model-agnostic description of a decoder-only transformer.
///
/// Godwit is deliberately driven by this spec rather than by per-model code
/// paths. Kernels receive these values as specialisation constants, so adding a
/// model family should mean writing a loader, not a runtime.
public struct ArchitectureSpec: Codable, Sendable, Equatable {
    public let name: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let rmsNormEpsilon: Float
    public let activation: FeedForwardActivation
    /// Applied to final logits when non-nil. Gemma uses 30.0; most models omit it.
    public let logitSoftcap: Float?
    /// Whether the embedding matrix is reused as the output head.
    public let tiedEmbedding: Bool
    /// Learned per-head term in the attention softmax denominator.
    ///
    /// GPT-OSS ships `self_attn.sinks`; omitting it does not crash, it just
    /// makes attention quietly wrong, so it is modelled explicitly.
    public let attentionSinks: Bool
    /// Whether Q/K/V/O projections carry bias vectors. Many models have none.
    public let attentionBias: Bool
    /// Whether expert projections carry bias vectors.
    public let expertBias: Bool
    /// Whether the router carries a bias vector.
    public let routerBias: Bool
    /// Clamp bound for `gptOssClampedGLU`. GPT-OSS uses 7.0.
    public let activationLimit: Float
    /// Sigmoid steepness for `gptOssClampedGLU`. GPT-OSS uses 1.702.
    public let activationAlpha: Float
    /// Whether q and k are RMS-normalised per head before RoPE. Qwen3 does
    /// this; GPT-OSS does not. Omitting it runs and is quietly wrong.
    public let queryKeyNorm: Bool
    /// Where this family keeps its tensors, and how its experts are stored.
    public let naming: TensorNaming
    public let layers: [LayerSpec]

    public init(
        name: String,
        hiddenSize: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        keyValueHeads: Int,
        headDimension: Int,
        vocabularySize: Int,
        ropeTheta: Float,
        rmsNormEpsilon: Float,
        activation: FeedForwardActivation,
        logitSoftcap: Float?,
        tiedEmbedding: Bool,
        attentionSinks: Bool = false,
        attentionBias: Bool = false,
        expertBias: Bool = false,
        routerBias: Bool = false,
        activationLimit: Float = 7.0,
        activationAlpha: Float = 1.702,
        queryKeyNorm: Bool = false,
        naming: TensorNaming = .gptOSS,
        layers: [LayerSpec]
    ) {
        self.name = name
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads
        self.headDimension = headDimension
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.rmsNormEpsilon = rmsNormEpsilon
        self.activation = activation
        self.logitSoftcap = logitSoftcap
        self.tiedEmbedding = tiedEmbedding
        self.attentionSinks = attentionSinks
        self.attentionBias = attentionBias
        self.expertBias = expertBias
        self.routerBias = routerBias
        self.activationLimit = activationLimit
        self.activationAlpha = activationAlpha
        self.queryKeyNorm = queryKeyNorm
        self.naming = naming
        self.layers = layers
    }

    public var layerCount: Int { layers.count }

    /// Layers whose experts must be streamed, in layer order.
    public var routedLayerIndices: [Int] {
        layers.indices.filter { layers[$0].isMixtureOfExperts }
    }

    /// The largest number of experts any single layer activates per token.
    ///
    /// This bounds the per-layer slot cache and the kernel's routed-blob array.
    public var maxExpertsPerToken: Int {
        layers.map(\.expertsPerToken).max() ?? 0
    }

    /// Decoding tolerates manifests written before a field existed.
    ///
    /// An installation is expensive — hours of transfer and tens of gigabytes —
    /// so adding a spec field must not invalidate one that is already on disk.
    /// Defaults here match the behaviour of the version that lacked the field.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        keyValueHeads = try container.decode(Int.self, forKey: .keyValueHeads)
        headDimension = try container.decode(Int.self, forKey: .headDimension)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        rmsNormEpsilon = try container.decode(Float.self, forKey: .rmsNormEpsilon)
        activation = try container.decode(FeedForwardActivation.self, forKey: .activation)
        logitSoftcap = try container.decodeIfPresent(Float.self, forKey: .logitSoftcap)
        tiedEmbedding = try container.decode(Bool.self, forKey: .tiedEmbedding)
        layers = try container.decode([LayerSpec].self, forKey: .layers)

        attentionSinks = try container.decodeIfPresent(Bool.self, forKey: .attentionSinks) ?? false
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        expertBias = try container.decodeIfPresent(Bool.self, forKey: .expertBias) ?? false
        routerBias = try container.decodeIfPresent(Bool.self, forKey: .routerBias) ?? false
        activationLimit = try container.decodeIfPresent(Float.self, forKey: .activationLimit) ?? 7.0
        activationAlpha = try container.decodeIfPresent(Float.self, forKey: .activationAlpha) ?? 1.702
        // Both added when Qwen3 arrived. An install predating them is GPT-OSS,
        // which has neither QK-norm nor any naming but its own — so the
        // defaults reproduce exactly what that install already assumed.
        queryKeyNorm = try container.decodeIfPresent(Bool.self, forKey: .queryKeyNorm) ?? false
        naming = try container.decodeIfPresent(TensorNaming.self, forKey: .naming) ?? .gptOSS
    }
}

extension ArchitectureSpec {
    /// GPT-OSS-20B — the same architecture at two thirds the depth and a
    /// quarter the experts.
    ///
    /// It exists to test whether this type actually drives the runtime. Every
    /// other property is identical to the 120B: same tensor names, same MXFP4,
    /// same YaRN, same attention sinks, same clamped GLU, same tokeniser. Only
    /// `num_hidden_layers` and `num_local_experts` differ, so if it needs any
    /// code change beyond this declaration, the parameterisation is a fiction.
    public static var gptOSS20B: ArchitectureSpec {
        gptOSS(layers: 24, experts: 32, name: "gpt-oss-20b")
    }

    /// GPT-OSS-120B, transcribed from the published `config.json` and confirmed
    /// against the safetensors header (see `Scripts/analysis/fetch_expert.py`).
    ///
    /// The first target. Note the flags: attention sinks, and biases on
    /// attention, experts, and router — all present here and absent from most
    /// contemporary models.
    public static var gptOSS120B: ArchitectureSpec {
        gptOSS(layers: 36, experts: 128, name: "gpt-oss-120b")
    }

    /// The shape both GPT-OSS sizes share.
    /// Qwen3-30B-A3B. The first non-GPT-OSS family, and the one that decides
    /// whether `ArchitectureSpec` describes a model or merely parameterises
    /// one. It differs in almost everything that GPT-OSS fixed: no attention
    /// sinks, no biases, no sliding window, plain RoPE, plain SwiGLU, and
    /// QK-norm, which nothing here had before.
    public static var qwen3MoE30B: ArchitectureSpec {
        let layers = (0..<48).map { _ in
            LayerSpec(attention: .full, window: 0,
                      routedExpertCount: 128, expertsPerToken: 8,
                      hasSharedExpert: false)
        }
        return ArchitectureSpec(
            name: "qwen3-30b-a3b",
            hiddenSize: 2048,
            // The expert inner width, 768, not the dense one. Qwen3 also has a
            // 6144 `intermediate_size`, which no routed layer uses — every
            // layer here is MoE, so carrying the dense figure would size the
            // expert buffers eight times too large.
            intermediateSize: 768,
            attentionHeads: 32,
            keyValueHeads: 4,
            headDimension: 128,
            vocabularySize: 151936,
            ropeTheta: 1_000_000,
            rmsNormEpsilon: 1e-6,
            activation: .silu,
            logitSoftcap: nil,
            tiedEmbedding: false,
            attentionSinks: false,
            attentionBias: false,
            expertBias: false,
            routerBias: false,
            queryKeyNorm: true,
            naming: .qwen3MoE,
            layers: layers)
    }

    private static func gptOSS(layers layerCount: Int, experts: Int,
                               name: String) -> ArchitectureSpec {
        // Alternating sliding and full attention, starting sliding.
        let layers = (0..<layerCount).map { index in
            LayerSpec(
                attention: index.isMultiple(of: 2) ? .sliding : .full,
                window: 128,
                routedExpertCount: experts,
                expertsPerToken: 4,
                hasSharedExpert: false)
        }
        return ArchitectureSpec(
            name: name,
            hiddenSize: 2880,
            intermediateSize: 2880,
            attentionHeads: 64,
            keyValueHeads: 8,
            headDimension: 64,
            vocabularySize: 201088,
            ropeTheta: 150_000,
            rmsNormEpsilon: 1e-5,
            activation: .gptOssClampedGLU,
            logitSoftcap: nil,
            tiedEmbedding: false,
            attentionSinks: true,
            attentionBias: true,
            expertBias: true,
            routerBias: true,
            queryKeyNorm: false,
            naming: .gptOSS,
            layers: layers)
    }
}
