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
public enum FeedForwardActivation: String, Codable, Sendable {
    case silu
    case geluTanh
    /// GPT-OSS's expert gate: `(up + 1) * gate * sigmoid(alpha * gate)`, with
    /// `gate` clamped above only and `up` clamped both ways.
    ///
    /// Named separately because `config.json` reports `hidden_act: "silu"`,
    /// which is not what the experts actually compute. The `+1` shift and the
    /// asymmetric clamp are both load-bearing.
    case gptOssClampedGLU
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
}

extension ArchitectureSpec {
    /// GPT-OSS-120B, transcribed from the published `config.json` and confirmed
    /// against the safetensors header (see `Scripts/analysis/fetch_expert.py`).
    ///
    /// The first target. Note the flags: attention sinks, and biases on
    /// attention, experts, and router — all present here and absent from most
    /// contemporary models.
    public static var gptOSS120B: ArchitectureSpec {
        // 36 layers alternating sliding and full attention, starting sliding.
        let layers = (0..<36).map { index in
            LayerSpec(
                attention: index.isMultiple(of: 2) ? .sliding : .full,
                window: 128,
                routedExpertCount: 128,
                expertsPerToken: 4,
                hasSharedExpert: false)
        }
        return ArchitectureSpec(
            name: "gpt-oss-120b",
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
            layers: layers)
    }
}
