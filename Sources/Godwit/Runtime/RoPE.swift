import Foundation

/// Rotary position embeddings with YaRN frequency correction.
///
/// GPT-OSS was trained at 4,096 tokens and extended to 131,072 with YaRN, a
/// factor of 32. YaRN does not rescale positions uniformly: it interpolates
/// high-frequency components (which encode local order and would blur if
/// stretched) less than low-frequency ones (which encode long-range position and
/// tolerate stretching). The crossover is set by `betaFast` and `betaSlow`.
///
/// This matters even at short contexts. The correction changes the frequencies
/// themselves, not just how positions map onto them, so a prompt of ten tokens
/// is affected exactly as much as one of a hundred thousand.
public struct RoPE {
    public struct Configuration: Sendable {
        public let headDimension: Int
        public let theta: Float
        public let scalingFactor: Float
        public let originalContextLength: Int
        public let betaFast: Float
        public let betaSlow: Float

        public init(headDimension: Int, theta: Float, scalingFactor: Float,
                    originalContextLength: Int, betaFast: Float, betaSlow: Float) {
            self.headDimension = headDimension
            self.theta = theta
            self.scalingFactor = scalingFactor
            self.originalContextLength = originalContextLength
            self.betaFast = betaFast
            self.betaSlow = betaSlow
        }

        /// GPT-OSS-120B's published `rope_scaling` block.
        public static var gptOSS: Configuration {
            Configuration(headDimension: 64, theta: 150_000, scalingFactor: 32,
                          originalContextLength: 4096, betaFast: 32, betaSlow: 1)
        }

        /// Rotary embeddings with no frequency correction at all.
        ///
        /// Qwen3 sets `rope_scaling: null` — it was trained at its full 40,960
        /// context and needs no extension. A scaling factor of 1 makes every
        /// YaRN term collapse: interpolated equals base, so the ramp blends a
        /// value with itself, and `attentionScale` is already guarded at 1.
        /// So this is the same code path arriving at plain RoPE, rather than a
        /// second implementation to keep in step.
        public static func plain(headDimension: Int, theta: Float,
                                 contextLength: Int) -> Configuration {
            Configuration(headDimension: headDimension, theta: theta,
                          scalingFactor: 1, originalContextLength: contextLength,
                          betaFast: 32, betaSlow: 1)
        }

        /// The configuration a model's own spec implies.
        public static func forSpec(_ spec: ArchitectureSpec) -> Configuration {
            // GPT-OSS is the only family here that was extended after training,
            // so it is the only one carrying a YaRN block.
            if spec.activation == .gptOssClampedGLU {
                return Configuration(headDimension: spec.headDimension,
                                     theta: spec.ropeTheta, scalingFactor: 32,
                                     originalContextLength: 4096,
                                     betaFast: 32, betaSlow: 1)
            }
            return .plain(headDimension: spec.headDimension,
                          theta: spec.ropeTheta, contextLength: 40_960)
        }
    }

    public let configuration: Configuration
    /// Corrected inverse frequencies, one per rotation pair.
    public let inverseFrequencies: [Float]
    /// Uniform gain applied to cos and sin, YaRN's `attention_factor`.
    public let attentionScale: Float

    public init(configuration: Configuration) {
        self.configuration = configuration
        let pairs = configuration.headDimension / 2

        // Base frequencies, as without any scaling.
        let base: [Float] = (0..<pairs).map { index in
            1 / powf(configuration.theta,
                     Float(index * 2) / Float(configuration.headDimension))
        }
        // Pure interpolation: every position divided by the extension factor.
        let interpolated = base.map { $0 / configuration.scalingFactor }

        // Dimensions whose wavelength is short relative to the original context
        // are extrapolated; long ones are interpolated; between the two a linear
        // ramp blends them.
        let low = Self.correctionDimension(rotations: configuration.betaFast,
                                           configuration: configuration).rounded(.down)
        let high = Self.correctionDimension(rotations: configuration.betaSlow,
                                            configuration: configuration).rounded(.up)
        let lowBound = max(low, 0)
        let highBound = min(high, Float(pairs - 1))

        var frequencies = [Float](repeating: 0, count: pairs)
        for index in 0..<pairs {
            // 1 at the extrapolation end, 0 at the interpolation end.
            let span = max(highBound - lowBound, 1e-3)
            let ramp = min(max((Float(index) - lowBound) / span, 0), 1)
            let extrapolationWeight = 1 - ramp
            frequencies[index] = interpolated[index] * (1 - extrapolationWeight)
                + base[index] * extrapolationWeight
        }
        self.inverseFrequencies = frequencies

        // Stretching positions lowers attention entropy; this compensates.
        self.attentionScale = configuration.scalingFactor <= 1
            ? 1
            : 0.1 * logf(configuration.scalingFactor) + 1
    }

    /// The dimension index whose wavelength completes `rotations` cycles across
    /// the original training context.
    static func correctionDimension(rotations: Float, configuration: Configuration) -> Float {
        let numerator = Float(configuration.headDimension)
            * logf(Float(configuration.originalContextLength)
                   / (rotations * 2 * Float.pi))
        return numerator / (2 * logf(configuration.theta))
    }

    /// cos and sin for one position, each `headDimension / 2` long.
    ///
    /// NeoX layout: the table covers half the head dimension and is applied to
    /// both halves, rather than interleaved pairs.
    public func table(position: Int) -> (cos: [Float], sin: [Float]) {
        let angles = inverseFrequencies.map { Float(position) * $0 }
        return (angles.map { cosf($0) * attentionScale },
                angles.map { sinf($0) * attentionScale })
    }

    /// Applies the rotation to one head's vector, NeoX half-split.
    ///
    /// `first' = first*cos - second*sin`, `second' = second*cos + first*sin`.
    public func apply(to vector: [Float], position: Int) -> [Float] {
        let half = configuration.headDimension / 2
        precondition(vector.count == configuration.headDimension,
                     "expected \(configuration.headDimension) elements")
        let (cosines, sines) = table(position: position)
        var out = [Float](repeating: 0, count: vector.count)
        for i in 0..<half {
            let first = vector[i]
            let second = vector[i + half]
            out[i] = first * cosines[i] - second * sines[i]
            out[i + half] = second * cosines[i] + first * sines[i]
        }
        return out
    }
}
