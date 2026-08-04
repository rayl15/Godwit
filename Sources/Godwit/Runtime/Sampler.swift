import Foundation

/// Turns logits into a token.
///
/// Temperature, top-k and top-p, applied in that order — the conventional
/// ordering, and the one whose behaviour people's intuitions are calibrated to.
public struct Sampler {
    public struct Settings: Sendable {
        /// 0 means greedy: always the highest logit, fully deterministic.
        public var temperature: Float
        /// Keep only the k highest logits. 0 disables.
        public var topK: Int
        /// Keep the smallest set whose probability sums past this. 1 disables.
        public var topP: Float
        /// Divides the logit of any token already produced. 1 disables.
        public var repetitionPenalty: Float
        public var seed: UInt64

        public init(temperature: Float = 0.7, topK: Int = 40, topP: Float = 0.95,
                    repetitionPenalty: Float = 1.0, seed: UInt64 = 0) {
            self.temperature = temperature
            self.topK = topK
            self.topP = topP
            self.repetitionPenalty = repetitionPenalty
            self.seed = seed
        }

        public static var greedy: Settings {
            Settings(temperature: 0, topK: 0, topP: 1)
        }
    }

    public private(set) var settings: Settings
    private var state: UInt64

    public init(settings: Settings = Settings()) {
        self.settings = settings
        // A zero seed would leave the generator stuck, so it means "arbitrary".
        self.state = settings.seed == 0 ? UInt64.random(in: 1...UInt64.max) : settings.seed
    }

    /// splitmix64: small, fast, and reproducible across machines, which matters
    /// more here than statistical excellence — a wrong token is not subtle.
    private mutating func nextUnit() -> Float {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        return Float(z >> 40) / Float(1 << 24)
    }

    public mutating func pick(from logits: [Float], history: [Int] = []) -> Int {
        var scores = logits

        if settings.repetitionPenalty != 1 {
            // Divide when positive, multiply when negative — otherwise a penalty
            // would *raise* the score of a token whose logit is below zero.
            for token in Set(history) where token < scores.count {
                scores[token] = scores[token] > 0
                    ? scores[token] / settings.repetitionPenalty
                    : scores[token] * settings.repetitionPenalty
            }
        }

        guard settings.temperature > 0 else {
            var best = 0
            for index in scores.indices where scores[index] > scores[best] { best = index }
            return best
        }

        // Rank once, then narrow. Sorting 201,088 entries per token is wasteful
        // but honest; a partial selection is the obvious later optimisation.
        var ranked = scores.enumerated().sorted { $0.element > $1.element }
        if settings.topK > 0 && settings.topK < ranked.count {
            ranked = Array(ranked.prefix(settings.topK))
        }

        let peak = ranked[0].element
        var probabilities = ranked.map { expf(($0.element - peak) / settings.temperature) }
        let total = probabilities.reduce(0, +)
        for index in probabilities.indices { probabilities[index] /= total }

        if settings.topP < 1 {
            var running: Float = 0
            var keep = probabilities.count
            for index in probabilities.indices {
                running += probabilities[index]
                if running >= settings.topP { keep = index + 1; break }
            }
            ranked = Array(ranked.prefix(keep))
            probabilities = Array(probabilities.prefix(keep))
            let renormalised = probabilities.reduce(0, +)
            for index in probabilities.indices { probabilities[index] /= renormalised }
        }

        let target = nextUnit()
        var cumulative: Float = 0
        for index in probabilities.indices {
            cumulative += probabilities[index]
            if target <= cumulative { return ranked[index].offset }
        }
        return ranked[ranked.count - 1].offset
    }
}
