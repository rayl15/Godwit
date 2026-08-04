import Foundation
import Metal

/// Runs tokens through real layers and records what the routers actually chose.
///
/// This supersedes `RoutingTrace`, which drove the router with normalised token
/// embeddings because no layer existed to produce real hidden states. Here the
/// residual stream is the genuine article.
///
/// It exists to answer one question in particular: **is expert routing
/// predictable from the previous layer?**
///
/// The answer decides whether prefetch is possible at all. Layer *n+1*'s
/// attention needs layer *n*'s complete output, experts included, so nothing
/// can be computed ahead — there is no work to overlap the reads against, which
/// is the open problem left by GPT-OSS having no shared expert. Prediction is
/// the only way out, and the published evidence disagrees sharply:
/// TurboFieldfare measured 7% cross-layer predictability on Gemma, while
/// colibrì reports 71.6% one layer ahead. Neither was measured on this model.
public struct LayerTrace {
    public struct Report: Sendable {
        public let layerCount: Int
        public let tokenCount: Int
        public let topK: Int
        public let expertCount: Int
        /// `routing[layer][token]` — the experts chosen, in rank order.
        public let routing: [[[Int]]]
        public let seconds: Double

        /// Fraction of layer *n+1*'s experts already selected at layer *n*.
        ///
        /// This is the naive predictor: copy the previous layer's choices and
        /// see how many hit. It is the cheapest possible prefetch, so it is the
        /// right thing to measure first.
        public var adjacentOverlap: [Double] {
            guard layerCount > 1 else { return [] }
            return (0..<(layerCount - 1)).map { layer in
                var hits = 0
                var total = 0
                for token in 0..<tokenCount {
                    let previous = Set(routing[layer][token])
                    let next = routing[layer + 1][token]
                    hits += next.filter { previous.contains($0) }.count
                    total += next.count
                }
                return Double(hits) / Double(total)
            }
        }

        /// Same-layer reuse between consecutive tokens.
        ///
        /// A useful control: if adjacent *tokens* at one layer agree far more
        /// than adjacent *layers* do, then caching is the right lever and
        /// prefetching across layers is not.
        public var tokenToTokenOverlap: [Double] {
            guard tokenCount > 1 else { return [] }
            return (0..<layerCount).map { layer in
                var hits = 0
                var total = 0
                for token in 1..<tokenCount {
                    let previous = Set(routing[layer][token - 1])
                    let current = routing[layer][token]
                    hits += current.filter { previous.contains($0) }.count
                    total += current.count
                }
                return Double(hits) / Double(total)
            }
        }

        /// What a random guess would score, for scale.
        public var chanceOverlap: Double { Double(topK) / Double(expertCount) }

        /// Hit rate per slot count, replaying the real trace through the planner.
        public func hitRates(slots: [Int]) -> [(slots: Int, rate: Double)] {
            slots.filter { $0 >= topK }.map { slotCount in
                var hits = 0
                var total = 0
                for layer in 0..<layerCount {
                    var planner = ExpertCachePlanner(slotCount: slotCount,
                                                     expertCount: expertCount)
                    for token in 0..<tokenCount {
                        let plan = planner.plan(experts: Array(Set(routing[layer][token])))
                        hits += plan.hitCount
                        total += plan.experts.count
                    }
                }
                return (slotCount, Double(hits) / Double(total))
            }
        }
    }

    public let context: MetalContext
    public let reader: ModelReader

    public init(context: MetalContext, reader: ModelReader) {
        self.context = context
        self.reader = reader
    }

    /// Runs `tokenCount` tokens through the first `layerCount` layers.
    ///
    /// Token ids are strided across the vocabulary rather than drawn from real
    /// text, so this still lacks the temporal locality a genuine sequence has.
    /// The residual stream, however, is real.
    public func run(layerCount: Int, tokenCount: Int) throws -> Report {
        let spec = reader.manifest.spec
        let rope = RoPE(configuration: .gptOSS)
        let width = spec.hiddenSize

        var stream = try initialHidden(tokenCount: tokenCount)
        var routing: [[[Int]]] = []
        let started = Date()
        let cache = try KVCache(context: context, spec: spec,
                                maxContext: max(tokenCount, 128), layerCount: layerCount)

        for index in 0..<layerCount {
            let layer = TransformerLayer(context: context, reader: reader,
                                         index: index, rope: rope)
            let weights = try layer.loadWeights()
            let (next, trace) = try layer.forward(
                hidden: stream, tokenCount: tokenCount, positionBase: 0,
                weights: weights, cache: cache)
            stream = next
            routing.append(trace.routing.map(\.experts))
        }
        _ = width

        return Report(layerCount: layerCount, tokenCount: tokenCount,
                      topK: spec.layers[0].expertsPerToken,
                      expertCount: spec.layers[0].routedExpertCount,
                      routing: routing,
                      seconds: Date().timeIntervalSince(started))
    }

    /// Embeddings for a spread of token ids, dequantised from the trunk.
    private func initialHidden(tokenCount: Int) throws -> [Float16] {
        let spec = reader.manifest.spec
        let width = spec.hiddenSize
        let embed = try reader.loadTrunk(section: "embed", device: context.device)
        let shape = try reader.trunkShape("embed")
        let rows = shape[0]
        let groups = width / Int8Affine.groupSize

        let codes = embed.contents()
        let meta = codes.advanced(by: rows * width)
        var out = [Float16](repeating: 0, count: tokenCount * width)

        let stride = max(1, spec.vocabularySize / max(tokenCount, 1))
        for token in 0..<tokenCount {
            let id = (token * stride) % rows
            let codeRow = codes.advanced(by: id * width)
                .bindMemory(to: UInt8.self, capacity: width)
            let metaRow = meta.advanced(by: id * groups * 4)
                .bindMemory(to: UInt16.self, capacity: groups * 2)
            for group in 0..<groups {
                let scale = BFloat16.toFloat(metaRow[group * 2])
                let zero = BFloat16.toFloat(metaRow[group * 2 + 1])
                for i in 0..<Int8Affine.groupSize {
                    let index = group * Int8Affine.groupSize + i
                    out[token * width + index] = Float16(Float(codeRow[index]) * scale + zero)
                }
            }
        }
        return out
    }
}
