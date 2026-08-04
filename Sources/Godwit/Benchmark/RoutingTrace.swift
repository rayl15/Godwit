import Foundation
import Metal

/// Measures how skewed the router's expert choices actually are.
///
/// Cache hit rate depends entirely on this, and until now it has been the
/// weakest input in the feasibility estimate: a Zipf exponent fitted to
/// someone else's published measurement on a different model. This replaces the
/// guess with the real router.
///
/// **Limitation, stated plainly.** Real hidden states at layer *n* are
/// `norm(residual + attention_output)`, and attention is not implemented yet.
/// This drives the router with normalised token embeddings instead, which have
/// the right dimensionality and scale but not the contextual structure a real
/// forward pass produces. Treat the skew as indicative, not final — it should
/// be re-measured once a full layer runs.
public struct RoutingTrace {
    public struct Report: Sendable {
        public let tokensRouted: Int
        public let expertCount: Int
        public let topK: Int
        /// Times each expert was selected, indexed by expert id.
        public let selectionCounts: [Int]
        /// Hit rate by slot count, from replaying the trace through the planner.
        public let hitRates: [(slots: Int, rate: Double)]

        /// Share of selections taken by the most popular tenth of experts.
        ///
        /// 0.1 means perfectly uniform routing; higher means skew the cache can
        /// exploit.
        public var topDecileShare: Double {
            let sorted = selectionCounts.sorted(by: >)
            let decile = max(1, expertCount / 10)
            let total = selectionCounts.reduce(0, +)
            guard total > 0 else { return 0 }
            return Double(sorted.prefix(decile).reduce(0, +)) / Double(total)
        }

        /// Experts never chosen at all.
        public var unusedExperts: Int { selectionCounts.filter { $0 == 0 }.count }

        /// Normalised entropy: 1.0 is uniform, lower means concentrated.
        public var normalisedEntropy: Double {
            let total = Double(selectionCounts.reduce(0, +))
            guard total > 0 else { return 0 }
            var entropy = 0.0
            for count in selectionCounts where count > 0 {
                let p = Double(count) / total
                entropy -= p * Foundation.log2(p)
            }
            return entropy / Foundation.log2(Double(expertCount))
        }
    }

    public let context: MetalContext
    public let reader: ModelReader

    public init(context: MetalContext, reader: ModelReader) {
        self.context = context
        self.reader = reader
    }

    /// Routes `tokenCount` embedding-derived states through layer `layer`'s
    /// router and replays the result through the cache planner.
    public func run(layer: Int, tokenCount: Int, slotCounts: [Int] = [4, 6, 8, 12, 16]) throws -> Report {
        let spec = reader.manifest.spec
        let device = context.device
        let expertCount = spec.layers[layer].routedExpertCount
        let topK = spec.layers[layer].expertsPerToken

        let weight = try reader.loadTrunk(section: "layer\(layer).router_w", device: device)
        let bias = try reader.loadTrunk(section: "layer\(layer).router_b", device: device)
        let normWeight = try reader.loadTrunk(section: "layer\(layer).post_norm", device: device)

        let router = Router(context: context, expertCount: expertCount,
                            hiddenSize: spec.hiddenSize, topK: topK)
        let embeddings = try EmbeddingTable(reader: reader, context: context)

        var counts = [Int](repeating: 0, count: expertCount)
        var trace: [[Int]] = []
        trace.reserveCapacity(tokenCount)

        // Spread token ids across the vocabulary; low ids are special tokens and
        // would not represent normal text.
        let stride = max(1, spec.vocabularySize / tokenCount)
        for index in 0..<tokenCount {
            let token = (index * stride) % spec.vocabularySize
            let hidden = try embeddings.normalisedRow(token: token,
                                                      normWeight: normWeight,
                                                      epsilon: spec.rmsNormEpsilon)
            let decision = try router.route(hidden: hidden, weight: weight, bias: bias)
            for expert in decision.experts { counts[expert] += 1 }
            trace.append(decision.experts)
        }

        // Replay through the real planner rather than a model of it.
        var hitRates: [(Int, Double)] = []
        for slots in slotCounts where slots >= topK {
            var planner = ExpertCachePlanner(slotCount: slots, expertCount: expertCount)
            var hits = 0
            var total = 0
            for selection in trace {
                let plan = planner.plan(experts: Array(Set(selection)))
                hits += plan.hitCount
                total += plan.experts.count
            }
            hitRates.append((slots, Double(hits) / Double(total)))
        }

        return Report(tokensRouted: tokenCount, expertCount: expertCount, topK: topK,
                      selectionCounts: counts, hitRates: hitRates)
    }
}

/// Reads and dequantises rows of the int8 embedding table.
struct EmbeddingTable {
    let reader: ModelReader
    let context: MetalContext
    let codes: MTLBuffer
    let meta: MTLBuffer
    let cols: Int

    init(reader: ModelReader, context: MetalContext) throws {
        self.reader = reader
        self.context = context
        let shape = try reader.trunkShape("embed")
        self.cols = shape[1]
        // The whole table: 0.57 GiB at int8, which is what it costs to be
        // resident at run time anyway.
        let buffer = try reader.loadTrunk(section: "embed", device: context.device)
        self.codes = buffer
        self.meta = buffer
    }

    /// Dequantises one token's embedding and applies RMSNorm.
    func normalisedRow(token: Int, normWeight: MTLBuffer, epsilon: Float) throws -> [Float16] {
        let shape = try reader.trunkShape("embed")
        let rows = shape[0]
        let groups = cols / Int8Affine.groupSize
        let base = codes.contents()
        let metaBase = base.advanced(by: rows * cols)

        var values = [Float](repeating: 0, count: cols)
        let codeRow = base.advanced(by: token * cols).bindMemory(to: UInt8.self, capacity: cols)
        let metaRow = metaBase.advanced(by: token * groups * 4)
            .bindMemory(to: UInt16.self, capacity: groups * 2)

        for group in 0..<groups {
            let scale = BFloat16.toFloat(metaRow[group * 2])
            let zero = BFloat16.toFloat(metaRow[group * 2 + 1])
            for i in 0..<Int8Affine.groupSize {
                let index = group * Int8Affine.groupSize + i
                values[index] = Float(codeRow[index]) * scale + zero
            }
        }

        // RMSNorm on the CPU: one row, and doing it here keeps the trace tool
        // free of a GPU round trip per token.
        var sumSquares: Float = 0
        for value in values { sumSquares += value * value }
        let inverse = 1 / (sumSquares / Float(cols) + epsilon).squareRoot()

        let weights = normWeight.contents().bindMemory(to: UInt16.self, capacity: cols)
        return (0..<cols).map { Float16(values[$0] * inverse * BFloat16.toFloat(weights[$0])) }
    }
}
