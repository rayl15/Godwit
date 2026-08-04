import Foundation

/// Measures what each expert is *for*, by watching which ones the router picks
/// for different kinds of text.
///
/// Every expert gets an affinity vector over a set of topic probes — how often
/// it fires for Python versus for Chinese versus for poetry, relative to how
/// often it fires at all. An expert that only ever activates on code is a code
/// specialist; one that fires uniformly is a generalist doing something the
/// model needs everywhere.
///
/// Positions come from principal components of those affinity vectors, so two
/// experts sit close together because they respond to the same material. This
/// is measurement, not a learned embedding: nothing here is trained, and the
/// axes are whatever directions the routing actually varies along.
public struct ExpertAtlas {
    public struct Probe: Sendable {
        public let topic: String
        public let text: String

        public init(topic: String, text: String) {
            self.topic = topic
            self.text = text
        }
    }

    /// One expert's measured character.
    public struct Point: Codable, Sendable {
        public let layer: Int
        public let expert: Int
        public let x: Float
        public let y: Float
        public let z: Float
        /// Times this expert was selected across every probe.
        public let activations: Int
        /// The topic it prefers most.
        public let topic: String
        /// How concentrated its preference is. 0 is a perfect generalist, 1 a
        /// pure specialist that only ever fires for one topic.
        public let specialisation: Float
    }

    public struct Atlas: Codable, Sendable {
        public let topics: [String]
        public let points: [Point]
        public let layerCount: Int
        public let expertCount: Int
        /// Share of variance each axis explains, for honesty about the layout.
        public let variance: [Float]
    }

    /// Probes chosen to be linguistically far apart, so a difference in routing
    /// means something. Each is long enough for the router to see a consistent
    /// register rather than one unusual token.
    public static let defaultProbes: [Probe] = [
        Probe(topic: "python", text: """
            def quicksort(items):
                if len(items) <= 1:
                    return items
                pivot = items[len(items) // 2]
                left = [x for x in items if x < pivot]
                right = [x for x in items if x > pivot]
                return quicksort(left) + [pivot] + quicksort(right)
            """),
        Probe(topic: "sql", text: """
            SELECT customers.name, SUM(orders.total) AS revenue
            FROM customers JOIN orders ON orders.customer_id = customers.id
            WHERE orders.created_at >= '2024-01-01' GROUP BY customers.name
            HAVING SUM(orders.total) > 1000 ORDER BY revenue DESC LIMIT 20;
            """),
        Probe(topic: "math", text: """
            Let f be continuous on [a,b] and differentiable on (a,b). By the mean
            value theorem there exists c in (a,b) such that f'(c) equals
            (f(b) - f(a)) / (b - a). Integrating both sides and applying Fubini
            gives the required bound on the remainder term.
            """),
        Probe(topic: "poetry", text: """
            The sea is calm tonight, the tide is full, the moon lies fair upon
            the straits; on the French coast the light gleams and is gone; the
            cliffs of England stand, glimmering and vast, out in the tranquil bay.
            """),
        Probe(topic: "legal", text: """
            The Party of the First Part hereby indemnifies and holds harmless the
            Party of the Second Part against any and all claims, liabilities, and
            expenses arising out of or in connection with the performance of its
            obligations hereunder, save where such claims arise from gross
            negligence or wilful misconduct.
            """),
        Probe(topic: "medical", text: """
            The patient presented with acute dyspnoea and pleuritic chest pain.
            Auscultation revealed decreased breath sounds at the left base.
            CT pulmonary angiography confirmed segmental pulmonary embolism, and
            anticoagulation with low molecular weight heparin was commenced.
            """),
        Probe(topic: "chinese", text: """
            今天天气很好，我打算去公园散步。公园里有很多人在锻炼身体，
            也有小孩子在玩耍。我喜欢在这样的天气里读书，感觉特别放松。
            """),
        Probe(topic: "japanese", text: """
            昨日は友達と一緒に映画を見に行きました。とても面白い作品で、
            最後のシーンでは思わず涙が出てしまいました。また見たいと思います。
            """),
        Probe(topic: "russian", text: """
            Вчера я прочитал очень интересную книгу о истории науки.
            Автор подробно описывает, как менялись представления учёных
            о строении вселенной на протяжении нескольких столетий.
            """),
        Probe(topic: "json", text: """
            {"users": [{"id": 1, "name": "Ada", "roles": ["admin", "editor"],
            "active": true, "meta": {"created": "2024-03-11T08:00:00Z",
            "tags": [], "score": 99.5}}, {"id": 2, "name": "Grace",
            "roles": ["viewer"], "active": false, "meta": null}]}
            """),
        Probe(topic: "chat", text: """
            hey! yeah I was thinking the same thing lol. do you want to grab
            coffee tomorrow before the thing? I'm free after like 10 or so.
            no worries if not, we can just catch up later this week
            """),
        Probe(topic: "history", text: """
            The Treaty of Westphalia in 1648 ended the Thirty Years' War and is
            conventionally taken to mark the emergence of the modern state system,
            establishing the principle that sovereigns held authority over
            religious matters within their own territories.
            """),
    ]

    public let context: MetalContext
    public let reader: ModelReader

    public init(context: MetalContext, reader: ModelReader) {
        self.context = context
        self.reader = reader
    }

    /// Runs every probe and builds the atlas.
    public func build(
        probes: [Probe] = defaultProbes,
        slots: Int = 8,
        progress: (String, Int, Int) -> Void = { _, _, _ in }
    ) throws -> Atlas {
        let tokenizer = try reader.loadTokenizer()
        let runner = ModelRunner(context: context, reader: reader)
        let weights = try runner.loadWeights()
        let expertCache = try runner.makeExpertCache(slots: slots)

        let layerCount = reader.manifest.layerCount
        let expertCount = reader.manifest.expertCount
        let topics = probes.map(\.topic)

        // counts[layer][expert][topic]
        var counts = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0, count: topics.count),
                                 count: expertCount),
            count: layerCount)

        for (index, probe) in probes.enumerated() {
            progress(probe.topic, index + 1, probes.count)
            let ids = tokenizer.encode(probe.text)
            let cache = try runner.makeCache(maxContext: ids.count + 8)
            _ = try runner.logits(
                tokens: ids, positionBase: 0, cache: cache, weights: weights,
                expertCache: expertCache,
                routing: { layer, decisions in
                    for decision in decisions {
                        for expert in decision.experts {
                            counts[layer][expert][index] += 1
                        }
                    }
                })
        }

        return Self.assemble(counts: counts, topics: topics,
                             layerCount: layerCount, expertCount: expertCount)
    }

    // MARK: - Analysis

    static func assemble(counts: [[[Float]]], topics: [String],
                         layerCount: Int, expertCount: Int) -> Atlas {
        let topicCount = topics.count

        // Normalise twice. First by topic, because probes differ in length and
        // a longer one would otherwise look like every expert's favourite.
        // Then by expert, so the vector describes preference rather than how
        // busy the expert is.
        var topicTotals = [Float](repeating: 0, count: topicCount)
        for layer in counts { for expert in layer {
            for t in 0..<topicCount { topicTotals[t] += expert[t] }
        } }

        var vectors: [[Float]] = []
        var identity: [(layer: Int, expert: Int, total: Float)] = []
        for layer in 0..<layerCount {
            for expert in 0..<expertCount {
                let raw = counts[layer][expert]
                let total = raw.reduce(0, +)
                guard total > 0 else { continue }
                var vector = [Float](repeating: 0, count: topicCount)
                for t in 0..<topicCount where topicTotals[t] > 0 {
                    vector[t] = raw[t] / topicTotals[t]
                }
                let sum = vector.reduce(0, +)
                if sum > 0 { for t in 0..<topicCount { vector[t] /= sum } }
                vectors.append(vector)
                identity.append((layer, expert, total))
            }
        }
        guard !vectors.isEmpty else {
            return Atlas(topics: topics, points: [], layerCount: layerCount,
                         expertCount: expertCount, variance: [])
        }

        let (axes, variance) = principalAxes(vectors, dimensions: 3)

        var mean = [Float](repeating: 0, count: topicCount)
        for vector in vectors { for t in 0..<topicCount { mean[t] += vector[t] } }
        for t in 0..<topicCount { mean[t] /= Float(vectors.count) }

        var points: [Point] = []
        points.reserveCapacity(vectors.count)
        for (index, vector) in vectors.enumerated() {
            let centred = (0..<topicCount).map { vector[$0] - mean[$0] }
            let coordinates = (0..<3).map { axis in
                (0..<topicCount).reduce(Float(0)) { $0 + centred[$1] * axes[axis][$1] }
            }
            let best = vector.indices.max { vector[$0] < vector[$1] } ?? 0
            points.append(Point(
                layer: identity[index].layer,
                expert: identity[index].expert,
                x: coordinates[0], y: coordinates[1], z: coordinates[2],
                activations: Int(identity[index].total),
                topic: topics[best],
                specialisation: concentration(vector)))
        }

        return Atlas(topics: topics, points: points, layerCount: layerCount,
                     expertCount: expertCount, variance: variance)
    }

    /// 0 for a uniform vector, 1 for one that puts everything on one topic.
    ///
    /// Normalised entropy rather than max-share: an expert split evenly between
    /// two topics is meaningfully different from one split across twelve, and
    /// the maximum alone cannot tell them apart.
    static func concentration(_ vector: [Float]) -> Float {
        let total = vector.reduce(0, +)
        guard total > 0, vector.count > 1 else { return 0 }
        var entropy: Float = 0
        for value in vector where value > 0 {
            let p = value / total
            entropy -= p * log(p)
        }
        return max(0, 1 - entropy / log(Float(vector.count)))
    }

    /// Top principal components, via Jacobi eigendecomposition of the
    /// covariance matrix.
    ///
    /// The matrix is only topics × topics — twelve by twelve — so an iterative
    /// method that is obviously correct beats anything cleverer.
    static func principalAxes(_ vectors: [[Float]], dimensions: Int)
        -> (axes: [[Float]], variance: [Float]) {
        let n = vectors[0].count
        var mean = [Float](repeating: 0, count: n)
        for vector in vectors { for i in 0..<n { mean[i] += vector[i] } }
        for i in 0..<n { mean[i] /= Float(vectors.count) }

        var covariance = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for vector in vectors {
            for i in 0..<n {
                let a = Double(vector[i] - mean[i])
                for j in i..<n {
                    let value = a * Double(vector[j] - mean[j])
                    covariance[i][j] += value
                    if i != j { covariance[j][i] += value }
                }
            }
        }
        let scale = Double(max(vectors.count - 1, 1))
        for i in 0..<n { for j in 0..<n { covariance[i][j] /= scale } }

        // Jacobi: repeatedly zero the largest off-diagonal element by rotation.
        var eigenvectors = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n { eigenvectors[i][i] = 1 }

        for _ in 0..<100 {
            var largest = 0.0
            var p = 0, q = 1
            for i in 0..<n { for j in (i + 1)..<n where abs(covariance[i][j]) > largest {
                largest = abs(covariance[i][j]); p = i; q = j
            } }
            if largest < 1e-12 { break }

            let theta = 0.5 * atan2(2 * covariance[p][q],
                                    covariance[p][p] - covariance[q][q])
            let c = cos(theta), s = sin(theta)
            for k in 0..<n {
                let kp = covariance[k][p], kq = covariance[k][q]
                covariance[k][p] = c * kp + s * kq
                covariance[k][q] = -s * kp + c * kq
            }
            for k in 0..<n {
                let pk = covariance[p][k], qk = covariance[q][k]
                covariance[p][k] = c * pk + s * qk
                covariance[q][k] = -s * pk + c * qk
            }
            for k in 0..<n {
                let kp = eigenvectors[k][p], kq = eigenvectors[k][q]
                eigenvectors[k][p] = c * kp + s * kq
                eigenvectors[k][q] = -s * kp + c * kq
            }
        }

        let order = (0..<n).sorted { covariance[$0][$0] > covariance[$1][$1] }
        let totalVariance = (0..<n).reduce(0.0) { $0 + max(covariance[$1][$1], 0) }
        var axes: [[Float]] = []
        var variance: [Float] = []
        for k in 0..<min(dimensions, n) {
            let column = order[k]
            axes.append((0..<n).map { Float(eigenvectors[$0][column]) })
            variance.append(totalVariance > 0
                            ? Float(max(covariance[column][column], 0) / totalVariance) : 0)
        }
        return (axes, variance)
    }
}
