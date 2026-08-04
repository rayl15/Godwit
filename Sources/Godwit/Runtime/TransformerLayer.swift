import Foundation
import Metal

/// One complete transformer layer.
///
/// ```
/// h += attention(norm(h))          <- resident weights
/// h += moe(norm(h))                <- streamed weights
/// ```
///
/// The asymmetry between those two lines is the entire project. The first
/// touches ~26M resident parameters; the second selects 4 experts out of 128 and
/// pulls ~50 MiB off disk to do it.
public struct TransformerLayer {
    public let context: MetalContext
    public let reader: ModelReader
    public let spec: ArchitectureSpec
    public let index: Int

    private let attention: Attention
    private let router: Router
    private let experts: ExpertRunner

    public init(context: MetalContext, reader: ModelReader, index: Int, rope: RoPE) {
        self.context = context
        self.reader = reader
        self.spec = reader.manifest.spec
        self.index = index
        self.attention = Attention(context: context, spec: spec, rope: rope)
        self.router = Router(context: context,
                             expertCount: spec.layers[index].routedExpertCount,
                             hiddenSize: spec.hiddenSize,
                             topK: spec.layers[index].expertsPerToken)
        self.experts = ExpertRunner(context: context, reader: reader)
    }

    /// Resident weights for this layer. Attention plus the two norms and the
    /// router — everything except the experts.
    public struct Weights {
        public let attention: Attention.Weights
        public let inputNorm: MTLBuffer
        public let postNorm: MTLBuffer
        public let routerWeight: MTLBuffer
        public let routerBias: MTLBuffer
    }

    public func loadWeights() throws -> Weights {
        let device = context.device
        return Weights(
            attention: try Attention.loadWeights(reader: reader, layer: index, device: device),
            inputNorm: try reader.loadTrunk(section: "layer\(index).input_norm", device: device),
            postNorm: try reader.loadTrunk(section: "layer\(index).post_norm", device: device),
            routerWeight: try reader.loadTrunk(section: "layer\(index).router_w", device: device),
            routerBias: try reader.loadTrunk(section: "layer\(index).router_b", device: device))
    }

    /// What the layer did, for tracing and cache accounting.
    public struct Trace: Sendable {
        public let routing: [Router.Decision]
    }

    /// Runs the layer over `tokenCount` tokens starting at `positionBase`.
    ///
    /// `hidden` is the residual stream, `[tokenCount, hiddenSize]`, and is
    /// returned updated.
    public func forward(
        hidden: [Float16], tokenCount: Int, positionBase: Int,
        weights: Weights, cache: KVCache, expertCache: ExpertCache? = nil
    ) throws -> (hidden: [Float16], trace: Trace) {
        let width = spec.hiddenSize
        var stream = hidden

        // --- Attention block ---
        var normed = try timed("cpu:norm") {
            try normalise(stream, tokenCount: tokenCount, weight: weights.inputNorm)
        }
        let attended = try attention.forward(hidden: normed, tokenCount: tokenCount,
                                             positionBase: positionBase,
                                             weights: weights.attention, layer: index,
                                             cache: cache)
        timed("cpu:residual") {
            for i in 0..<(tokenCount * width) {
                stream[i] = Float16(Float(stream[i]) + Float(attended[i]))
            }
        }

        // --- Mixture-of-experts block ---
        normed = try timed("cpu:norm") {
            try normalise(stream, tokenCount: tokenCount, weight: weights.postNorm)
        }

        var decisions: [Router.Decision] = []
        decisions.reserveCapacity(tokenCount)
        // Fallback for callers without a persistent cache: reuse within this
        // batch only. An expert is ~12.6 MiB, so even that is worth doing.
        var local: [Int: ExpertRunner.Weights] = [:]
        let offsets = experts.sectionOffsets

        for token in 0..<tokenCount {
            let slice = Array(normed[(token * width)..<((token + 1) * width)])
            let decision = try router.route(hidden: slice,
                                            weight: weights.routerWeight,
                                            bias: weights.routerBias)
            decisions.append(decision)

            // Ask for all of this token's experts at once, so the planner sees
            // the whole working set and never evicts one it is about to need.
            var resolved: [ExpertRunner.Weights] = []
            if let expertCache, !Self.overlapDisabled {
                // Coarse overlap: miss reads start immediately and the GPU runs
                // the resident experts while they are in flight. Their outputs
                // are summed, so splitting the batch changes only the addition
                // order — FP16-noise, checked by the layer regression test.
                let unique = Array(Set(decision.experts))
                let acquisition = expertCache.acquireAsync(layer: index, experts: unique)
                var slotByExpert: [Int: Int] = [:]
                var expertIsMiss: [Int: Bool] = [:]
                for (position, expert) in unique.enumerated() {
                    slotByExpert[expert] = acquisition.slots[position]
                    expertIsMiss[expert] = acquisition.missPositions.contains(position)
                }

                var hitPairs: [(ExpertRunner.Weights, Float)] = []
                var missPairs: [(ExpertRunner.Weights, Float)] = []
                for (position, expert) in decision.experts.enumerated() {
                    let weights = ExpertRunner.Weights(
                        buffer: expertCache.buffer(layer: index, slot: slotByExpert[expert]!),
                        offsets: offsets)
                    if expertIsMiss[expert] == true {
                        missPairs.append((weights, decision.weights[position]))
                    } else {
                        hitPairs.append((weights, decision.weights[position]))
                    }
                }

                let hitOutput = try experts.applyBatch(slice, experts: hitPairs)
                try acquisition.wait()
                let missOutput = try experts.applyBatch(slice, experts: missPairs)

                var combined = [Float](repeating: 0, count: width)
                for i in 0..<width { combined[i] = hitOutput[i] + missOutput[i] }

                timed("cpu:residual") {
                    for i in 0..<width {
                        stream[token * width + i] = Float16(
                            Float(stream[token * width + i]) + combined[i])
                    }
                }
                continue
            }
            if let expertCache {
                let unique = Array(Set(decision.experts))
                let slots = try expertCache.acquire(layer: index, experts: unique)
                var slotByExpert: [Int: Int] = [:]
                for (expert, slot) in zip(unique, slots) { slotByExpert[expert] = slot }
                resolved = decision.experts.map { expert in
                    ExpertRunner.Weights(
                        buffer: expertCache.buffer(layer: index, slot: slotByExpert[expert]!),
                        offsets: offsets)
                }
            } else {
                for expert in decision.experts {
                    if let cached = local[expert] {
                        resolved.append(cached)
                    } else {
                        let loaded = try experts.loadWeights(layer: index, expert: expert)
                        local[expert] = loaded
                        resolved.append(loaded)
                    }
                }
            }

            let combined = try experts.applyBatch(
                slice,
                experts: resolved.enumerated().map { ($1, decision.weights[$0]) })

            timed("cpu:residual") {
                for i in 0..<width {
                    stream[token * width + i] = Float16(
                        Float(stream[token * width + i]) + combined[i])
                }
            }
        }

        return (stream, Trace(routing: decisions))
    }

    /// A/B switch for the read/compute overlap, for paired measurement.
    static let overlapDisabled = ProcessInfo.processInfo.environment["GODWIT_NO_OVERLAP"] != nil

    /// Runs `body`, timing it only when a profiler is attached.
    ///
    /// Written as a helper because the obvious spelling is a trap:
    /// `profiler?.measure(name) { work }` skips the work entirely when the
    /// profiler is nil, since optional chaining short-circuits the whole
    /// expression including its closure argument. That silently removed the
    /// residual connections whenever profiling was off, and only the layer
    /// regression test caught it.
    @inline(__always)
    private func timed<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        guard let profiler = context.profiler else { return try body() }
        return try profiler.measure(name, body)
    }

    /// RMSNorm on the CPU.
    ///
    /// One row per token over 2,880 values is trivial next to the reads this
    /// layer performs; the GPU kernel exists for when the rest of the loop stops
    /// round-tripping through host memory.
    private func normalise(
        _ values: [Float16], tokenCount: Int, weight: MTLBuffer
    ) throws -> [Float16] {
        let width = spec.hiddenSize
        let weights = weight.contents().bindMemory(to: UInt16.self, capacity: width)
        var out = [Float16](repeating: 0, count: values.count)

        for token in 0..<tokenCount {
            let base = token * width
            var sumSquares: Float = 0
            for i in 0..<width {
                let value = Float(values[base + i])
                sumSquares += value * value
            }
            let inverse = 1 / (sumSquares / Float(width) + spec.rmsNormEpsilon).squareRoot()
            for i in 0..<width {
                out[base + i] = Float16(
                    Float(values[base + i]) * inverse * BFloat16.toFloat(weights[i]))
            }
        }
        return out
    }
}
