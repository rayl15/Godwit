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
            // Qwen3's router has no bias. Zeros rather than a second kernel,
            // for the same reason the attention biases get them.
            routerBias: reader.manifest.spec.routerBias
                ? try reader.loadTrunk(section: "layer\(index).router_b", device: device)
                : try Self.zeroBuffer(count: reader.manifest.expertCount, device: device))
    }

    /// What the layer did, for tracing and cache accounting.
    public struct Trace: Sendable {
        public let routing: [Router.Decision]
    }

    /// Runs the layer over `tokenCount` tokens starting at `positionBase`.
    ///
    /// `hidden` is the residual stream, `[tokenCount, hiddenSize]`, and is
    /// returned updated.
    /// Runs the layer.
    ///
    /// The residual stream is FP32, not FP16, and that is not a precision
    /// preference — it is a range requirement. GPT-OSS was trained in bfloat16,
    /// which reaches ~3e38; FP16 stops at 65504. Certain prompts drive
    /// activations past that, the stream becomes infinity, and the logits come
    /// out NaN — which surfaces as the model emitting token 0 forever rather
    /// than as any kind of error. Normalised values are still handed to the
    /// kernels as FP16, since RMSNorm bounds them.
    public func forward(
        hidden: [Float], tokenCount: Int, positionBase: Int,
        weights: Weights, cache: KVCache, expertCache: ExpertCache? = nil,
        routerInput: (([Float]) -> Void)? = nil
    ) throws -> (hidden: [Float], trace: Trace) {
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
                stream[i] += Float(attended[i])
            }
        }

        // --- Mixture-of-experts block ---
        normed = try timed("cpu:norm") {
            try normalise(stream, tokenCount: tokenCount, weight: weights.postNorm)
        }

        // `if let` rather than optional chaining: `f?(g())` short-circuits the
        // argument too, which is how a profiling call once silently removed the
        // work it was meant to measure.
        if let routerInput { routerInput(normed.map(Float.init)) }

        // Route every token first, then do the expert work grouped by expert.
        //
        // The order matters more than it looks. Iterating tokens and fetching
        // their four experts each re-reads the same 12.66 MiB blob whenever two
        // tokens share an expert — and adjacent tokens share 54% of theirs. For
        // a 23-token prompt that was 92 reads per layer where ~25 unique
        // experts would do, which is why prefill was no faster per token than
        // decoding one at a time.
        var decisions: [Router.Decision] = []
        decisions.reserveCapacity(tokenCount)
        for token in 0..<tokenCount {
            let slice = Array(normed[(token * width)..<((token + 1) * width)])
            decisions.append(try router.route(hidden: slice,
                                              weight: weights.routerWeight,
                                              bias: weights.routerBias))
        }

        let offsets = experts.sectionOffsets
        let normedBuffer = try context.buffer(normed)
        let accumulated = try context.emptyBuffer(of: Float.self, count: tokenCount * width)

        // expert -> the tokens that chose it, with their routing weights
        var demand: [Int: [(token: Int, weight: Float)]] = [:]
        for (token, decision) in decisions.enumerated() {
            for (position, expert) in decision.experts.enumerated() {
                demand[expert, default: []].append((token, decision.weights[position]))
            }
        }

        if let expertCache {
            // Only `slotCount` experts can be resident at once, so the unique
            // set is processed in groups that fit. Sorting by demand puts the
            // most-shared experts first, which costs nothing and keeps the
            // busiest ones together.
            let ordered = demand.keys.sorted {
                let left = demand[$0]!.count, right = demand[$1]!.count
                return left == right ? $0 < $1 : left > right
            }
            for group in stride(from: 0, to: ordered.count, by: expertCache.slotCount) {
                let chunk = Array(ordered[group..<min(group + expertCache.slotCount,
                                                      ordered.count)])
                // Start the misses reading, then run the experts already
                // resident while those are in flight. Reads are the majority of
                // decode, so any compute moved underneath them is free.
                let acquisition = expertCache.acquireAsync(layer: index, experts: chunk)
                var hits: [ExpertRunner.Assignment] = []
                var misses: [ExpertRunner.Assignment] = []
                for (position, expert) in chunk.enumerated() {
                    let expertWeights = ExpertRunner.Weights(
                        buffer: expertCache.buffer(layer: index, slot: acquisition.slots[position]),
                        offsets: offsets)
                    let target = acquisition.missPositions.contains(position)
                    for entry in demand[expert]! {
                        let assignment = ExpertRunner.Assignment(
                            weights: expertWeights, token: entry.token,
                            routingWeight: entry.weight)
                        if target { misses.append(assignment) } else { hits.append(assignment) }
                    }
                }
                try experts.applyAssignments(inputs: normedBuffer, assignments: hits,
                                             outputs: accumulated)
                try acquisition.wait()
                try experts.applyAssignments(inputs: normedBuffer, assignments: misses,
                                             outputs: accumulated)
            }
        } else {
            for (expert, entries) in demand {
                let expertWeights = try experts.loadWeights(layer: index, expert: expert)
                try experts.applyAssignments(
                    inputs: normedBuffer,
                    assignments: entries.map {
                        ExpertRunner.Assignment(weights: expertWeights, token: $0.token,
                                                routingWeight: $0.weight)
                    },
                    outputs: accumulated)
            }
        }

        timed("cpu:residual") {
            let combined = accumulated.contents().bindMemory(
                to: Float.self, capacity: tokenCount * width)
            for i in 0..<(tokenCount * width) {
                stream[i] += combined[i]
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
        _ values: [Float], tokenCount: Int, weight: MTLBuffer
    ) throws -> [Float16] {
        let width = spec.hiddenSize
        let weights = weight.contents().bindMemory(to: UInt16.self, capacity: width)
        var out = [Float16](repeating: 0, count: values.count)

        for token in 0..<tokenCount {
            let base = token * width
            var sumSquares: Float = 0
            for i in 0..<width {
                let value = values[base + i]
                sumSquares += value * value
            }
            let inverse = 1 / (sumSquares / Float(width) + spec.rmsNormEpsilon).squareRoot()
            for i in 0..<width {
                out[base + i] = Float16(values[base + i] * inverse * BFloat16.toFloat(weights[i]))
            }
        }
        return out
    }
}

extension TransformerLayer {
    /// A zero-filled BF16 buffer, for a bias the model does not have.
    static func zeroBuffer(count: Int, device: MTLDevice) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(count * 2, 4),
                                             options: .storageModeShared) else {
            throw ExpertBlobError.missingSection("zero bias buffer")
        }
        memset(buffer.contents(), 0, buffer.length)
        return buffer
    }
}
