import Foundation
import Godwit

// GPT-OSS-120B decode budget, from docs/ESTIMATE.md. Every token multiplies
// this many quantised expert weights, and it has to fit inside the time the
// expert reads take, or compute becomes the bottleneck instead of I/O.
let expertParams = 2 * 2880 * 2880 + 2880 * 2880
let weightsPerToken = Double(4 * 36 * expertParams)
let ioBudgetSeconds = 0.225   // 8 cache slots, measured NVMe bandwidth
let requiredWeightsPerSecond = weightsPerToken / ioBudgetSeconds

func runDequantBenchmark(only: String?, rows: Int, cols: Int, threadgroups: Int) {
    do {
        let context = try MetalContext()
        let benchmark = DequantGEMVBenchmark(context: context)

        // One configuration per process, on purpose.
        //
        // Running every variant in one process made each one heat the GPU for
        // the next: the *unchanged* naive baseline measured 27 G w/s when five
        // configs preceded it and 13 when eight did. Position in the run was
        // worth 2x, which is larger than any difference between the kernels,
        // so the comparison was meaningless. Whatever runs must run alone.
        let result: DequantGEMVBenchmark.Result
        switch only {
        case "naive", nil:
            result = try benchmark.runMXFP4(rows: rows, cols: cols,
                                            iterations: 200, validate: true)
        case "affine":
            result = try benchmark.runAffineInt4(rows: rows, cols: cols, iterations: 200)
        case "persistent":
            result = try benchmark.runMXFP4Persistent(rows: rows, cols: cols,
                                                      iterations: 200,
                                                      threadgroups: threadgroups)
        case "staged":
            result = try benchmark.runMXFP4Persistent(rows: rows, cols: cols,
                                                      iterations: 200,
                                                      threadgroups: threadgroups, variant: .staged)
        case "lut":
            result = try benchmark.runMXFP4Persistent(rows: rows, cols: cols,
                                                      iterations: 200,
                                                      threadgroups: threadgroups, variant: .lut)
        default:
            FileHandle.standardError.write(Data(
                "unknown variant '\(only!)' — use naive, affine, persistent, staged\n".utf8))
            exit(2)
        }

        print(String(format: "%-28@ %5dx%-5d  %6.2f G w/s  %6.2f GiB/s",
                     result.label as NSString, rows, cols,
                     result.weightsPerSecond / 1e9, result.gibPerSecond))
        fflush(stdout)
    } catch {
        FileHandle.standardError.write(Data("benchmark failed: \(error)\n".utf8))
        exit(1)
    }
}

func runKernelAB(rows: Int, cols: Int, pairs: Int) {
    do {
        let context = try MetalContext()
        let benchmark = DequantGEMVBenchmark(context: context)
        let result = try benchmark.compareInterleaved(
            { r, c, i in try benchmark.runMXFP4(rows: r, cols: c, iterations: i, validate: false) },
            { r, c, i in try benchmark.runMXFP4Persistent(rows: r, cols: c, iterations: i,
                                                          variant: .multirow) },
            rows: rows, cols: cols, pairs: pairs)

        print("A: \(result.labelA)")
        print("B: \(result.labelB)")
        print("shape \(rows)x\(cols), \(result.pairs) interleaved pairs\n")
        print(String(format: "median B/A ratio  %.3f", result.medianRatio))
        print(String(format: "B faster in       %d of %d pairs", result.bWins, result.pairs))
        let p = result.signTestP
        print(String(format: "sign test p       %.4f", p))
        if p < 0.01 {
            print(result.medianRatio > 1
                  ? "\nB is faster — real effect"
                  : "\nB is SLOWER — real effect")
        } else {
            print("\nno detectable difference on this hardware")
        }
    } catch {
        FileHandle.standardError.write(Data("A/B failed: \(error)\n".utf8))
        exit(1)
    }
}

func runLogits(model: String, tokens: [Int], topK: Int) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let runner = ModelRunner(context: context, reader: reader)

        FileHandle.standardError.write(Data("loading resident weights...\n".utf8))
        let loadStart = Date()
        let weights = try runner.loadWeights()
        FileHandle.standardError.write(Data(String(
            format: "loaded in %.1fs\n", Date().timeIntervalSince(loadStart)).utf8))

        let cache = try runner.makeCache(maxContext: tokens.count + 16)
        let started = Date()
        let logits = try runner.logits(tokens: tokens, cache: cache,
                                       weights: weights) { done, total in
            FileHandle.standardError.write(Data("\rlayer \(done)/\(total)".utf8))
        }
        FileHandle.standardError.write(Data("\n".utf8))
        let elapsed = Date().timeIntervalSince(started)

        let ranked = logits.enumerated().sorted { $0.element > $1.element }.prefix(topK)
        // Softmax over the top-k only, for a readable sense of confidence.
        let peak = ranked.first?.element ?? 0
        let exps = ranked.map { exp(Double($0.element - peak)) }
        let total = exps.reduce(0, +)

        print(String(format: "%d prompt tokens, %.2fs (%.2fs/token)",
                     tokens.count, elapsed, elapsed / Double(tokens.count)))
        print("top \(topK) next-token candidates:")
        for (rank, entry) in ranked.enumerated() {
            print(String(format: "  %2d. id %-7d logit %8.3f  p(top%d) %5.1f%%",
                         rank + 1, entry.offset, entry.element, topK,
                         exps[rank] / total * 100))
        }
    } catch {
        FileHandle.standardError.write(Data("logits failed: \(error)\n".utf8))
        exit(1)
    }
}

func runGenerate(model: String, tokens: [Int], count: Int, stop: Set<Int> = [],
                 slots: Int = 8, profile: Bool = false, pageCache: Bool = false) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let runner = ModelRunner(context: context, reader: reader)
        let weights = try runner.loadWeights()

        let cache = try runner.makeCache(maxContext: tokens.count + count + 16)
        let expertCache = try runner.makeExpertCache(slots: slots,
                                                     bypassPageCache: !pageCache)
        var profiler: Profiler?
        if profile {
            let p = Profiler()
            context.profiler = p
            expertCache.profiler = p
            profiler = p
        }
        FileHandle.standardError.write(Data(String(
            format: "expert cache: %d slots x %d layers = %.2f GiB\n",
            expertCache.slotCount, expertCache.layerCount,
            Double(expertCache.byteCount) / 1_073_741_824).utf8))
        var produced: [Int] = []

        // Prefill: the whole prompt in one pass, filling the cache.
        let prefillStart = Date()
        var logits = try runner.logits(tokens: tokens, positionBase: 0,
                                       cache: cache, weights: weights,
                                       expertCache: expertCache)
        let prefill = Date().timeIntervalSince(prefillStart)
        FileHandle.standardError.write(Data(String(
            format: "prefill %d tokens in %.2fs (%.1f tok/s)\n",
            tokens.count, prefill, Double(tokens.count) / prefill).utf8))

        // Prefill's reads used to be discarded here, so every cache figure this
        // tool has ever printed described decode only. For a short generation
        // prefill is the larger share of the I/O — a 15-token prompt touches
        // most of each layer's experts before a single token is produced.
        let prefillStats = expertCache.stats
        FileHandle.standardError.write(Data(String(
            format: "  prefill reads: %d experts, %.2f GiB (%.1f%% of them cached)\n",
            prefillStats.hits + prefillStats.misses,
            Double(prefillStats.bytesRead) / 1_073_741_824,
            prefillStats.hitRate * 100).utf8))

        // Decode: one token at a time against the cache.
        expertCache.resetStats()
        profiler?.reset()
        let decodeStart = Date()
        for step in 0..<count {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            produced.append(best)
            if stop.contains(best) { break }
            if step == count - 1 { break }
            logits = try runner.logits(tokens: [best], positionBase: cache.length,
                                       cache: cache, weights: weights,
                                       expertCache: expertCache)
            let rate = Double(step + 1) / Date().timeIntervalSince(decodeStart)
            FileHandle.standardError.write(Data(String(
                format: "\rdecode %d/%d  (%.2f tok/s)", step + 1, count, rate).utf8))
        }
        let decode = Date().timeIntervalSince(decodeStart)
        FileHandle.standardError.write(Data(String(
            format: "\ndecode %d tokens in %.2fs (%.2f tok/s)\n",
            count, decode, Double(count) / decode).utf8))
        if let profiler {
            profiler.setTotal(decode)
            FileHandle.standardError.write(Data(("\n" + profiler.report() + "\n").utf8))
        }
        let s = expertCache.stats
        FileHandle.standardError.write(Data(String(
            format: "expert cache: %.1f%% hit (%d hits, %d misses), %.2f GiB read\n",
            s.hitRate * 100, s.hits, s.misses,
            Double(s.bytesRead) / 1_073_741_824).utf8))

        print(produced.map(String.init).joined(separator: ","))
    } catch {
        FileHandle.standardError.write(Data("generate failed: \(error)\n".utf8))
        exit(1)
    }
}

func runTokenize(model: String, text: String) {
    do {
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let started = Date()
        let tokenizer = try reader.loadTokenizer()
        let load = Date().timeIntervalSince(started)

        let ids = tokenizer.encode(text)
        let round = tokenizer.decode(ids)
        print(String(format: "loaded %d entries in %.2fs", tokenizer.count, load))
        print("text   : \(text.debugDescription)")
        print("ids    : \(ids.map(String.init).joined(separator: ","))")
        print("pieces : \(ids.map { tokenizer.decode([$0]).debugDescription }.joined(separator: " "))")
        print("decoded: \(round.debugDescription)")
        print(round == text ? "\nround-trip exact" : "\nROUND-TRIP MISMATCH")
        if round != text { exit(1) }
    } catch {
        FileHandle.standardError.write(Data("tokenize failed: \(error)\n".utf8))
        exit(1)
    }
}

func runChat(model: String, prompt: String?, system: String, count: Int,
             slots: Int, settings: Sampler.Settings, showAnalysis: Bool) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let tokenizer = try reader.loadTokenizer()
        let runner = ModelRunner(context: context, reader: reader)

        FileHandle.standardError.write(Data("loading \(reader.manifest.model)...\n".utf8))
        let weights = try runner.loadWeights()
        let expertCache = try runner.makeExpertCache(slots: slots)
        let stops = Conversation.stopTokens(tokenizer)

        var conversation = Conversation(system: system)
        let interactive = prompt == nil
        if interactive {
            FileHandle.standardError.write(Data(
                "ready. blank line or /exit to quit, /reset to clear history.\n\n".utf8))
        }

        while true {
            let question: String
            if let prompt, conversation.messages.count <= 1 {
                question = prompt
            } else if interactive {
                FileHandle.standardError.write(Data("> ".utf8))
                guard let line = readLine(), !line.isEmpty else { break }
                if line == "/exit" { break }
                if line == "/reset" {
                    conversation.reset()
                    FileHandle.standardError.write(Data("history cleared\n\n".utf8))
                    continue
                }
                question = line
            } else { break }

            conversation.append(Conversation.Message(.user, question))
            let ids = try conversation.encode(with: tokenizer)

            // Each turn re-encodes the whole conversation, so the cache is
            // rebuilt from scratch. Reusing it across turns needs prefix
            // matching, which is the obvious next step.
            let cache = try runner.makeCache(maxContext: ids.count + count + 16)
            var sampler = Sampler(settings: settings)

            let prefillStart = Date()
            var logits = try runner.logits(tokens: ids, positionBase: 0, cache: cache,
                                           weights: weights, expertCache: expertCache)
            let prefill = Date().timeIntervalSince(prefillStart)

            var produced: [Int] = []
            var emitted = ""
            let decodeStart = Date()
            for step in 0..<count {
                let next = sampler.pick(from: logits, history: produced)
                if stops.contains(next) { break }
                produced.append(next)

                // Decode the whole run each time: a multi-byte character can
                // span tokens, so decoding one at a time would print mojibake.
                // Stream only the answer channel unless asked for the reasoning,
                // and emit the delta so nothing is printed twice.
                let text = tokenizer.decode(produced)
                let visible = showAnalysis ? text : Conversation.split(text).final
                // Only append when the new text genuinely extends what was
                // printed. Channel markers resolve mid-stream and can shorten
                // or replace the visible span, and diffing by length alone
                // splices fragments together.
                if visible.hasPrefix(emitted) {
                    if visible.count > emitted.count {
                        print(String(visible.dropFirst(emitted.count)), terminator: "")
                        fflush(stdout)
                    }
                } else if !visible.isEmpty {
                    print("\r\u{1B}[2K" + visible, terminator: "")
                    fflush(stdout)
                }
                emitted = visible
                if step == count - 1 { break }
                logits = try runner.logits(tokens: [next], positionBase: cache.length,
                                           cache: cache, weights: weights,
                                           expertCache: expertCache)
            }
            let decode = Date().timeIntervalSince(decodeStart)

            let reply = tokenizer.decode(produced)
            let parts = Conversation.split(reply)
            conversation.append(Conversation.Message(.assistant, parts.final))

            // A turn can spend its whole budget reasoning and never reach an
            // answer — Qwen3 does this readily, and GPT-OSS on open questions.
            // Printing nothing at all is indistinguishable from a crash, so say
            // what happened and show the reasoning that was produced.
            if parts.final.isEmpty, !showAnalysis {
                let stopped = produced.count >= count
                FileHandle.standardError.write(Data(
                    (stopped
                     ? "\n[stopped at the \(count)-token limit while still "
                       + "reasoning, before reaching an answer]\n"
                     : "\n[ended without reaching an answer]\n").utf8))
                if let analysis = parts.analysis, !analysis.isEmpty {
                    print(analysis)
                }
            }

            print()
            FileHandle.standardError.write(Data(String(
                format: "\n[%d prompt tokens %.1fs · %d generated %.2f tok/s]\n\n",
                ids.count, prefill, produced.count,
                Double(produced.count) / max(decode, 0.001)).utf8))

            if prompt != nil { break }
        }
    } catch {
        FileHandle.standardError.write(Data("chat failed: \(error)\n".utf8))
        exit(1)
    }
}

func runServe(model: String, port: UInt16, slots: Int,
              settings: Sampler.Settings, maxTokens: Int) {
    do {
        FileHandle.standardError.write(Data("loading model...\n".utf8))
        // `--model` names one install; the registry root is its parent, so the
        // dashboard can offer whatever else is installed alongside it.
        let target = URL(fileURLWithPath: model)
        let server = try ChatServer(root: target.deletingLastPathComponent(),
                                    initial: FileManager.default.fileExists(
                                        atPath: target.appendingPathComponent("manifest.json").path)
                                        ? target : nil,
                                    slots: slots, settings: settings, maxTokens: maxTokens)
        try server.listen(port: port)
    } catch {
        FileHandle.standardError.write(Data("serve failed: \(error)\n".utf8))
        exit(1)
    }
}

func runRange(model: String, output: String, slots: Int) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let expertRange = ExpertRange(context: context, reader: reader)

        FileHandle.standardError.write(Data("probing the router...\n".utf8))
        let started = Date()
        let result = try expertRange.build(slots: slots) { topic, done, total in
            FileHandle.standardError.write(Data(String(
                format: "  %2d/%2d  %@\n", done, total, topic as NSString).utf8))
        }

        let encoder = JSONEncoder()
        try encoder.encode(result).write(to: URL(fileURLWithPath: output))

        let specialists = result.points.filter { $0.specialisation > 0.35 }
        var byTopic: [String: Int] = [:]
        for point in specialists { byTopic[point.topic, default: 0] += 1 }

        print(String(format: "\n%d experts characterised in %.1f min",
                     result.points.count, Date().timeIntervalSince(started) / 60))
        print(String(format: "axes explain %.0f%% / %.0f%% / %.0f%% of variance",
                     (result.variance.first ?? 0) * 100,
                     result.variance.count > 1 ? result.variance[1] * 100 : 0,
                     result.variance.count > 2 ? result.variance[2] * 100 : 0))
        print("\n\(specialists.count) specialists (concentration > 0.35):")
        for (topic, count) in byTopic.sorted(by: { $0.value > $1.value }) {
            print(String(format: "  %-10@ %4d", topic as NSString, count))
        }
        print("\nwritten to \(output)")
    } catch {
        FileHandle.standardError.write(Data("range failed: \(error)\n".utf8))
        exit(1)
    }
}

func runExpertVerification(directory: String) {
    do {
        let context = try MetalContext()
        let report = try ExpertVerification(context: context)
            .run(directory: URL(fileURLWithPath: directory))

        let mib = Double(report.bytesRead) / 1_048_576
        print("expert  \(report.rows) x \(report.cols)")
        print(String(format: "read    %.2f MiB in %.1f ms (%.2f GiB/s)",
                     mib, report.readSeconds * 1000,
                     mib / 1024 / report.readSeconds))
        print(String(format: "compute %.2f ms", report.computeSeconds * 1000))
        print(String(format: "error   max %.3e  mean %.3e  (worst row %d, |y| ~ %.3f)",
                     report.maxRelativeError, report.meanRelativeError,
                     report.worstRow, report.referenceMagnitude))
        print(report.passed
              ? "\nPASS — Metal output matches the NumPy reference on real weights"
              : "\nFAIL — output diverges from the reference")
        if !report.passed { exit(1) }
    } catch {
        FileHandle.standardError.write(Data("verification failed: \(error)\n".utf8))
        exit(1)
    }
}

func runInstall(directory: String, layerLimit: Int?, repository: String?) async {
    do {
        // The spec is chosen by repository, so installing a different model is
        // a flag rather than a code change. That is the claim being tested.
        //
        // Resolved through ModelRegistry rather than by substring here: an
        // unrecognised repository used to fall through to GPT-OSS-120B, which
        // meant a typo installed a real model under the wrong architecture and
        // said nothing until the weights were already on disk.
        let name = repository ?? "openai/gpt-oss-120b"
        guard let spec = ModelRegistry.spec(for: name) else {
            let known = ModelRegistry.catalogue
                .map { "  \($0.id)  \($0.title)" }.joined(separator: "\n")
            FileHandle.standardError.write(Data(
                "unknown model '\(name)'. Known:\n\(known)\n".utf8))
            exit(2)
        }
        let options = Installer.Options(repository: name,
                                        layerLimit: layerLimit)
        let installer = Installer(spec: spec, options: options)
        let target = URL(fileURLWithPath: directory)

        let started = Date()
        print("installing \(options.repository) -> \(directory)")
        if let layerLimit { print("limited to the first \(layerLimit) layers") }

        let manifest = try await installer.install(to: target) { progress in
            let gib = Double(progress.bytesWritten) / 1_073_741_824
            print(String(format: "  %-8@ %3d/%-3d  %7.2f GiB",
                         progress.stage as NSString,
                         progress.completed, progress.total, gib))
            // An install runs for hours and is usually redirected to a file,
            // where Swift buffers stdout and the progress never appears.
            fflush(stdout)
        }
        try manifest.validate(at: target)

        let minutes = Date().timeIntervalSince(started) / 60
        print(String(format: "\ndone in %.1f min — %.2f GiB, %d layers%@",
                     minutes,
                     Double(manifest.bytesWritten) / 1_073_741_824,
                     manifest.layerCount,
                     manifest.isComplete ? "" : " (PARTIAL install)"))
    } catch {
        FileHandle.standardError.write(Data("install failed: \(error)\n".utf8))
        exit(1)
    }
}

func runExpertCheck(model: String, reference: String) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let runner = ExpertRunner(context: context, reader: reader)

        let refDir = URL(fileURLWithPath: reference)
        let caseData = try Data(contentsOf: refDir.appendingPathComponent("case.json"))
        let info = try JSONSerialization.jsonObject(with: caseData) as! [String: Any]
        let layer = info["layer"] as! Int
        let expert = info["expert"] as! Int
        let hidden = info["hiddenSize"] as! Int

        func load<T>(_ name: String, _ type: T.Type, _ count: Int) throws -> [T] {
            let data = try Data(contentsOf: refDir.appendingPathComponent(name))
            return data.withUnsafeBytes {
                Array(UnsafeBufferPointer(start: $0.bindMemory(to: T.self).baseAddress!,
                                          count: count))
            }
        }
        let x: [Float16] = try load("x.f16", Float16.self, hidden)
        let expected: [Float] = try load("y.f32", Float.self, hidden)

        let weights = try runner.loadWeights(layer: layer, expert: expert)
        let produced = try runner.apply(x, weights: weights)

        var magnitude: Float = 0
        for value in expected { magnitude += abs(value) }
        magnitude = max(magnitude / Float(expected.count), 1e-6)

        var worst: Float = 0
        var worstIndex = 0
        for i in expected.indices {
            let error = abs(produced[i] - expected[i]) / magnitude
            if error > worst { worst = error; worstIndex = i }
        }

        print("layer \(layer) expert \(expert), \(hidden) outputs")
        print(String(format: "max relative error %.3e at index %d (|y| ~ %.4f)",
                     worst, worstIndex, magnitude))
        if worst < 5e-3 {
            print("\nPASS — GPU expert matches the NumPy reference")
        } else {
            print("\nFAIL — expert output diverges")
            exit(1)
        }
    } catch {
        FileHandle.standardError.write(Data("expert check failed: \(error)\n".utf8))
        exit(1)
    }
}

func runRoutingTrace(model: String, layer: Int, tokens: Int) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let report = try RoutingTrace(context: context, reader: reader)
            .run(layer: layer, tokenCount: tokens)

        print("layer \(layer): \(report.tokensRouted) tokens, top-\(report.topK) of \(report.expertCount)\n")
        print(String(format: "top-decile share   %.1f%%   (10.0%% would be uniform)",
                     report.topDecileShare * 100))
        print(String(format: "normalised entropy %.3f    (1.000 would be uniform)",
                     report.normalisedEntropy))
        print("experts never used \(report.unusedExperts) of \(report.expertCount)\n")

        print("slots  hit rate   reads/token (36 layers)")
        for entry in report.hitRates {
            let reads = Double(report.topK * 36) * (1 - entry.rate)
            print(String(format: "%5d   %6.1f%%   %6.1f", entry.slots, entry.rate * 100, reads))
        }

        let sorted = report.selectionCounts.enumerated().sorted { $0.element > $1.element }
        let hottest = sorted.prefix(5).map { "\($0.offset)(\($0.element))" }.joined(separator: " ")
        print("\nhottest experts: \(hottest)")
    } catch {
        FileHandle.standardError.write(Data("routing trace failed: \(error)\n".utf8))
        exit(1)
    }
}

func runAttentionCheck(model: String, reference: String) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let spec = reader.manifest.spec
        let rope = RoPE(configuration: .forSpec(spec))
        let attention = Attention(context: context, spec: spec, rope: rope)

        let refDir = URL(fileURLWithPath: reference)
        let info = try JSONSerialization.jsonObject(
            with: Data(contentsOf: refDir.appendingPathComponent("case.json"))) as! [String: Any]
        let layer = info["layer"] as! Int
        let tokens = info["tokens"] as! Int
        let hiddenSize = info["hiddenSize"] as! Int

        func load<T>(_ name: String, _ type: T.Type, _ count: Int) throws -> [T] {
            let data = try Data(contentsOf: refDir.appendingPathComponent(name))
            return data.withUnsafeBytes {
                Array(UnsafeBufferPointer(start: $0.bindMemory(to: T.self).baseAddress!,
                                          count: count))
            }
        }
        let hidden: [Float16] = try load("hidden.f16", Float16.self, tokens * hiddenSize)
        let expected: [Float] = try load("y.f32", Float.self, tokens * hiddenSize)

        let weights = try Attention.loadWeights(reader: reader, layer: layer,
                                                device: context.device)
        let cache = try KVCache(context: context, spec: reader.manifest.spec,
                                maxContext: max(tokens, 128), layerCount: layer + 1)
        let produced = try attention.forward(hidden: hidden, tokenCount: tokens,
                                             positionBase: 0, weights: weights,
                                             layer: layer, cache: cache)

        // The output distribution has a long tail -- the largest element is
        // ~40x the mean -- so normalising by the mean inflates errors on big
        // elements into apparent percentages. RMS is the honest scale.
        var sumSquares: Float = 0
        for value in expected { sumSquares += value * value }
        let rms = max((sumSquares / Float(expected.count)).squareRoot(), 1e-6)

        var worst: Float = 0
        var errorSquares: Float = 0
        for i in expected.indices {
            let delta = abs(Float(produced[i]) - expected[i])
            errorSquares += delta * delta
            worst = max(worst, delta / rms)
        }
        let relativeRMS = (errorSquares / Float(expected.count)).squareRoot() / rms

        // The tolerance is derived from the number format rather than picked.
        // Outputs are FP16, whose significand is 11 bits, so the largest element
        // carries an unavoidable quantisation of maxAbs * 2^-11. Expressed
        // against RMS — and this output's peak is ~26x its RMS — that alone
        // permits over 1e-2. A fixed threshold below that would fail a correct
        // kernel; two ULP leaves room for accumulation without hiding a bug.
        let maxAbs = expected.map(abs).max() ?? 1
        let ulpBound = maxAbs * Float(exp2(-11.0)) / rms
        let maxTolerance = 2 * ulpBound
        let rmsTolerance: Float = 1e-3

        let sliding = info["sliding"] as! Bool
        print("layer \(layer), \(tokens) tokens, \(sliding ? "sliding" : "full") attention")
        print(String(format: "error vs RMS: max %.3e (bound %.3e), RMS %.3e (bound %.1e)",
                     worst, maxTolerance, relativeRMS, rmsTolerance))
        print(String(format: "|y| RMS %.4f, peak %.3f (%.1fx RMS)", rms, maxAbs, maxAbs / rms))

        if relativeRMS < rmsTolerance && worst < maxTolerance {
            print("\nPASS — GPU attention matches the NumPy reference within FP16 resolution")
        } else {
            print("\nFAIL — attention diverges beyond what the format explains")
            exit(1)
        }
    } catch {
        FileHandle.standardError.write(Data("attention check failed: \(error)\n".utf8))
        exit(1)
    }
}

func runLayerCheck(model: String, reference: String) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let rope = RoPE(configuration: .forSpec(reader.manifest.spec))

        let refDir = URL(fileURLWithPath: reference)
        let info = try JSONSerialization.jsonObject(
            with: Data(contentsOf: refDir.appendingPathComponent("case.json"))) as! [String: Any]
        let layerIndex = info["layer"] as! Int
        let tokens = info["tokens"] as! Int
        let hiddenSize = info["hiddenSize"] as! Int
        let expectedRouting = info["routing"] as! [[Int]]

        func load<T>(_ name: String, _ count: Int) throws -> [T] {
            let data = try Data(contentsOf: refDir.appendingPathComponent(name))
            return data.withUnsafeBytes {
                Array(UnsafeBufferPointer(start: $0.bindMemory(to: T.self).baseAddress!,
                                          count: count))
            }
        }
        let hidden: [Float16] = try load("hidden.f16", tokens * hiddenSize)
        let expected: [Float16] = try load("y.f16", tokens * hiddenSize)

        let layer = TransformerLayer(context: context, reader: reader,
                                     index: layerIndex, rope: rope)
        let weights = try layer.loadWeights()
        // A fresh cache, so attending over it is equivalent to attending within
        // the batch — which keeps this a valid regression test against the
        // reference generated before the cache existed.
        let cache = try KVCache(context: context, spec: reader.manifest.spec,
                                maxContext: max(tokens, 128), layerCount: layerIndex + 1)
        let started = Date()
        // The reference stores FP16; the runtime's residual stream is FP32.
        let (producedWide, trace) = try layer.forward(
            hidden: hidden.map(Float.init), tokenCount: tokens, positionBase: 0,
            weights: weights, cache: cache)
        let produced = producedWide.map(Float16.init)
        let elapsed = Date().timeIntervalSince(started)

        // Routing must match exactly. It is a discrete choice, so any drift
        // means a different set of experts ran, not a rounding difference.
        var routingMatches = true
        for token in 0..<tokens where Array(trace.routing[token].experts) != expectedRouting[token] {
            routingMatches = false
            print("routing mismatch at token \(token): "
                  + "\(trace.routing[token].experts) vs \(expectedRouting[token])")
        }

        var sumSquares: Float = 0
        for value in expected { sumSquares += Float(value) * Float(value) }
        let rms = max((sumSquares / Float(expected.count)).squareRoot(), 1e-6)
        var worst: Float = 0
        var errorSquares: Float = 0
        for i in expected.indices {
            let delta = abs(Float(produced[i]) - Float(expected[i]))
            errorSquares += delta * delta
            worst = max(worst, delta / rms)
        }
        let relativeRMS = (errorSquares / Float(expected.count)).squareRoot() / rms
        let maxAbs = expected.map { abs(Float($0)) }.max() ?? 1
        let bound = 4 * maxAbs * Float(exp2(-11.0)) / rms

        print("layer \(layerIndex), \(tokens) tokens, \(String(format: "%.2f", elapsed))s")
        print("routing: \(routingMatches ? "exact match" : "MISMATCH")")
        print(String(format: "error vs RMS: max %.3e (bound %.3e), RMS %.3e",
                     worst, bound, relativeRMS))
        print(String(format: "|y| RMS %.4f, peak %.2f", rms, maxAbs))

        if routingMatches && relativeRMS < 2e-3 && worst < bound {
            print("\nPASS — full layer matches the NumPy reference")
        } else {
            print("\nFAIL — layer diverges")
            exit(1)
        }
    } catch {
        FileHandle.standardError.write(Data("layer check failed: \(error)\n".utf8))
        exit(1)
    }
}

func runLayerTrace(model: String, layers: Int, tokens: Int) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let available = min(layers, reader.manifest.layerCount)
        let report = try LayerTrace(context: context, reader: reader)
            .run(layerCount: available, tokenCount: tokens)

        print("\(available) layers x \(report.tokenCount) tokens, "
              + "top-\(report.topK) of \(report.expertCount) "
              + String(format: "(%.1fs)", report.seconds))
        print(String(format: "chance overlap: %.1f%%\n", report.chanceOverlap * 100))

        let adjacent = report.adjacentOverlap
        if !adjacent.isEmpty {
            let mean = adjacent.reduce(0, +) / Double(adjacent.count)
            print("=== can layer n predict layer n+1? ===")
            print(String(format: "mean overlap %.1f%%  (min %.1f%%, max %.1f%%)",
                         mean * 100, (adjacent.min() ?? 0) * 100, (adjacent.max() ?? 0) * 100))
            print(String(format: "vs chance    %.1f%%  -> %@",
                         report.chanceOverlap * 100,
                         mean > report.chanceOverlap * 3
                            ? "predictable; prefetch is worth building" as NSString
                            : "barely above chance; prefetch will not pay" as NSString))
        }

        let sameLayer = report.tokenToTokenOverlap
        if !sameLayer.isEmpty {
            let mean = sameLayer.reduce(0, +) / Double(sameLayer.count)
            print(String(format: "\n=== control: same layer, adjacent tokens ===\nmean overlap %.1f%%",
                         mean * 100))
        }

        // A high hit rate is only meaningful if routing is actually varying.
        // If the residual stream degenerates and every token picks the same
        // experts, the cache looks brilliant and the measurement is worthless.
        var distinctPerLayer: [Int] = []
        var collapsed = 0
        for layer in 0..<available {
            let used = Set(report.routing[layer].flatMap { $0 })
            distinctPerLayer.append(used.count)
            let first = Set(report.routing[layer][0])
            if report.routing[layer].allSatisfy({ Set($0) == first }) { collapsed += 1 }
        }
        let meanDistinct = Double(distinctPerLayer.reduce(0, +)) / Double(available)
        let ceiling = min(report.tokenCount * report.topK, report.expertCount)
        print(String(format: "\n=== is routing actually varying? ===\n"
                     + "distinct experts per layer: mean %.1f of %d "
                     + "(ceiling %d for %d tokens)",
                     meanDistinct, report.expertCount, ceiling, report.tokenCount))
        print("layers where every token routed identically: \(collapsed) of \(available)")

        print("\n=== cache hit rate on the real trace ===")
        print("slots  hit rate")
        for entry in report.hitRates(slots: [4, 6, 8, 12, 16]) {
            print(String(format: "%5d   %6.1f%%", entry.slots, entry.rate * 100))
        }
    } catch {
        FileHandle.standardError.write(Data("layer trace failed: \(error)\n".utf8))
        exit(1)
    }
}

let arguments = Array(CommandLine.arguments.dropFirst())

switch arguments.first {
case "version":
    print("godwit 0.0.1-dev")
case "bench" where arguments.dropFirst().first == "dequant":
    var only: String?
    var benchRows = 5760
    var benchCols = 2880
    var benchTG = 64
    var benchFlags = arguments.dropFirst(2).makeIterator()
    while let flag = benchFlags.next() {
        switch flag {
        case "--only": only = benchFlags.next()
        case "--rows": benchRows = benchFlags.next().flatMap(Int.init) ?? 5760
        case "--cols": benchCols = benchFlags.next().flatMap(Int.init) ?? 2880
        case "--threadgroups": benchTG = benchFlags.next().flatMap(Int.init) ?? 64
        default:
            FileHandle.standardError.write(Data("unknown flag: \(flag)\n".utf8))
            exit(2)
        }
    }
    runDequantBenchmark(only: only, rows: benchRows, cols: benchCols, threadgroups: benchTG)
case "install":
    var target: String?
    var limit: Int?
    var repo: String?
    var rest = arguments.dropFirst().makeIterator()
    while let flag = rest.next() {
        switch flag {
        case "--output", "-o": target = rest.next()
        case "--layers": limit = rest.next().flatMap(Int.init)
        case "--model-id": repo = rest.next()
        default:
            FileHandle.standardError.write(Data("unknown flag: \(flag)\n".utf8))
            exit(2)
        }
    }
    guard let target else {
        FileHandle.standardError.write(Data(
            "usage: godwit install --output <dir> [--model-id repo] [--layers N]\n".utf8))
        exit(2)
    }
    await runInstall(directory: target, layerLimit: limit, repository: repo)
case "trace-layers":
    var traceModel: String?
    var traceLayers = 36
    var traceTokens = 24
    var traceFlags = arguments.dropFirst().makeIterator()
    while let flag = traceFlags.next() {
        switch flag {
        case "--model", "-m": traceModel = traceFlags.next()
        case "--layers": traceLayers = traceFlags.next().flatMap(Int.init) ?? 36
        case "--tokens": traceTokens = traceFlags.next().flatMap(Int.init) ?? 24
        default:
            FileHandle.standardError.write(Data("unknown flag: \(flag)\n".utf8))
            exit(2)
        }
    }
    guard let traceModel else {
        FileHandle.standardError.write(Data(
            "usage: godwit trace-layers --model <dir> [--layers N] [--tokens N]\n".utf8))
        exit(2)
    }
    runLayerTrace(model: traceModel, layers: traceLayers, tokens: traceTokens)
case "dump-hidden":
    let dhArgs = Array(arguments.dropFirst())
    guard dhArgs.count >= 3 else {
        FileHandle.standardError.write(Data(
            "usage: godwit dump-hidden <gwt-dir> <prompt> <out.bin> [tokens]\n".utf8))
        exit(2)
    }
    runHiddenDump(model: dhArgs[0], prompt: dhArgs[1], output: dhArgs[2],
                  tokens: dhArgs.count > 3 ? Int(dhArgs[3]) ?? 32 : 32)
case "dump-routing":
    let drArgs = Array(arguments.dropFirst())
    guard drArgs.count >= 2 else {
        FileHandle.standardError.write(Data(
            "usage: godwit dump-routing <gwt-dir> <prompt> [tokens]\n".utf8))
        exit(2)
    }
    runRoutingDump(model: drArgs[0], prompt: drArgs[1],
                   tokens: drArgs.count > 2 ? Int(drArgs[2]) ?? 32 : 32)
case "trace-routing":
    var model: String?
    var layer = 0
    var tokens = 2000
    var flags = arguments.dropFirst().makeIterator()
    while let flag = flags.next() {
        switch flag {
        case "--model", "-m": model = flags.next()
        case "--layer": layer = flags.next().flatMap(Int.init) ?? 0
        case "--tokens": tokens = flags.next().flatMap(Int.init) ?? 2000
        default:
            FileHandle.standardError.write(Data("unknown flag: \(flag)\n".utf8))
            exit(2)
        }
    }
    guard let model else {
        FileHandle.standardError.write(Data(
            "usage: godwit trace-routing --model <dir> [--layer N] [--tokens N]\n".utf8))
        exit(2)
    }
    runRoutingTrace(model: model, layer: layer, tokens: tokens)
case "check-mxfp4":
    let mxArgs = Array(arguments.dropFirst())
    guard mxArgs.count == 1 else {
        FileHandle.standardError.write(Data(
            "usage: godwit check-mxfp4 <gwt-dir>\n".utf8))
        exit(2)
    }
    runMXFP4Check(model: mxArgs[0])
case "check-layer":
    let layerArgs = Array(arguments.dropFirst())
    guard layerArgs.count == 2 else {
        FileHandle.standardError.write(Data(
            "usage: godwit check-layer <gwt-dir> <reference-dir>\n".utf8))
        exit(2)
    }
    runLayerCheck(model: layerArgs[0], reference: layerArgs[1])
case "check-attention":
    let attArgs = Array(arguments.dropFirst())
    guard attArgs.count == 2 else {
        FileHandle.standardError.write(Data(
            "usage: godwit check-attention <gwt-dir> <reference-dir>\n".utf8))
        exit(2)
    }
    runAttentionCheck(model: attArgs[0], reference: attArgs[1])
case "ab-kernel":
    var abRows = 5760, abCols = 2880, abPairs = 40
    var abFlags = arguments.dropFirst().makeIterator()
    while let flag = abFlags.next() {
        switch flag {
        case "--rows": abRows = abFlags.next().flatMap(Int.init) ?? 5760
        case "--cols": abCols = abFlags.next().flatMap(Int.init) ?? 2880
        case "--pairs": abPairs = abFlags.next().flatMap(Int.init) ?? 40
        default: break
        }
    }
    runKernelAB(rows: abRows, cols: abCols, pairs: abPairs)
case "generate":
    var genModel: String?
    var genTokens: [Int] = []
    var genCount = 16
    var genStop: [Int] = []
    var genSlots = 8
    var genProfile = false
    var genPageCache = false
    var genFlags = arguments.dropFirst().makeIterator()
    while let flag = genFlags.next() {
        switch flag {
        case "--model", "-m": genModel = genFlags.next()
        case "--tokens": genTokens = (genFlags.next() ?? "")
            .split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        case "--count", "-n": genCount = genFlags.next().flatMap(Int.init) ?? 16
        case "--slots": genSlots = genFlags.next().flatMap(Int.init) ?? 8
        case "--profile": genProfile = true
        case "--page-cache": genPageCache = true
        case "--stop": genStop = (genFlags.next() ?? "")
            .split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        default: break
        }
    }
    guard let genModel, !genTokens.isEmpty else {
        FileHandle.standardError.write(Data(
            "usage: godwit generate --model <dir> --tokens 1,2,3 [--count N]\n".utf8))
        exit(2)
    }
    runGenerate(model: genModel, tokens: genTokens, count: genCount, stop: Set(genStop), slots: genSlots,
                profile: genProfile, pageCache: genPageCache)
case "range":
    var rangeModel: String?
    var rangeOut = "range.json"
    var rangeSlots = 8
    var rangeFlags = arguments.dropFirst().makeIterator()
    while let flag = rangeFlags.next() {
        switch flag {
        case "--model", "-m": rangeModel = rangeFlags.next()
        case "--output", "-o": rangeOut = rangeFlags.next() ?? "range.json"
        case "--slots": rangeSlots = rangeFlags.next().flatMap(Int.init) ?? 8
        default: break
        }
    }
    guard let rangeModel else {
        FileHandle.standardError.write(Data(
            "usage: godwit range --model <dir> [--output range.json]\n".utf8))
        exit(2)
    }
    runRange(model: rangeModel, output: rangeOut, slots: rangeSlots)
case "serve":
    var serveModel: String?
    var servePort: UInt16 = 8080
    var serveSlots = 8
    var serveMax = 1024
    var serveSettings = Sampler.Settings()
    var serveFlags = arguments.dropFirst().makeIterator()
    while let flag = serveFlags.next() {
        switch flag {
        case "--model", "-m": serveModel = serveFlags.next()
        case "--port": servePort = serveFlags.next().flatMap(UInt16.init) ?? 8080
        case "--slots": serveSlots = serveFlags.next().flatMap(Int.init) ?? 8
        case "--max", "-n": serveMax = serveFlags.next().flatMap(Int.init) ?? 1024
        case "--temperature", "-t":
            serveSettings.temperature = serveFlags.next().flatMap(Float.init) ?? 0.7
        // The README has claimed serve took these for a while; it did not.
        // Qwen3's recommended settings are top-p 0.95 and top-k 20, so the
        // dashboard could not be run the way its own model asks to be.
        case "--top-k": serveSettings.topK = serveFlags.next().flatMap(Int.init) ?? 40
        case "--top-p": serveSettings.topP = serveFlags.next().flatMap(Float.init) ?? 0.95
        case "--repetition-penalty":
            serveSettings.repetitionPenalty = serveFlags.next().flatMap(Float.init) ?? 1.1
        case "--seed":
            if let value = serveFlags.next().flatMap(UInt64.init) {
                serveSettings.seed = value
            }
        case "--greedy": serveSettings = .greedy
        case "--lookahead": setenv("GODWIT_LOOKAHEAD", "1", 1)
        default:
            FileHandle.standardError.write(Data("unknown flag: \(flag)\n".utf8))
            exit(2)
        }
    }
    guard let serveModel else {
        FileHandle.standardError.write(Data(
            ("usage: godwit serve --model <dir> [--port 8080] [--temperature N] "
             + "[--top-k N] [--top-p N] [--repetition-penalty N] [--seed N] [--greedy]\n").utf8))
        exit(2)
    }
    runServe(model: serveModel, port: servePort, slots: serveSlots,
             settings: serveSettings, maxTokens: serveMax)
case "chat":
    var chatModel: String?
    var chatPrompt: String?
    var chatSystem = "You are a helpful assistant."
    var chatCount = 1024
    var chatSlots = 8
    var chatSettings = Sampler.Settings()
    var showAnalysis = false
    var chatFlags = arguments.dropFirst().makeIterator()
    while let flag = chatFlags.next() {
        switch flag {
        case "--model", "-m": chatModel = chatFlags.next()
        case "--prompt", "-p": chatPrompt = chatFlags.next()
        case "--system": chatSystem = chatFlags.next() ?? chatSystem
        case "--max", "-n": chatCount = chatFlags.next().flatMap(Int.init) ?? 1024
        case "--slots": chatSlots = chatFlags.next().flatMap(Int.init) ?? 8
        case "--temperature", "-t":
            chatSettings.temperature = chatFlags.next().flatMap(Float.init) ?? 0.7
        case "--top-k": chatSettings.topK = chatFlags.next().flatMap(Int.init) ?? 40
        case "--top-p": chatSettings.topP = chatFlags.next().flatMap(Float.init) ?? 0.95
        case "--repetition-penalty":
            chatSettings.repetitionPenalty = chatFlags.next().flatMap(Float.init) ?? 1.0
        case "--seed": chatSettings.seed = chatFlags.next().flatMap(UInt64.init) ?? 0
        case "--greedy": chatSettings = .greedy
        case "--lookahead": setenv("GODWIT_LOOKAHEAD", "1", 1)
        case "--show-analysis": showAnalysis = true
        default:
            FileHandle.standardError.write(Data("unknown flag: \(flag)\n".utf8))
            exit(2)
        }
    }
    guard let chatModel else {
        FileHandle.standardError.write(Data(
            "usage: godwit chat --model <dir> [--prompt TEXT] [--temperature N] [--greedy]\n".utf8))
        exit(2)
    }
    runChat(model: chatModel, prompt: chatPrompt, system: chatSystem, count: chatCount,
            slots: chatSlots, settings: chatSettings, showAnalysis: showAnalysis)
case "tokenize":
    let tokArgs = Array(arguments.dropFirst())
    guard tokArgs.count >= 2 else {
        FileHandle.standardError.write(Data("usage: godwit tokenize <gwt-dir> <text>\n".utf8))
        exit(2)
    }
    runTokenize(model: tokArgs[0], text: tokArgs[1...].joined(separator: " "))
case "logits":
    var logitsModel: String?
    var logitsTokens: [Int] = []
    var logitsTopK = 10
    var logitsFlags = arguments.dropFirst().makeIterator()
    while let flag = logitsFlags.next() {
        switch flag {
        case "--model", "-m": logitsModel = logitsFlags.next()
        case "--tokens": logitsTokens = (logitsFlags.next() ?? "")
            .split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        case "--top": logitsTopK = logitsFlags.next().flatMap(Int.init) ?? 10
        default: break
        }
    }
    guard let logitsModel, !logitsTokens.isEmpty else {
        FileHandle.standardError.write(Data(
            "usage: godwit logits --model <dir> --tokens 1,2,3 [--top N]\n".utf8))
        exit(2)
    }
    runLogits(model: logitsModel, tokens: logitsTokens, topK: logitsTopK)
case "check-expert":
    let rest = Array(arguments.dropFirst())
    guard rest.count == 2 else {
        FileHandle.standardError.write(Data(
            "usage: godwit check-expert <gwt-dir> <reference-dir>\n".utf8))
        exit(2)
    }
    runExpertCheck(model: rest[0], reference: rest[1])
case "verify-expert":
    guard let directory = arguments.dropFirst().first else {
        FileHandle.standardError.write(Data("usage: godwit verify-expert <fixture-dir>\n".utf8))
        exit(2)
    }
    runExpertVerification(directory: directory)
case let other?:
    FileHandle.standardError.write(Data("unknown command: \(other)\n".utf8))
    exit(1)
case nil:
    print("""
    Godwit — streaming mixture-of-experts inference for memory-constrained machines

    usage: godwit <command>

    commands:
      chat             talk to the model (interactive, or --prompt for one shot)
      serve            web dashboard with live expert routing on 127.0.0.1
      range            probe the router to measure what each expert specialises in
      version          print the version
      bench dequant    compare MXFP4 against affine int4 fused GEMV throughput
      verify-expert    check the Metal kernel against real GPT-OSS weights
      install          stream and repack the checkpoint into a .gwt directory
      check-expert     run one installed expert and compare against a reference
      check-attention  run one layer's attention and compare against a reference
      check-layer      run a complete transformer layer against a reference
      logits           run the full model and show next-token candidates
      generate         greedily generate tokens from a prompt
      tokenize         encode and decode text with the model's tokeniser
      check-mxfp4      re-encode installed MXFP4 and demand byte equality
      dump-routing     write every routing decision, with weights, as JSON
      dump-hidden      write the residual entering each layer, for analysis
      ab-kernel        A/B two kernels with thermal drift controlled
      trace-routing    measure router skew from embeddings (superseded)
      trace-layers     run real layers and test whether routing is predictable
    """)
}

/// Checks the MXFP4 encoder against OpenAI's own bytes.
///
/// GPT-OSS ships MXFP4, so its installed expert weights are already on the
/// grid. Decoding them and encoding them back must return the identical bytes —
/// if it does not, the encoder disagrees with the format, and every Qwen3
/// weight it writes would be quietly wrong in the same way.
///
/// This is the check that caught the exponent being floored rather than ceiled.
func runMXFP4Check(model: String) {
    let root = URL(fileURLWithPath: model)
    guard let manifestData = try? Data(contentsOf: root.appendingPathComponent("manifest.json")),
          let manifest = try? JSONDecoder().decode(GodwitManifest.self, from: manifestData)
    else {
        FileHandle.standardError.write(Data("cannot read \(model)/manifest.json\n".utf8))
        exit(1)
    }
    guard let blocksSection = manifest.expertSections["gate_up_blocks"],
          let scalesSection = manifest.expertSections["gate_up_scales"] else {
        FileHandle.standardError.write(Data("no MXFP4 expert sections in this install\n".utf8))
        exit(1)
    }

    let layerFile = root.appendingPathComponent("experts")
        .appendingPathComponent(String(format: "layer_%02d.bin", 0))
    guard let handle = try? FileHandle(forReadingFrom: layerFile) else {
        FileHandle.standardError.write(Data("cannot open \(layerFile.path)\n".utf8))
        exit(1)
    }
    defer { try? handle.close() }

    var checkedBlocks = 0
    var byteMismatches = 0
    var scaleMismatches = 0
    let expertsToCheck = min(4, manifest.expertCount)

    for expert in 0..<expertsToCheck {
        let base = expert * manifest.expertStride
        try? handle.seek(toOffset: UInt64(base + blocksSection.offset))
        let packed = handle.readData(ofLength: blocksSection.length)
        try? handle.seek(toOffset: UInt64(base + scalesSection.offset))
        let scales = handle.readData(ofLength: scalesSection.length)
        guard packed.count == blocksSection.length,
              scales.count == scalesSection.length else { continue }

        let blockCount = scales.count
        let decoded = packed.withUnsafeBytes { p in
            scales.withUnsafeBytes { s in
                MXFP4.decode(packed: p, scales: s, blockCount: blockCount)
            }
        }
        let (repacked, rescaled) = MXFP4.encode(decoded)

        checkedBlocks += blockCount
        for index in 0..<min(repacked.count, packed.count)
        where repacked[index] != packed[packed.startIndex + index] {
            byteMismatches += 1
        }
        for index in 0..<min(rescaled.count, scales.count)
        where rescaled[index] != scales[scales.startIndex + index] {
            scaleMismatches += 1
        }
    }

    print("MXFP4 round trip against \(manifest.model)")
    print("  experts checked   \(expertsToCheck), layer 0, gate_up")
    print("  blocks            \(checkedBlocks)")
    print("  packed mismatches \(byteMismatches)")
    print("  scale mismatches  \(scaleMismatches)")
    if byteMismatches == 0 && scaleMismatches == 0 {
        print("  encoder agrees with the shipped format exactly")
    } else {
        print("  ENCODER DISAGREES WITH THE FORMAT")
        exit(1)
    }
}

/// Dumps every routing decision, with the router's weights, as JSON lines.
///
/// The weights are the point. `trace-routing` reports which experts fire, which
/// answers questions about caching; this answers a different one — how much
/// each chosen expert actually contributes, and therefore how much it would
/// cost to serve some of them at lower precision.
func runRoutingDump(model: String, prompt: String, tokens: Int) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let tokenizer = try reader.loadTokenizer()
        let runner = ModelRunner(context: context, reader: reader)
        let weights = try runner.loadWeights()
        let expertCache = try runner.makeExpertCache(slots: 8)

        var conversation = Conversation(system: "You are a helpful assistant.")
        conversation.append(Conversation.Message(.user, prompt))
        let ids = try conversation.encode(with: tokenizer)
        let cache = try runner.makeCache(maxContext: ids.count + tokens + 16)
        var sampler = Sampler(settings: .greedy)

        var step = 0
        func emit(_ layer: Int, _ decisions: [Router.Decision]) {
            guard let last = decisions.last else { return }
            let experts = last.experts.map(String.init).joined(separator: ",")
            let ws = last.weights.map { String(format: "%.6f", $0) }
                .joined(separator: ",")
            print("{\"step\":\(step),\"layer\":\(layer),"
                  + "\"experts\":[\(experts)],\"weights\":[\(ws)]}")
        }

        var logits = try runner.logits(tokens: ids, positionBase: 0, cache: cache,
                                       weights: weights, expertCache: expertCache,
                                       routing: emit)
        var produced: [Int] = []
        for _ in 0..<tokens {
            let next = sampler.pick(from: logits, history: produced)
            produced.append(next)
            step += 1
            logits = try runner.logits(tokens: [next], positionBase: cache.length,
                                       cache: cache, weights: weights,
                                       expertCache: expertCache, routing: emit)
        }
        FileHandle.standardError.write(Data(
            "dumped \(step + 1) steps\n".utf8))
    } catch {
        FileHandle.standardError.write(Data("routing dump failed: \(error)\n".utf8))
        exit(1)
    }
}

/// Dumps the residual stream entering every layer, for the last token of each
/// decode step, as raw float32.
///
/// Exists to test lookahead prefetching without building it: this vector is
/// what a lookahead scheme would have available one layer early, so applying
/// layer n's router to it offline measures the prediction directly.
func runHiddenDump(model: String, prompt: String, output: String, tokens: Int) {
    do {
        let context = try MetalContext()
        let reader = try ModelReader(directory: URL(fileURLWithPath: model))
        let tokenizer = try reader.loadTokenizer()
        let runner = ModelRunner(context: context, reader: reader)
        let weights = try runner.loadWeights()
        let expertCache = try runner.makeExpertCache(slots: 8)
        let width = reader.manifest.spec.hiddenSize

        var conversation = Conversation(system: "You are a helpful assistant.")
        conversation.append(Conversation.Message(.user, prompt))
        let ids = try conversation.encode(with: tokenizer)
        let cache = try runner.makeCache(maxContext: ids.count + tokens + 16)
        var sampler = Sampler(settings: .greedy)

        var out = Data()
        var tokenCount = ids.count
        func capture(_ layer: Int, _ stream: [Float]) {
            // Last row only: that is the token being generated.
            let start = (tokenCount - 1) * width
            var row = Array(stream[start..<(start + width)])
            out.append(Data(bytes: &row, count: width * 4))
        }

        var routerIn = Data()
        func captureRouter(_ layer: Int, _ v: [Float]) {
            let start = (tokenCount - 1) * width
            var row = Array(v[start..<(start + width)])
            routerIn.append(Data(bytes: &row, count: width * 4))
        }

        var logits = try runner.logits(tokens: ids, positionBase: 0, cache: cache,
                                       weights: weights, expertCache: expertCache,
                                       hidden: capture, routerInput: captureRouter)
        var produced: [Int] = []
        for _ in 0..<tokens {
            let next = sampler.pick(from: logits, history: produced)
            produced.append(next)
            tokenCount = 1
            logits = try runner.logits(tokens: [next], positionBase: cache.length,
                                       cache: cache, weights: weights,
                                       expertCache: expertCache, hidden: capture,
                                       routerInput: captureRouter)
        }
        try out.write(to: URL(fileURLWithPath: output))
        try routerIn.write(to: URL(fileURLWithPath: output + ".router"))
        FileHandle.standardError.write(Data(
            "wrote \(out.count / (width * 4)) vectors of \(width)\n".utf8))
    } catch {
        FileHandle.standardError.write(Data("hidden dump failed: \(error)\n".utf8))
        exit(1)
    }
}
