import Foundation
import Godwit

// GPT-OSS-120B decode budget, from docs/ESTIMATE.md. Every token multiplies
// this many quantised expert weights, and it has to fit inside the time the
// expert reads take, or compute becomes the bottleneck instead of I/O.
let expertParams = 2 * 2880 * 2880 + 2880 * 2880
let weightsPerToken = Double(4 * 36 * expertParams)
let ioBudgetSeconds = 0.225   // 8 cache slots, measured NVMe bandwidth
let requiredWeightsPerSecond = weightsPerToken / ioBudgetSeconds

func runDequantBenchmark() {
    do {
        let context = try MetalContext()
        let benchmark = DequantGEMVBenchmark(context: context)

        print("device: \(context.device.name)")
        print("""
              budget: \(String(format: "%.2f", weightsPerToken / 1e9))B expert weights/token \
              in \(Int(ioBudgetSeconds * 1000)) ms
              """)
        print("        => need \(String(format: "%.1f", requiredWeightsPerSecond / 1e9)) G weights/s\n")

        // GPT-OSS expert shapes: gate+up is 5760x2880, down is 2880x2880.
        let shapes = [(rows: 5760, cols: 2880), (rows: 2880, cols: 2880)]

        for shape in shapes {
            print("--- \(shape.rows) x \(shape.cols) ---")
            let results = [
                try benchmark.runMXFP4(rows: shape.rows, cols: shape.cols,
                                       iterations: 200, validate: true),
                try benchmark.runAffineInt4(rows: shape.rows, cols: shape.cols,
                                            iterations: 200),
            ]
            for result in results {
                let headroom = result.weightsPerSecond / requiredWeightsPerSecond
                let error = result.maxAbsoluteError.map { String(format: "%.2e", $0) } ?? "-"
                print(String(
                    format: "  %-32@  %6.1f G w/s  %6.1f GiB/s  %5.2fx budget  err %@",
                    result.label as NSString,
                    result.weightsPerSecond / 1e9,
                    result.gibPerSecond,
                    headroom,
                    error as NSString))
            }
            print()
        }
    } catch {
        FileHandle.standardError.write(Data("benchmark failed: \(error)\n".utf8))
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

func runInstall(directory: String, layerLimit: Int?) async {
    do {
        let options = Installer.Options(layerLimit: layerLimit)
        let installer = Installer(options: options)
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
        let rope = RoPE(configuration: .gptOSS)
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
        let produced = try attention.forward(hidden: hidden, tokenCount: tokens,
                                             positionBase: 0, weights: weights, layer: layer)

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
        let rope = RoPE(configuration: .gptOSS)

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
        let started = Date()
        let (produced, trace) = try layer.forward(
            hidden: hidden, tokenCount: tokens, positionBase: 0, weights: weights)
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
    runDequantBenchmark()
case "install":
    var target: String?
    var limit: Int?
    var rest = arguments.dropFirst().makeIterator()
    while let flag = rest.next() {
        switch flag {
        case "--output", "-o": target = rest.next()
        case "--layers": limit = rest.next().flatMap(Int.init)
        default:
            FileHandle.standardError.write(Data("unknown flag: \(flag)\n".utf8))
            exit(2)
        }
    }
    guard let target else {
        FileHandle.standardError.write(Data(
            "usage: godwit install --output <dir> [--layers N]\n".utf8))
        exit(2)
    }
    await runInstall(directory: target, layerLimit: limit)
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
    godwit — streaming mixture-of-experts inference for memory-constrained machines

    usage: godwit <command>

    commands:
      version          print the version
      bench dequant    compare MXFP4 against affine int4 fused GEMV throughput
      verify-expert    check the Metal kernel against real GPT-OSS weights
      install          stream and repack the checkpoint into a .gwt directory
      check-expert     run one installed expert and compare against a reference
      check-attention  run one layer's attention and compare against a reference
      check-layer      run a complete transformer layer against a reference
      trace-routing    measure router skew from embeddings (superseded)
      trace-layers     run real layers and test whether routing is predictable
    """)
}
