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
    """)
}
