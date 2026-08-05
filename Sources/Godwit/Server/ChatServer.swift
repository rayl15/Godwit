import Foundation

/// Serves the dashboard and streams generation to it.
///
/// One model, one turn at a time. There is a single set of expert slots and a
/// single Metal queue, so concurrent requests would interleave into each
/// other's cache; the second caller waits.
public final class ChatServer {
    private let context: MetalContext
    private let reader: ModelReader
    private let runner: ModelRunner
    private let tokenizer: Tokenizer
    private let weights: ModelRunner.Weights
    private let expertCache: ExpertCache
    private let settings: Sampler.Settings
    private let maxTokens: Int
    private let system: String
    private let turnLock = NSLock()

    private let directory: URL

    public init(directory: URL, slots: Int = 8, settings: Sampler.Settings = Sampler.Settings(),
                maxTokens: Int = 1024,
                system: String = "You are a helpful assistant.") throws {
        self.directory = directory
        self.context = try MetalContext()
        self.reader = try ModelReader(directory: directory)
        self.tokenizer = try reader.loadTokenizer()
        self.runner = ModelRunner(context: context, reader: reader)
        self.weights = try runner.loadWeights()
        self.expertCache = try runner.makeExpertCache(slots: slots)
        self.settings = settings
        self.maxTokens = maxTokens
        self.system = system
    }

    public func listen(port: UInt16) throws {
        let spec = reader.manifest.spec
        let trunkBytes = reader.manifest.trunkSections.reduce(0) { $0 + $1.length }
        let page = WebUI.page(
            model: reader.manifest.model,
            layers: reader.manifest.layerCount,
            experts: reader.manifest.expertCount,
            topK: spec.maxExpertsPerToken,
            slots: expertCache.slotCount,
            cacheGiB: Double(expertCache.byteCount) / 1_073_741_824,
            trunkGiB: Double(trunkBytes) / 1_073_741_824)

        let server = HTTPServer(port: port) { [weak self] request in
            guard let self else { return .notFound }
            switch request.path {
            case "/":
                return .ok(contentType: "text/html; charset=utf-8", body: Data(page.utf8))
            case "/api/range":
                // Optional: the dashboard shows a hint when it is absent.
                let path = directory.appendingPathComponent("range.json")
                guard let data = try? Data(contentsOf: path) else {
                    return .json("{\"points\":[]}")
                }
                return .ok(contentType: "application/json", body: data)
            case "/api/chat":
                guard let question = request.query["q"], !question.isEmpty else {
                    return .json("{\"error\":\"missing q\"}")
                }
                return .stream { stream in self.generate(question: question, to: stream) }
            default:
                return .notFound
            }
        }

        try server.start()
        print("Godwit dashboard on http://127.0.0.1:\(port)")
        print("model \(reader.manifest.model), \(expertCache.slotCount) expert slots")
        server.run()
    }

    private func generate(question: String, to stream: HTTPServer.EventStream) {
        turnLock.lock()
        defer { turnLock.unlock() }

        do {
            var conversation = Conversation(system: system)
            conversation.append(Conversation.Message(.user, question))
            let ids = try conversation.encode(with: tokenizer)
            let stops = Conversation.stopTokens(tokenizer)

            let cache = try runner.makeCache(maxContext: ids.count + maxTokens + 16)
            var sampler = Sampler(settings: settings)
            expertCache.resetStats()

            stream.send(event: "stats", json: "{\"stage\":\"prefill\"}")

            let started = Date()
            var logits = try runner.logits(
                tokens: ids, positionBase: 0, cache: cache, weights: weights,
                expertCache: expertCache,
                routing: { layer, decisions in
                    // The last token's choices are the interesting ones — the
                    // earlier rows are prompt history the user has already seen.
                    guard let last = decisions.last else { return }
                    stream.send(event: "routing",
                                json: "{\"layer\":\(layer),\"experts\":\(last.experts)}")
                })
            let ttft = Date().timeIntervalSince(started)
            stream.send(event: "stats",
                        json: "{\"stage\":\"decode\",\"ttft\":\(ttft)}")

            var produced: [Int] = []
            let decodeStart = Date()

            for step in 0..<maxTokens {
                guard stream.isOpen else { break }
                let next = sampler.pick(from: logits, history: produced)
                if stops.contains(next) { break }
                produced.append(next)

                let full = tokenizer.decode(produced)
                let visible = Conversation.split(full).final
                stream.send(event: "token", json: "{\"text\":\(quote(visible))}")

                let elapsed = Date().timeIntervalSince(decodeStart)
                let stats = expertCache.stats
                stream.send(event: "stats", json: """
                    {"stage":"decode","rate":\(Double(produced.count) / max(elapsed, 0.001)),\
                    "tokens":\(produced.count),"hit":\(stats.hitRate),\
                    "read":\(Double(stats.bytesRead) / 1_073_741_824),\
                    "misses":\(stats.misses)}
                    """)

                if step == maxTokens - 1 { break }
                logits = try runner.logits(
                    tokens: [next], positionBase: cache.length, cache: cache,
                    weights: weights, expertCache: expertCache,
                    routing: { layer, decisions in
                        guard let last = decisions.last else { return }
                        stream.send(event: "routing",
                                    json: "{\"layer\":\(layer),\"experts\":\(last.experts)}")
                    })
            }

            let parts = Conversation.split(tokenizer.decode(produced))
            if let analysis = parts.analysis, !analysis.isEmpty {
                stream.send(event: "analysis", json: "{\"text\":\(quote(analysis))}")
            }
            stream.send(event: "done", json: "{}")
        } catch {
            stream.send(event: "token", json: "{\"text\":\(quote("error: \(error)"))}")
            stream.send(event: "done", json: "{}")
        }
    }

    /// JSON string escaping. Server-sent events are newline-delimited, so an
    /// unescaped newline in a reply would truncate the frame.
    private func quote(_ text: String) -> String {
        var out = "\""
        for character in text.unicodeScalars {
            switch character {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if character.value < 0x20 {
                    out += String(format: "\\u%04x", character.value)
                } else {
                    out.unicodeScalars.append(character)
                }
            }
        }
        return out + "\""
    }
}
