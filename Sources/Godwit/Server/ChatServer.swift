import Foundation

/// Serves the dashboard and streams generation to it.
///
/// One model is loaded at a time. There is a single set of expert slots and a
/// single Metal queue, so two resident models would compete for both — and on
/// a 16 GB machine two expert caches would not fit anyway. Switching releases
/// the old one before loading the new.
public final class ChatServer {
    /// Everything tied to one loaded model.
    private final class Loaded {
        let directory: URL
        let reader: ModelReader
        let runner: ModelRunner
        let tokenizer: Tokenizer
        let weights: ModelRunner.Weights
        let expertCache: ExpertCache
        let session: ChatSession

        init(directory: URL, context: MetalContext, slots: Int,
             system: String, maxTokens: Int) throws {
            self.directory = directory
            self.reader = try ModelReader(directory: directory)
            self.tokenizer = try reader.loadTokenizer()
            self.runner = ModelRunner(context: context, reader: reader)
            self.weights = try runner.loadWeights()
            self.expertCache = try runner.makeExpertCache(slots: slots)
            self.session = ChatSession(runner: runner, tokenizer: tokenizer,
                                       system: system, reserve: maxTokens)
        }
    }

    private let context: MetalContext
    private let registry: ModelRegistry
    private let settings: Sampler.Settings
    private let maxTokens: Int
    private let system: String
    private let slots: Int
    private let turnLock = NSLock()
    private var loaded: Loaded?
    /// Set while an install is running, so a second one is refused.
    private var installing: String?
    /// Set while a range probe is running. It holds the same lock generation
    /// does, so a second one would deadlock rather than queue.
    private var probing = false

    public init(root: URL, initial: URL? = nil, slots: Int = 8,
                settings: Sampler.Settings = Sampler.Settings(),
                maxTokens: Int = 1024,
                system: String = "You are a helpful assistant.") throws {
        self.context = try MetalContext()
        self.registry = ModelRegistry(root: root)
        self.settings = settings
        self.maxTokens = maxTokens
        self.system = system
        self.slots = slots
        if let initial {
            self.loaded = try Loaded(directory: initial, context: context,
                                     slots: slots, system: system,
                                     maxTokens: maxTokens)
        }
    }

    /// Loads a model, releasing the previous one first.
    ///
    /// The order matters on a machine this size: holding both while the new one
    /// allocates would need the sum of two expert caches.
    private func load(directory: URL) throws {
        turnLock.lock()
        defer { turnLock.unlock() }
        loaded = nil
        loaded = try Loaded(directory: directory, context: context, slots: slots,
                            system: system, maxTokens: maxTokens)
    }

    public func listen(port: UInt16) throws {
        let server = HTTPServer(port: port) { [weak self] request in
            guard let self else { return .notFound }
            switch request.path {
            case "/":
                return .ok(contentType: "text/html; charset=utf-8",
                           body: Data(self.page().utf8))
            case "/api/models":
                return .json(self.modelsJSON())
            case "/api/select":
                guard let name = request.query["name"] else {
                    return .json("{\"error\":\"missing name\"}")
                }
                do {
                    try self.load(directory: self.registry.root.appendingPathComponent(name))
                    return .json("{\"ok\":true}")
                } catch {
                    return .json("{\"error\":\(Self.quoted("\(error)"))}")
                }
            case "/api/install":
                guard let id = request.query["id"] else {
                    return .json("{\"error\":\"missing id\"}")
                }
                return .stream { stream in self.install(id: id, to: stream) }
            case "/api/range/build":
                return .stream { stream in self.buildRange(to: stream) }
            case "/api/range":
                guard let loaded = self.loaded,
                      let data = try? Data(contentsOf:
                        loaded.directory.appendingPathComponent("range.json"))
                else { return .json("{\"points\":[]}") }
                return .ok(contentType: "application/json", body: data)
            case "/api/chat/reset":
                self.turnLock.lock()
                self.loaded?.session.reset()
                self.turnLock.unlock()
                return .json("{\"ok\":true}")
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
        print("models in \(registry.root.path)")
        for model in registry.installed() {
            print("  \(model.name)  \(model.layers)L x \(model.experts)E  "
                  + String(format: "%.1f GiB", Double(model.bytes) / 1_073_741_824))
        }
        server.run()
    }

    private func page() -> String {
        let spec = loaded?.reader.manifest.spec
        return WebUI.page(
            model: loaded?.reader.manifest.model ?? "none loaded",
            layers: loaded?.reader.manifest.layerCount ?? 36,
            experts: loaded?.reader.manifest.expertCount ?? 128,
            topK: spec?.maxExpertsPerToken ?? 4,
            slots: slots,
            cacheGiB: Double(loaded?.expertCache.byteCount ?? 0) / 1_073_741_824,
            trunkGiB: Double(loaded?.reader.manifest.trunkSections
                .reduce(0) { $0 + $1.length } ?? 0) / 1_073_741_824)
    }

    private func modelsJSON() -> String {
        let encoder = JSONEncoder()
        let installed = (try? encoder.encode(registry.installed()))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        let available = (try? encoder.encode(ModelRegistry.catalogue))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        let active = loaded.map { Self.quoted($0.directory.lastPathComponent) } ?? "null"
        let busy = installing.map { Self.quoted($0) } ?? "null"
        return "{\"installed\":\(installed),\"available\":\(available),"
            + "\"active\":\(active),\"installing\":\(busy)}"
    }

    /// Streams an install, then loads what it produced.
    private func install(id: String, to stream: HTTPServer.EventStream) {
        guard ModelRegistry.spec(for: id) != nil else {
            stream.send(event: "error", json: "{\"message\":\"unsupported model\"}")
            return
        }
        turnLock.lock()
        if installing != nil {
            turnLock.unlock()
            stream.send(event: "error", json: "{\"message\":\"an install is already running\"}")
            return
        }
        installing = id
        turnLock.unlock()
        defer { turnLock.lock(); installing = nil; turnLock.unlock() }

        let name = id.split(separator: "/").last.map(String.init) ?? "model"
        let target = registry.root.appendingPathComponent("\(name).gwt")

        // An install takes hours. Run it to completion even if the browser tab
        // that started it goes away — the bytes are worth more than the stream.
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var failure: Error?
        Task {
            do {
                let installer = Installer(spec: ModelRegistry.spec(for: id)!,
                                          options: .init(repository: id))
                _ = try await installer.install(to: target) { progress in
                    stream.send(event: "progress", json: """
                        {"stage":\(Self.quoted(progress.stage)),\
                        "done":\(progress.completed),"total":\(progress.total),\
                        "bytes":\(progress.bytesWritten)}
                        """)
                }
            } catch { failure = error }
            semaphore.signal()
        }
        semaphore.wait()

        if let failure {
            stream.send(event: "error", json: "{\"message\":\(Self.quoted("\(failure)"))}")
            return
        }
        stream.send(event: "done", json: "{\"name\":\(Self.quoted("\(name).gwt"))}")
    }

    /// Probes the router and writes a range map for the loaded model.
    ///
    /// Runs on the request thread while holding the turn lock, because it uses
    /// the same runner and expert slots a chat turn does. That makes the
    /// dashboard unresponsive to generation for a few minutes, which is
    /// honest: there is one GPU queue and one set of slots, and pretending
    /// otherwise would mean two models' worth of experts in 16 GB.
    private func buildRange(to stream: HTTPServer.EventStream) {
        turnLock.lock()
        if probing {
            turnLock.unlock()
            stream.send(event: "error", json: "{\"message\":\"already probing\"}")
            return
        }
        guard let loaded else {
            turnLock.unlock()
            stream.send(event: "error", json: "{\"message\":\"no model loaded\"}")
            return
        }
        probing = true
        defer { probing = false; turnLock.unlock() }

        do {
            let probe = ExpertRange(context: context, reader: loaded.reader)
            let map = try probe.build(slots: slots) { topic, done, total in
                stream.send(event: "progress", json:
                    "{\"topic\":\(Self.quoted(topic)),\"done\":\(done),"
                    + "\"total\":\(total)}")
            }
            let data = try JSONEncoder().encode(map)
            try data.write(to: loaded.directory.appendingPathComponent("range.json"))
            stream.send(event: "done", json: "{\"experts\":\(map.points.count)}")
        } catch {
            stream.send(event: "error",
                        json: "{\"message\":\(Self.quoted("\(error)"))}")
        }
    }

    private static func quoted(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                out += scalar.value < 0x20 ? String(format: "\\u%04x", scalar.value)
                                           : String(scalar)
            }
        }
        return out + "\""
    }

    private func generate(question: String, to stream: HTTPServer.EventStream) {
        turnLock.lock()
        defer { turnLock.unlock() }

        guard let loaded else {
            stream.send(event: "token", json: "{\"text\":\"no model loaded\"}")
            stream.send(event: "done", json: "{}")
            return
        }
        let runner = loaded.runner, tokenizer = loaded.tokenizer
        let weights = loaded.weights, expertCache = loaded.expertCache

        do {
            let stops = Conversation.stopTokens(tokenizer)
            // The session holds the conversation and its KV cache, so a
            // follow-up prefills only its own new tokens instead of the whole
            // history. On the first turn this is identical to starting cold.
            let turn = try loaded.session.begin(question: question)
            let cache = turn.cache
            var sampler = Sampler(settings: settings)
            expertCache.resetStats()

            stream.send(event: "stats", json: "{\"stage\":\"prefill\"}")

            let started = Date()
            var logits = try runner.logits(
                tokens: turn.tokens, positionBase: turn.positionBase,
                cache: cache, weights: weights,
                expertCache: expertCache,
                routing: { layer, decisions in
                    // The last token's choices are the interesting ones — the
                    // earlier rows are prompt history the user has already seen.
                    guard let last = decisions.last else { return }
                    stream.send(event: "routing",
                                json: "{\"layer\":\(layer),\"experts\":\(last.experts)}")
                })
            let ttft = Date().timeIntervalSince(started)
            stream.send(event: "stats", json:
                "{\"stage\":\"decode\",\"ttft\":\(ttft),"
                + "\"prompt\":\(turn.promptCount),\"reused\":\(turn.reused)}")

            var produced: [Int] = []
            var lastAnalysis = -1
            let decodeStart = Date()

            var stopped = false
            for step in 0..<maxTokens {
                guard stream.isOpen else { stopped = true; break }
                let next = sampler.pick(from: logits, history: produced)
                if stops.contains(next) { stopped = true; break }
                produced.append(next)
                loaded.session.record(next)

                let full = tokenizer.decode(produced)
                let parts = Conversation.split(full)
                stream.send(event: "token", json: "{\"text\":\(quote(parts.final))}")
                // Reasoning is emitted as it arrives, not held until the end.
                // GPT-OSS routinely spends hundreds of tokens in the analysis
                // channel before the first word of the answer, and at these
                // speeds that is minutes of a blank reply — indistinguishable
                // from a hang. Only resend when it has actually grown.
                if let analysis = parts.analysis, analysis.count != lastAnalysis {
                    lastAnalysis = analysis.count
                    stream.send(event: "analysis", json: "{\"text\":\(quote(analysis))}")
                }

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

            // A turn can spend its whole budget in the analysis channel and
            // never reach an answer — GPT-OSS does this on open-ended questions.
            // Unless the client is told, the reply is just an empty box.
            let reply = Conversation.split(tokenizer.decode(produced)).final
            loaded.session.finish(reply: reply)
            let answered = !reply.isEmpty
            stream.send(event: "done", json:
                "{\"truncated\":\(!stopped),\"answered\":\(answered),"
                + "\"limit\":\(maxTokens)}")
        } catch {
            stream.send(event: "token", json: "{\"text\":\(quote("error: \(error)"))}")
            stream.send(event: "done", json: "{}")
        }
    }

    /// JSON string escaping. Server-sent events are newline-delimited, so an
    /// unescaped newline in a reply would truncate the frame.
    private func quote(_ text: String) -> String { Self.quoted(text) }

}
