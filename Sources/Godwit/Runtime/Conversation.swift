import Foundation

/// Builds prompts in GPT-OSS's harmony format.
///
/// The format is load-bearing, not decorative. Fed bare prose the model gives
/// one good continuation and then drifts into conversational fragments, because
/// raw text is out of distribution for it. Wrapped properly it emits a
/// structured reply: an `analysis` channel where it reasons, then a `final`
/// channel with the answer.
public struct Conversation {
    public struct Message: Sendable {
        public enum Role: String, Sendable {
            case system, developer, user, assistant
        }
        public let role: Role
        public let content: String

        public init(_ role: Role, _ content: String) {
            self.role = role
            self.content = content
        }
    }

    public private(set) var messages: [Message]

    public init(system: String? = nil) {
        messages = []
        if let system { messages.append(Message(.system, system)) }
    }

    public mutating func append(_ message: Message) { messages.append(message) }

    public mutating func reset(keepingSystem: Bool = true) {
        messages = keepingSystem ? messages.filter { $0.role == .system } : []
    }

    /// Which turn markers a family uses.
    public enum Format: String, Sendable, Codable {
        /// GPT-OSS: `<|start|>role<|message|>content<|end|>`, with analysis and
        /// final channels inside the assistant turn.
        case harmony
        /// Qwen3 and most others: `<|im_start|>role\ncontent<|im_end|>`.
        case chatML

        /// The format a checkpoint uses, inferred from what its tokeniser has.
        ///
        /// Read from the vocabulary rather than declared in the spec: the
        /// tokeniser travels with the weights, so it cannot disagree with them,
        /// whereas a flag can be set wrong.
        public static func detect(_ tokenizer: Tokenizer) -> Format {
            tokenizer.id(of: "<|start|>") != nil ? .harmony : .chatML
        }
    }

    /// Encodes the conversation and opens an assistant turn for the model.
    public func encode(with tokenizer: Tokenizer) throws -> [Int] {
        switch Format.detect(tokenizer) {
        case .harmony: return try encodeHarmony(with: tokenizer)
        case .chatML: return try encodeChatML(with: tokenizer)
        }
    }

    /// `<|im_start|>role\ncontent<|im_end|>\n`, ending with an open assistant
    /// turn for the model to complete.
    private func encodeChatML(with tokenizer: Tokenizer) throws -> [Int] {
        guard let imStart = tokenizer.id(of: "<|im_start|>"),
              let imEnd = tokenizer.id(of: "<|im_end|>") else {
            throw TokenizerError.malformed("missing ChatML tokens")
        }
        var ids: [Int] = []
        for entry in messages {
            ids.append(imStart)
            ids.append(contentsOf: tokenizer.encode(entry.role.rawValue + "\n"))
            ids.append(contentsOf: tokenizer.encode(entry.content))
            ids.append(imEnd)
            ids.append(contentsOf: tokenizer.encode("\n"))
        }
        ids.append(imStart)
        ids.append(contentsOf: tokenizer.encode("assistant\n"))
        return ids
    }

    private func encodeHarmony(with tokenizer: Tokenizer) throws -> [Int] {
        func special(_ name: String) throws -> Int {
            guard let id = tokenizer.id(of: name) else {
                throw TokenizerError.malformed("missing special token \(name)")
            }
            return id
        }
        let start = try special("<|start|>")
        let message = try special("<|message|>")
        let end = try special("<|end|>")

        var ids: [Int] = []
        for entry in messages {
            ids.append(start)
            ids.append(contentsOf: tokenizer.encode(entry.role.rawValue))
            ids.append(message)
            ids.append(contentsOf: tokenizer.encode(entry.content))
            ids.append(end)
        }
        // The trailing open turn is what the model completes.
        ids.append(start)
        ids.append(contentsOf: tokenizer.encode("assistant"))
        return ids
    }

    /// Tokens that end an assistant turn.
    public static func stopTokens(_ tokenizer: Tokenizer) -> Set<Int> {
        Set(["<|return|>", "<|endoftext|>", "<|im_end|>"]
            .compactMap { tokenizer.id(of: $0) })
    }

    /// Extracts the `final` channel from a harmony reply, discarding the
    /// model's reasoning.
    ///
    /// The analysis channel is genuinely useful for debugging and genuinely
    /// noise for a user reading an answer, so it is separated rather than
    /// stripped.
    public static func split(_ reply: String) -> (analysis: String?, final: String) {
        // Qwen3 marks reasoning with <think> rather than harmony channels, and
        // emits plain text otherwise.
        if reply.contains("<think>") || reply.contains("</think>") {
            if let close = reply.range(of: "</think>") {
                let thought = reply[..<close.lowerBound]
                    .replacingOccurrences(of: "<think>", with: "")
                return (thought.trimmingCharacters(in: .whitespacesAndNewlines),
                        String(reply[close.upperBound...])
                            .trimmingCharacters(in: .whitespacesAndNewlines))
            }
            return (reply.replacingOccurrences(of: "<think>", with: ""), "")
        }

        func channel(_ name: String) -> String? {
            guard let head = reply.range(of: "<|channel|>\(name)<|message|>") else { return nil }
            let rest = reply[head.upperBound...]
            for terminator in ["<|end|>", "<|return|>", "<|start|>"] {
                if let stop = rest.range(of: terminator) {
                    return String(rest[..<stop.lowerBound])
                }
            }
            return String(rest)
        }
        let analysis = channel("analysis")
        if let final = channel("final") { return (analysis, final) }
        // A reply that has started emitting channel markers has no answer text
        // yet, even if the marker is only half-written. Returning the partial
        // marker as content is how "<|channel|>analysis" ends up on screen.
        if reply.contains("<|channel|>") || "<|channel|>".hasPrefix(reply) {
            return (analysis, "")
        }
        return (analysis, reply)
    }
}
