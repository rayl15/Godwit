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

    /// Encodes the conversation and opens an assistant turn for the model.
    public func encode(with tokenizer: Tokenizer) throws -> [Int] {
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
        Set(["<|return|>", "<|endoftext|>"].compactMap { tokenizer.id(of: $0) })
    }

    /// Extracts the `final` channel from a harmony reply, discarding the
    /// model's reasoning.
    ///
    /// The analysis channel is genuinely useful for debugging and genuinely
    /// noise for a user reading an answer, so it is separated rather than
    /// stripped.
    public static func split(_ reply: String) -> (analysis: String?, final: String) {
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
