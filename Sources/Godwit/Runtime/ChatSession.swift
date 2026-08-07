import Foundation

/// A conversation and the caches it has warmed.
///
/// Chat re-encodes the whole conversation every turn, so prefilling from
/// scratch made time to first token grow with the conversation without bound.
/// A session keeps its KV cache between turns and prefills only the tokens the
/// cache does not already hold, which flattens that curve.
///
/// Both the CLI and the dashboard drive this. They used to share nothing, and
/// the dashboard had no conversation at all — every question started a new one.
public final class ChatSession {
    /// What a turn needs to run: the cache to run against, the tokens still to
    /// be prefilled, and where they start.
    public struct Turn {
        /// The cache, rewound to the end of the reusable prefix.
        public let cache: KVCache
        /// The suffix to prefill — everything the cache does not already hold.
        public let tokens: [Int]
        /// Absolute position of the first token in `tokens`, and equally the
        /// length of the prefix that was reused.
        public let positionBase: Int
        /// The full prompt length, for reporting what the reuse saved.
        public let promptCount: Int

        public var reused: Int { positionBase }
    }

    public private(set) var conversation: Conversation

    private let runner: ModelRunner
    private let tokenizer: Tokenizer
    /// Generation headroom to allocate beyond the prompt.
    private let reserve: Int

    private var cache: KVCache?
    /// The tokens the cache was actually built from — prompt plus everything
    /// sampled. Reuse is measured against this rather than against the
    /// conversation, and the difference is not cosmetic: an assistant turn is
    /// re-encoded from its answer channel alone and wrapped in template markup
    /// that was never generated, so the two diverge partway through.
    private var cached: [Int] = []

    public init(runner: ModelRunner, tokenizer: Tokenizer, system: String,
                reserve: Int) {
        self.runner = runner
        self.tokenizer = tokenizer
        self.reserve = reserve
        self.conversation = Conversation(system: system)
    }

    /// How much of `ids` the cache already holds.
    ///
    /// Capped one short of the prompt because logits come from the last
    /// position: a prefill of nothing produces none, so there must always be a
    /// token left to run even when the prompt is entirely cached.
    public static func reusablePrefix(cached: [Int], ids: [Int]) -> Int {
        var reused = 0
        while reused < min(cached.count, ids.count - 1), cached[reused] == ids[reused] {
            reused += 1
        }
        return reused
    }

    /// Adds the question and returns what remains to be prefilled.
    public func begin(question: String) throws -> Turn {
        conversation.append(Conversation.Message(.user, question))
        let ids = try conversation.encode(with: tokenizer)

        // Grown in steps rather than to fit, so a conversation is not
        // reallocated — and its prefix lost — on every turn.
        let needed = ids.count + reserve + 16
        if cache == nil || cache!.maxContext < needed {
            cache = try runner.makeCache(
                maxContext: max(needed, 2 * (cache?.maxContext ?? 0), 2048))
            cached = []
        }
        let cache = cache!

        // An escape hatch, because reuse is the kind of optimisation that fails
        // silently: a stale prefix still produces fluent text. This is how the
        // two paths were shown to agree token for token.
        if ProcessInfo.processInfo.environment["GODWIT_NO_PREFIX_REUSE"] != nil {
            cached = []
        }

        let reused = Self.reusablePrefix(cached: cached, ids: ids)
        cache.truncate(to: reused)
        cached = ids
        return Turn(cache: cache, tokens: Array(ids[reused...]),
                    positionBase: reused, promptCount: ids.count)
    }

    /// Records a sampled token, which the decode step wrote into the cache.
    public func record(_ token: Int) {
        cached.append(token)
    }

    /// Closes the turn with the answer the user saw.
    public func finish(reply: String) {
        conversation.append(Conversation.Message(.assistant, reply))
    }

    /// Clears the history. The allocated cache is kept and rewound — the
    /// buffers are the expensive part and nothing reads past `length`.
    public func reset() {
        conversation.reset()
        cache?.reset()
        cached = []
    }
}
