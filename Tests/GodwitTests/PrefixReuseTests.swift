import Metal
import Testing

@testable import Godwit

/// Prefix reuse across conversational turns.
///
/// Chat re-encodes the whole conversation every turn, so turn N used to prefill
/// N turns' worth of tokens from scratch. Keeping the KV cache and prefilling
/// only the new suffix makes time-to-first-token flat instead of growing.
///
/// The risk is that reuse fails silently — a stale prefix still produces
/// fluent text — so the properties it depends on are pinned here. These call
/// `ChatSession` directly, which is the same code the CLI and the dashboard
/// both run.
@Suite("Prefix reuse", .enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct PrefixReuseTests {
    private func cache(_ maxContext: Int = 64) throws -> KVCache {
        try KVCache(context: try MetalContext(), spec: .qwen3MoE30B,
                    maxContext: maxContext, layerCount: 2)
    }

    private func reusable(cached: [Int], ids: [Int]) -> Int {
        ChatSession.reusablePrefix(cached: cached, ids: ids)
    }

    @Test("Truncation rewinds to a prefix and accepts the boundaries")
    func truncateRewinds() throws {
        let cache = try cache()
        cache.advance(by: 20)
        cache.truncate(to: 8)
        #expect(cache.length == 8)
        cache.advance(by: 5)
        #expect(cache.length == 13)
        cache.truncate(to: 13)          // no-op at the head
        #expect(cache.length == 13)
        cache.truncate(to: 0)           // equivalent to reset
        #expect(cache.length == 0)
    }

    @Test("A continued conversation reuses its shared prefix")
    func continuedTurnReusesPrefix() {
        // Turn one's prompt, then what was sampled, then turn two's markup and
        // question — the real shape, where the cache holds prompt + generation.
        let turnOne = [1, 2, 3, 4]
        let cached = turnOne + [90, 91]                 // + sampled tokens
        let turnTwo = turnOne + [90, 91, 5, 6, 7]
        #expect(reusable(cached: cached, ids: turnTwo) == 6)
    }

    /// The case that makes naive reuse wrong. An assistant turn is re-encoded
    /// from its answer channel alone, so the tokens in the cache and the tokens
    /// in the new prompt diverge partway through — and everything after the
    /// divergence must be recomputed.
    @Test("Divergence in the middle stops the reuse there")
    func divergenceStopsReuse() {
        let cached = [1, 2, 3, 40, 41, 42]
        let ids = [1, 2, 3, 50, 51, 52, 53]
        #expect(reusable(cached: cached, ids: ids) == 3)
    }

    @Test("An unrelated conversation reuses nothing")
    func freshConversationReusesNothing() {
        #expect(reusable(cached: [9, 8, 7], ids: [1, 2, 3]) == 0)
        #expect(reusable(cached: [], ids: [1, 2, 3]) == 0)
    }

    /// Reuse must never consume the whole prompt, even when the new prompt is
    /// a prefix of what is already cached — there would be no position left to
    /// read logits from.
    @Test("At least one token is always left to run")
    func alwaysLeavesATokenToRun() {
        let ids = [1, 2, 3, 4]
        #expect(reusable(cached: ids, ids: ids) == 3)
        #expect(reusable(cached: ids + [5, 6], ids: ids) == 3)
    }
}
