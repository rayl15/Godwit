import Foundation
import Testing

@testable import Godwit

@Suite("Sampler")
struct SamplerTests {
    @Test("Greedy is deterministic and picks the maximum")
    func greedyPicksMax() {
        var sampler = Sampler(settings: .greedy)
        let logits: [Float] = [0.1, 5.0, 2.0, -3.0, 4.9]
        for _ in 0..<5 { #expect(sampler.pick(from: logits) == 1) }
    }

    @Test("A seed makes sampling reproducible")
    func seededIsReproducible() {
        let settings = Sampler.Settings(temperature: 1.0, topK: 0, topP: 1, seed: 42)
        let logits = (0..<50).map { Float($0) / 10 }
        var first = Sampler(settings: settings)
        var second = Sampler(settings: settings)
        let a = (0..<20).map { _ in first.pick(from: logits) }
        let b = (0..<20).map { _ in second.pick(from: logits) }
        #expect(a == b)
    }

    @Test("top-k restricts the candidate set")
    func topKRestricts() {
        var sampler = Sampler(settings: .init(temperature: 2.0, topK: 2, topP: 1, seed: 7))
        let logits: [Float] = [10, 9, 0, 0, 0, 0, 0, 0]
        let picks = Set((0..<200).map { _ in sampler.pick(from: logits) })
        #expect(picks.isSubset(of: [0, 1]), "got \(picks)")
    }

    @Test("Repetition penalty pushes both signs downward")
    func repetitionPenaltyDirection() {
        // Dividing a negative logit would raise it, which is the opposite of a
        // penalty — so the sign has to be handled explicitly.
        var sampler = Sampler(settings: .init(temperature: 0, topK: 0, topP: 1,
                                              repetitionPenalty: 2.0))
        // Token 0 is positive and ahead; penalising it should hand over to 1.
        #expect(sampler.pick(from: [4.0, 3.0], history: [0]) == 1)
        // Token 0 is negative and ahead; penalising must not promote it.
        // Values chosen so the result is not a tie: -1.0 * 2 = -2.0 against
        // -1.5. A tie would go to the lower index and prove nothing.
        #expect(sampler.pick(from: [-1.0, -1.5], history: [0]) == 1)
    }
}

@Suite("Conversation")
struct ConversationTests {
    @Test("Partial channel markers are not shown as content")
    func partialMarkerHidden() {
        // Mid-stream the text is a half-written marker. Treating it as the
        // answer printed "<|channel|>analysis" to the user.
        #expect(Conversation.split("<|channel|>").final.isEmpty)
        #expect(Conversation.split("<|channel|>analysis").final.isEmpty)
        #expect(Conversation.split("<|channel|>analysis<|message|>thinking").final.isEmpty)
    }

    @Test("The final channel is extracted, reasoning separated")
    func splitsChannels() {
        let reply = "<|channel|>analysis<|message|>Let me think.<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>"
        let parts = Conversation.split(reply)
        #expect(parts.analysis == "Let me think.")
        #expect(parts.final == "The answer is 42.")
    }

    @Test("Plain text with no markers passes through")
    func plainTextPassesThrough() {
        #expect(Conversation.split("just words").final == "just words")
    }

    @Test("Streaming the final channel only ever grows")
    func finalChannelGrowsMonotonically() {
        // What the display relies on: once answer text starts, each longer
        // reply must extend the previous visible string rather than replace it.
        let full = "<|channel|>analysis<|message|>hmm<|end|><|start|>assistant"
            + "<|channel|>final<|message|>Hello there"
        var previous = ""
        for length in stride(from: 1, through: full.count, by: 1) {
            let visible = Conversation.split(String(full.prefix(length))).final
            if !visible.isEmpty && !previous.isEmpty {
                #expect(visible.hasPrefix(previous) || previous.hasPrefix(visible),
                        "\(previous.debugDescription) -> \(visible.debugDescription)")
            }
            previous = visible
        }
        #expect(previous == "Hello there")
    }
}
