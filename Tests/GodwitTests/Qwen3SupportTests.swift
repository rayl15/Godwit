import Foundation
import Testing

@testable import Godwit

@Suite("Qwen3 support")
struct Qwen3SupportTests {
    /// Plain RoPE is reached by collapsing YaRN rather than by a second
    /// implementation, so the thing worth checking is that it really does
    /// collapse — a scaling factor of 1 must leave the base frequencies
    /// untouched.
    @Test("A scaling factor of one leaves RoPE unscaled")
    func plainRopeIsUnscaled() {
        let plain = RoPE(configuration: .plain(headDimension: 128,
                                               theta: 1_000_000,
                                               contextLength: 40_960))
        #expect(plain.attentionScale == 1)

        for (index, frequency) in plain.inverseFrequencies.enumerated() {
            let expected = 1 / powf(1_000_000, Float(index * 2) / 128)
            #expect(abs(frequency - expected) < expected * 1e-5,
                    "pair \(index): \(frequency) is not the unscaled \(expected)")
        }
    }

    @Test("GPT-OSS still gets YaRN, and it still changes something")
    func gptOSSKeepsYaRN() {
        let yarn = RoPE(configuration: .forSpec(.gptOSS120B))
        #expect(yarn.attentionScale > 1, "YaRN compensates for stretched positions")
        // Long-wavelength pairs are interpolated, so they must end up below
        // where plain RoPE would put them.
        let plain = RoPE(configuration: .plain(headDimension: 64, theta: 150_000,
                                               contextLength: 4096))
        let last = yarn.inverseFrequencies.count - 1
        #expect(yarn.inverseFrequencies[last] < plain.inverseFrequencies[last])
    }

    @Test("Each family gets the RoPE its spec implies")
    func ropeFollowsSpec() {
        let qwen = RoPE.Configuration.forSpec(.qwen3MoE30B)
        #expect(qwen.scalingFactor == 1)
        #expect(qwen.theta == 1_000_000)
        #expect(qwen.headDimension == 128)

        let oss = RoPE.Configuration.forSpec(.gptOSS120B)
        #expect(oss.scalingFactor == 32)
        #expect(oss.theta == 150_000)
        #expect(oss.headDimension == 64)

        // The 20B must not quietly diverge from the 120B here.
        #expect(RoPE.Configuration.forSpec(.gptOSS20B).scalingFactor == 32)
    }

    /// Qwen3 marks reasoning with `<think>`, GPT-OSS with harmony channels.
    /// Both have to come apart, because the dashboard streams them separately.
    @Test("Reply splitting handles both reasoning conventions")
    func splitsBothReasoningStyles() {
        let qwen = Conversation.split("<think>weigh it up</think>Paris.")
        #expect(qwen.analysis == "weigh it up")
        #expect(qwen.final == "Paris.")

        // A think block still open means reasoning has begun and no answer
        // exists yet — the case that used to render as an empty reply.
        let midThought = Conversation.split("<think>still going")
        #expect(midThought.final.isEmpty)
        #expect(midThought.analysis?.contains("still going") == true)

        let harmony = Conversation.split(
            "<|channel|>analysis<|message|>hmm<|end|>"
            + "<|channel|>final<|message|>Paris.<|end|>")
        #expect(harmony.analysis == "hmm")
        #expect(harmony.final == "Paris.")

        // Plain text with neither convention is all answer.
        #expect(Conversation.split("Just an answer.").final == "Just an answer.")
    }

    /// The expert layout has to describe both families, and the sizes are what
    /// the installer writes and the reader reads — a disagreement here is
    /// corrupt weights rather than a crash.
    @Test("Expert layout matches Qwen3's shape")
    func qwen3ExpertLayout() {
        let spec = ArchitectureSpec.qwen3MoE30B
        let layout = ExpertLayout(hiddenSize: spec.hiddenSize,
                                  intermediateSize: spec.intermediateSize,
                                  expertCount: 128,
                                  expertBias: spec.expertBias)

        // gate and up interleaved: 2 x 768 rows of 2048, at 16 bytes per
        // 32-weight block.
        #expect(layout.size(of: .gateUpBlocks) == 2 * 768 * (2048 / 32) * 16)
        #expect(layout.size(of: .gateUpScales) == 2 * 768 * (2048 / 32))
        #expect(layout.size(of: .downBlocks) == 2048 * (768 / 32) * 16)
        #expect(layout.size(of: .downScales) == 2048 * (768 / 32))

        // Biases are allocated even though Qwen3 has none: the kernel adds one
        // unconditionally, and a zero-length section would leave it reading
        // whatever alignment padding follows.
        #expect(layout.size(of: .gateUpBias) > 0)
        #expect(layout.size(of: .downBias) > 0)

        // Every section starts on a page so a read never shares one with a
        // section the caller did not ask for.
        for section in ExpertLayout.Section.allCases {
            #expect(layout.offset(expert: 0, section: section)
                    % ExpertLayout.pageSize == 0)
        }
        #expect(layout.stride % ExpertLayout.pageSize == 0)
    }

    @Test("Both families still describe themselves consistently")
    func specsRemainCoherent() {
        for spec in [ArchitectureSpec.gptOSS120B, .gptOSS20B, .qwen3MoE30B] {
            #expect(spec.headDimension > 0)
            #expect(spec.attentionHeads % spec.keyValueHeads == 0,
                    "\(spec.name): grouped-query attention needs whole groups")
            #expect(spec.hiddenSize % MXFP4.blockSize == 0,
                    "\(spec.name): expert rows must divide into whole MXFP4 blocks")
            #expect(spec.intermediateSize % MXFP4.blockSize == 0,
                    "\(spec.name): down_proj rows must divide into whole blocks")
        }
    }
}
