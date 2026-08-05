import Foundation
import Testing

@testable import Godwit

@Suite("Expert range map")
struct ExpertRangeTests {
    @Test("An expert seen too few times is not credited with a preference")
    func thresholdMarksUndersampled() {
        // Two experts, both firing only on topic 0. One fired twice, the other
        // forty times. They have identical affinity vectors, so nothing except
        // the activation count can tell them apart — which is exactly the
        // failure the flag exists to catch.
        let table: [[[Float]]] = [[
            [2, 0, 0, 0],
            [40, 0, 0, 0],
        ]]
        let map = ExpertRange.assemble(
            counts: table, topics: ["a", "b", "c", "d"],
            layerCount: 1, expertCount: 2, minimumActivations: 8)

        let shy = map.points.first { $0.activations == 2 }
        let sure = map.points.first { $0.activations == 40 }
        #expect(shy?.confident == false)
        #expect(sure?.confident == true)
        // Both still score as pure specialists, which is the point: the
        // specialisation number alone cannot distinguish them.
        #expect(shy?.specialisation == sure?.specialisation)
        #expect(map.minimumActivations == 8)
    }

    @Test("Under-sampled experts stay in the map rather than being dropped")
    func undersampledStillPlotted() {
        let table: [[[Float]]] = [[[1, 0], [1, 0], [50, 2]]]
        let map = ExpertRange.assemble(
            counts: table, topics: ["a", "b"],
            layerCount: 1, expertCount: 3, minimumActivations: 10)
        #expect(map.points.count == 3, "they are real routing, just faint")
        #expect(map.points.filter { $0.confident }.count == 1)
    }

    @Test("An expert that never fires is absent entirely")
    func silentExpertOmitted() {
        let table: [[[Float]]] = [[[0, 0], [5, 5]]]
        let map = ExpertRange.assemble(
            counts: table, topics: ["a", "b"],
            layerCount: 1, expertCount: 2, minimumActivations: 1)
        #expect(map.points.count == 1)
        #expect(map.points[0].expert == 1)
    }

    @Test("Concentration is 0 for a generalist and 1 for a pure specialist")
    func concentrationBounds() {
        #expect(ExpertRange.concentration([1, 1, 1, 1]) == 0)
        #expect(ExpertRange.concentration([1, 0, 0, 0]) == 1)
        // Split evenly across two of four is between the extremes, and is
        // distinguishable from a split across all four — which a plain
        // max-share measure could not manage.
        let half = ExpertRange.concentration([1, 1, 0, 0])
        #expect(half > 0 && half < 1)
        #expect(half > ExpertRange.concentration([2, 1, 1, 1]))
    }

    @Test("Several probes sharing a topic collapse into one axis")
    func probesGroupByTopic() {
        // Two probes per topic must not give a topic two columns, which would
        // split its own experts across both and halve every count.
        var seen: [String] = []
        var topicOf: [Int] = []
        for probe in ExpertRange.defaultProbes {
            if let existing = seen.firstIndex(of: probe.topic) {
                topicOf.append(existing)
            } else {
                topicOf.append(seen.count)
                seen.append(probe.topic)
            }
        }
        #expect(seen.count == 12, "twelve topics")
        #expect(ExpertRange.defaultProbes.count == seen.count * 2,
                "two probes each, so labels rest on more than one sample")
        #expect(topicOf.max() == seen.count - 1)
    }

    @Test("Every probe topic is sampled by more than one text")
    func everyTopicSampledTwice() {
        var perTopic: [String: Int] = [:]
        for probe in ExpertRange.defaultProbes {
            perTopic[probe.topic, default: 0] += 1
        }
        #expect(perTopic.values.allSatisfy { $0 >= 2 })
        // Distinct text, not the same string twice — otherwise the second
        // sample measures nothing new.
        let texts = Set(ExpertRange.defaultProbes.map(\.text))
        #expect(texts.count == ExpertRange.defaultProbes.count)
    }
}
