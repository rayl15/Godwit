import Testing

@testable import Godwit

@Suite("Expert cache planner")
struct ExpertCachePlannerTests {
    @Test("First request misses everything")
    func coldStartMisses() {
        var planner = ExpertCachePlanner(slotCount: 4, expertCount: 16)
        let plan = planner.plan(experts: [3, 7])

        #expect(plan.missCount == 2)
        #expect(plan.hitCount == 0)
        #expect(Set(plan.slots).count == 2, "distinct experts need distinct slots")
    }

    @Test("Repeating a request hits the cache")
    func warmRepeatHits() {
        var planner = ExpertCachePlanner(slotCount: 4, expertCount: 16)
        let first = planner.plan(experts: [3, 7])
        let second = planner.plan(experts: [3, 7])

        #expect(second.missCount == 0)
        #expect(second.slots == first.slots, "residency should be stable")
    }

    @Test("A duplicate expert in one request is fetched once")
    func duplicateWithinRequest() {
        var planner = ExpertCachePlanner(slotCount: 4, expertCount: 16)
        _ = planner.plan(experts: [5])
        let plan = planner.plan(experts: [5, 5])

        #expect(plan.missCount == 0)
        #expect(plan.slots[0] == plan.slots[1])
    }

    @Test("Eviction spares the frequently used expert")
    func evictsLeastFrequent() {
        var planner = ExpertCachePlanner(slotCount: 2, expertCount: 8)
        // Expert 1 is used repeatedly; expert 2 only once.
        _ = planner.plan(experts: [1, 2])
        _ = planner.plan(experts: [1])
        _ = planner.plan(experts: [1])

        // Expert 3 must displace something, and it should not be expert 1.
        let plan = planner.plan(experts: [3])
        #expect(plan.missCount == 1)
        #expect(planner.residentExperts.contains(1), "hot expert must survive")
        #expect(!planner.residentExperts.contains(2), "cold expert should be evicted")
    }

    @Test("Never assigns two experts the same slot")
    func slotsStayDisjoint() {
        var planner = ExpertCachePlanner(slotCount: 8, expertCount: 64)
        // A deterministic stride that revisits experts irregularly.
        for step in 0..<200 {
            let experts = (0..<6).map { (step * 7 + $0 * 11) % 64 }
            let plan = planner.plan(experts: Array(Set(experts)))
            #expect(Set(plan.slots).count == plan.experts.count)
            #expect(plan.slots.allSatisfy { $0 >= 0 && $0 < 8 })
        }
    }

    @Test("A skewed workload converges to a high hit rate")
    func skewedWorkloadCaches() {
        var planner = ExpertCachePlanner(slotCount: 16, expertCount: 128)
        var hits = 0
        var total = 0
        // 80% of requests draw from a hot set of 12, the rest are uniform.
        for step in 0..<1_000 {
            let experts: [Int] = (0..<8).map { position in
                let draw = (step * 31 + position * 17) % 10
                return draw < 8 ? (step * 3 + position) % 12
                                : 12 + ((step * 13 + position * 7) % 116)
            }
            let unique = Array(Set(experts))
            let plan = planner.plan(experts: unique)
            hits += plan.hitCount
            total += unique.count
        }
        let hitRate = Double(hits) / Double(total)
        #expect(hitRate > 0.5, "skewed routing should cache well, got \(hitRate)")
    }
}
