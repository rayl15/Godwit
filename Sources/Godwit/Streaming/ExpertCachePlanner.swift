import Foundation

/// The outcome of planning one token's expert working set against the cache.
public struct CachePlan: Sendable, Equatable {
    /// Requested experts, in the order the router produced them.
    public let experts: [Int]
    /// `slots[i]` is where `experts[i]` will live once the plan is executed.
    public let slots: [Int]
    /// Indices into `experts` that must be read from storage.
    public let missIndices: [Int]

    public var hitCount: Int { experts.count - missIndices.count }
    public var missCount: Int { missIndices.count }

    /// Experts that must actually be fetched, in request order.
    public var missingExperts: [Int] { missIndices.map { experts[$0] } }

    /// Slots that misses will be read into, parallel to `missingExperts`.
    public var missingSlots: [Int] { missIndices.map { slots[$0] } }
}

/// Chooses which cache slot each routed expert should occupy.
///
/// This is deliberately pure: it decides placement and reports what must be
/// fetched, but performs no I/O and holds no buffers. That keeps the eviction
/// policy testable without a GPU, a model, or a filesystem — the policy is the
/// part most likely to change, and the part where mistakes are quietest.
///
/// Eviction is least-frequently-used, tie-broken by least-recently-used. LFU
/// wins here because expert popularity within a single generation is heavily
/// skewed and fairly stable, so recency alone keeps evicting experts that are
/// about to be wanted again.
public struct ExpertCachePlanner: Sendable {
    private struct Slot {
        var expert: Int?
        var lastTouched: Int
    }

    public let slotCount: Int
    public let expertCount: Int

    private var slots: [Slot]
    private var useCount: [Int]
    private var clock: Int

    /// - Parameters:
    ///   - slotCount: resident experts per layer. Must be at least the router's
    ///     top-k, or a single token could not be planned.
    ///   - expertCount: total experts in the layer, for the frequency table.
    public init(slotCount: Int, expertCount: Int) {
        precondition(slotCount > 0, "need at least one slot")
        precondition(expertCount > 0, "need at least one expert")
        self.slotCount = slotCount
        self.expertCount = expertCount
        self.slots = Array(repeating: Slot(expert: nil, lastTouched: 0), count: slotCount)
        self.useCount = Array(repeating: 0, count: expertCount)
        self.clock = 0
    }

    /// Experts currently resident, for tests and diagnostics.
    public var residentExperts: [Int] {
        slots.compactMap(\.expert)
    }

    /// Plans placement for one token's experts and commits the result.
    ///
    /// The planner assumes the caller will fetch every miss before the layer
    /// runs; a caller that abandons a plan must discard this planner too, since
    /// the slots are already marked as holding the new experts.
    public mutating func plan(experts: [Int]) -> CachePlan {
        precondition(experts.count <= slotCount,
                     "cannot hold \(experts.count) experts in \(slotCount) slots")
        precondition(experts.allSatisfy { $0 >= 0 && $0 < expertCount },
                     "expert id out of range")

        clock += 1

        var assigned = [Int](repeating: -1, count: experts.count)
        var taken = [Bool](repeating: false, count: slotCount)
        var missIndices: [Int] = []

        // Pass 1: honour residency. A duplicate expert id in one request maps to
        // the same slot rather than being fetched twice.
        var residentSlot: [Int: Int] = [:]
        for (slot, contents) in slots.enumerated() {
            if let expert = contents.expert { residentSlot[expert] = slot }
        }
        for (index, expert) in experts.enumerated() {
            guard let slot = residentSlot[expert] else { continue }
            assigned[index] = slot
            taken[slot] = true
            slots[slot].lastTouched = clock
        }

        // Pass 2: place misses into the least valuable free slots.
        for index in experts.indices where assigned[index] == -1 {
            missIndices.append(index)
        }
        if !missIndices.isEmpty {
            var candidates = (0..<slotCount).filter { !taken[$0] }
            candidates.sort(by: isLessValuable)

            for (offset, index) in missIndices.enumerated() {
                let slot = candidates[offset]
                assigned[index] = slot
                taken[slot] = true
                slots[slot].expert = experts[index]
                slots[slot].lastTouched = clock
            }
        }

        // Frequency is counted per request, after placement, so an expert's own
        // arrival never protects it from eviction within the same token.
        for expert in Set(experts) {
            useCount[expert] += 1
        }

        return CachePlan(experts: experts, slots: assigned, missIndices: missIndices)
    }

    /// Orders slots worst-to-best as eviction candidates.
    private func isLessValuable(_ lhs: Int, _ rhs: Int) -> Bool {
        // Empty slots are always the cheapest thing to fill.
        switch (slots[lhs].expert, slots[rhs].expert) {
        case (nil, nil): return lhs < rhs
        case (nil, _): return true
        case (_, nil): return false
        case let (leftExpert?, rightExpert?):
            let leftUses = useCount[leftExpert]
            let rightUses = useCount[rightExpert]
            if leftUses != rightUses { return leftUses < rightUses }
            return slots[lhs].lastTouched < slots[rhs].lastTouched
        }
    }
}
