"""Expert-cache hit-rate simulation.

Same policy as Godwit's ExpertCachePlanner: LFU with LRU tie-break, one
independent cache per layer. Routing is modelled as Zipf-distributed expert
popularity, drawn independently per layer -- justified by TurboFieldfare's
finding that one layer's choices predict only 7% of the next layer's.

The Zipf exponent is the one free parameter, so we calibrate it against a
published measurement rather than guessing: TurboFieldfare reports expert I/O
falling from 166 to 88 ms/token at 16 slots with top-8 of 128, i.e. a ~47% hit
rate. Whatever exponent reproduces that is the exponent we carry to GPT-OSS.
"""
import random

def zipf_weights(n, alpha):
    w = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    total = sum(w)
    return [x / total for x in w]

def simulate(n_experts, top_k, slots, alpha, tokens=4000, seed=7):
    rng = random.Random(seed)
    weights = zipf_weights(n_experts, alpha)
    population = list(range(n_experts))

    slot_expert = [None] * slots
    use_count = [0] * n_experts
    last_touch = [0] * slots
    clock = 0
    hits = misses = 0

    for _ in range(tokens):
        clock += 1
        chosen = set()
        while len(chosen) < top_k:
            chosen.update(rng.choices(population, weights=weights, k=top_k - len(chosen)))
        chosen = list(chosen)

        resident = {e: s for s, e in enumerate(slot_expert) if e is not None}
        taken = set()
        need = []
        for e in chosen:
            if e in resident:
                hits += 1
                taken.add(resident[e])
                last_touch[resident[e]] = clock
            else:
                need.append(e)

        free = [s for s in range(slots) if s not in taken]
        # Worst-first: empty slots, then fewest uses, then least recently touched.
        free.sort(key=lambda s: (
            slot_expert[s] is not None,
            use_count[slot_expert[s]] if slot_expert[s] is not None else -1,
            last_touch[s]))
        for e, s in zip(need, free):
            slot_expert[s] = e
            last_touch[s] = clock
            misses += 1
        for e in chosen:
            use_count[e] += 1

    return hits / (hits + misses)

# --- Calibrate against TurboFieldfare's published 166 -> 88 ms/token ---
TARGET = 1 - 88 / 166
print(f"Calibration target (TurboFieldfare, top-8/128, 16 slots): {TARGET:.1%} hit rate\n")
print("alpha  simulated hit rate")
best = None
for alpha in [x / 20 for x in range(0, 31)]:
    hr = simulate(128, 8, 16, alpha)
    if best is None or abs(hr - TARGET) < abs(best[1] - TARGET):
        best = (alpha, hr)
    if alpha * 20 % 5 == 0:
        print(f"{alpha:4.2f}   {hr:.1%}")
alpha_fit = best[0]
print(f"\nBest fit: alpha = {alpha_fit:.2f} (hit rate {best[1]:.1%})\n")

# --- Apply to GPT-OSS-120B: 128 experts, top-4, 36 layers ---
print("GPT-OSS-120B (128 experts, top-4) at the calibrated exponent:")
print("slots/layer   cache RAM    hit rate   reads/token")
EXPERT_MIB = 13219200 / 1048576
for slots in [4, 6, 8, 12, 16]:
    hr = simulate(128, 4, slots, alpha_fit)
    ram = slots * 36 * EXPERT_MIB / 1024
    reads = 4 * 36 * (1 - hr)
    print(f"{slots:>6}      {ram:6.2f} GiB    {hr:6.1%}     {reads:6.1f}")

print("\nSensitivity -- hit rate if routing is more/less skewed than calibrated:")
print("alpha   4 slots   8 slots")
for alpha in [0.0, 0.5, alpha_fit, 1.0, 1.5]:
    a4 = simulate(128, 4, 4, alpha)
    a8 = simulate(128, 4, 8, alpha)
    tag = "  <- calibrated" if abs(alpha - alpha_fit) < 1e-9 else ""
    print(f"{alpha:4.2f}   {a4:6.1%}   {a8:6.1%}{tag}")
