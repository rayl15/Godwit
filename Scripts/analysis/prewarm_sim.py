#!/usr/bin/env python3
"""Does pre-warming the expert cache from a topic prior raise the hit rate?

`topic_prior.py` establishes that the range map predicts what fires — a Python
generation uses `python` experts 2.0x more than chance, Chinese uses `chinese`
experts 5.6x more. This asks the only question that follows: does knowing that
in advance actually save reads?

Simulates the real policy — one cache per layer, 8 slots, LFU with LRU
tie-break — against real routing traces, with and without seeding the cache
from the topic prior. Nothing about routing changes, so output would be
bit-identical either way; the only difference is what happens to be resident.

    python3 Scripts/analysis/prewarm_sim.py scratch/qwen3.gwt \\
        python:/tmp/trace_python.jsonl
"""

import collections
import json
import pathlib
import sys


class LayerCache:
    """LFU with LRU tie-break, matching ExpertCachePlanner."""

    def __init__(self, slots):
        self.slots = slots
        self.resident = {}          # expert -> [uses, last_used]
        self.clock = 0

    def preload(self, experts):
        for expert in experts[:self.slots]:
            # Seeded entries start with no usage credit, so a wrong guess is
            # evicted as soon as anything real competes for the slot.
            self.resident[expert] = [0, -1]

    def access(self, expert):
        self.clock += 1
        if expert in self.resident:
            self.resident[expert][0] += 1
            self.resident[expert][1] = self.clock
            return True
        if len(self.resident) >= self.slots:
            victim = min(self.resident, key=lambda e: (self.resident[e][0],
                                                       self.resident[e][1]))
            del self.resident[victim]
        self.resident[expert] = [1, self.clock]
        return False


def topic_affinity(root):
    """(layer, expert) -> (topic, specialisation), confident labels only."""
    data = json.loads((pathlib.Path(root) / "range.json").read_text())
    out = {}
    for p in data["points"]:
        if not p.get("confident", True):
            continue
        out[(p["layer"], p["expert"])] = (p["topic"], p["specialisation"])
    return out


def simulate(trace, slots, seed_by_layer=None):
    caches = collections.defaultdict(lambda: LayerCache(slots))
    if seed_by_layer:
        for layer, experts in seed_by_layer.items():
            caches[layer].preload(experts)
    hits = misses = 0
    for decision in trace:
        cache = caches[decision["layer"]]
        for expert in decision["experts"]:
            if cache.access(expert):
                hits += 1
            else:
                misses += 1
    return hits, misses


def main():
    root = sys.argv[1]
    slots = 8
    affinity = topic_affinity(root)

    print(f"{'trace':>10}{'topic':>10}{'baseline':>11}{'prewarmed':>12}"
          f"{'reads saved':>14}")
    print("-" * 57)

    for arg in sys.argv[2:]:
        name, path = arg.split(":", 1)
        trace = [json.loads(l) for l in
                 pathlib.Path(path).read_text().splitlines() if l.strip()]

        # Seed each layer with its most specialised experts for this topic,
        # plus `chat` experts, which topic_prior.py shows fire for everything.
        seed = collections.defaultdict(list)
        for (layer, expert), (topic, spec) in affinity.items():
            if topic in (name, "chat"):
                seed[layer].append((spec if topic == name else spec * 0.5, expert))
        seed_by_layer = {layer: [e for _, e in sorted(v, reverse=True)]
                         for layer, v in seed.items()}

        base_h, base_m = simulate(trace, slots)
        warm_h, warm_m = simulate(trace, slots, seed_by_layer)
        base_rate = base_h / (base_h + base_m)
        warm_rate = warm_h / (warm_h + warm_m)
        saved = (base_m - warm_m) / base_m if base_m else 0

        print(f"{name:>10}{name:>10}{base_rate:>10.1%}{warm_rate:>11.1%}"
              f"{saved:>13.1%}")

    print("-" * 57)
    print("\nReads saved is the reduction in cache misses, which is the reduction")
    print("in bytes off disk — 71.6% of decode time. Routing is untouched, so")
    print("the generated text is identical either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
