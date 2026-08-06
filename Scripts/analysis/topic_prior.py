#!/usr/bin/env python3
"""Does the range map predict which experts a generation will use?

Cache-aware routing (arXiv:2412.00099 and others) raises hit rate by biasing
selection toward resident experts. That works, and it changes which experts
run, so it trades output quality for speed.

This asks whether a different lever exists here: the range map already measures
what each expert specialises in. If a generation about Python disproportionately
uses the experts the map labels `python`, the cache could be pre-warmed from the
prompt's topic — changing only what is resident, never what runs, so the output
stays bit-identical.

That whole idea rests on one number: enrichment. Of the experts that actually
fire during a Python generation, how many more than chance are `python`
experts? At 1.0x there is no signal and the idea is dead. This measures it
before any cache work is done.

    python3 Scripts/analysis/topic_prior.py scratch/qwen3.gwt \\
        python:/tmp/trace_python.jsonl chinese:/tmp/trace_chinese.jsonl
"""

import collections
import json
import pathlib
import sys


def load_map(root):
    """expert (layer, index) -> topic, for experts confident enough to label."""
    data = json.loads((pathlib.Path(root) / "range.json").read_text())
    labelled, by_topic = {}, collections.Counter()
    for point in data["points"]:
        # Under-sampled experts carry a topic label that is noise; the map
        # records that, so honour it here rather than inflating the prior.
        if not point.get("confident", True):
            continue
        if point["specialisation"] < 0.35:
            continue
        labelled[(point["layer"], point["expert"])] = point["topic"]
        by_topic[point["topic"]] += 1
    return labelled, by_topic, data["topics"]


def load_trace(path):
    fired = collections.Counter()
    total = 0
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        decision = json.loads(line)
        for expert in decision["experts"]:
            fired[(decision["layer"], expert)] += 1
            total += 1
    return fired, total


def main():
    root = sys.argv[1]
    traces = [arg.split(":", 1) for arg in sys.argv[2:]]

    labelled, by_topic, topics = load_map(root)
    total_labelled = sum(by_topic.values())

    print(f"range map: {total_labelled} experts carry a confident topic label")
    print("           " + ", ".join(f"{t} {by_topic[t]}" for t in topics
                                    if by_topic[t]) + "\n")

    print(f"{'trace':>10}{'topic':>10}{'share fired':>14}{'share of map':>14}"
          f"{'enrichment':>13}")
    print("-" * 61)

    for name, path in traces:
        fired, total_fires = load_trace(path)
        # Weight by how often each expert fired, not just whether it did: an
        # expert used 40 times matters more than one used twice.
        topic_fires = collections.Counter()
        labelled_fires = 0
        for expert, count in fired.items():
            topic = labelled.get(expert)
            if topic:
                topic_fires[topic] += count
                labelled_fires += count

        for topic in topics:
            if by_topic[topic] == 0 or topic_fires[topic] == 0:
                continue
            share_fired = topic_fires[topic] / max(labelled_fires, 1)
            share_map = by_topic[topic] / total_labelled
            lift = share_fired / share_map if share_map else 0
            flag = ""
            if topic == name:
                flag = "  <-- matching topic"
            if abs(lift - 1) < 0.05 and topic == name:
                flag += " (no signal)"
            print(f"{name:>10}{topic:>10}{share_fired:>13.1%}{share_map:>13.1%}"
                  f"{lift:>12.2f}x{flag}")
        print("-" * 61)

    print("\nEnrichment is share-of-fires divided by share-of-map. 1.0x means the")
    print("generation used that topic's experts exactly as often as their")
    print("presence in the map predicts — i.e. the label carries no information")
    print("about what will fire, and pre-warming from topic cannot help.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
