#!/usr/bin/env python3
"""Can a layer's experts be predicted one layer early?

RESULTS.md records that adjacent layers share only 4.8% of their experts, and
for a while this repo concluded from that that prefetching does not work. It
does not follow: reusing the previous layer's *choices* fails, but that is not
what a lookahead scheme does. colibrì runs the next layer's router early, and
Pre-Attention Expert Prediction (arXiv:2511.10676) fits a linear probe to
pre-attention activations.

This measures the first of those directly. At layer n the residual entering the
layer is available before attention runs, so the question is:

    topk( router_n( norm_n( h_in ) ) )   vs   the selection actually made

The difference between them is exactly one attention block plus a norm. If the
overlap is high, the experts can be fetched a layer early for free. If it is
near chance, the idea is dead here regardless of what it does elsewhere.

    python3 Scripts/analysis/lookahead_accuracy.py hidden.bin routing.jsonl model.gwt
"""

import json
import pathlib
import sys

import numpy as np


def bf16(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class Trunk:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.sections = {s["name"]: s for s in self.manifest["trunkSections"]}
        self.file = open(self.root / "trunk.bin", "rb")

    def raw(self, name):
        s = self.sections[name]
        self.file.seek(s["offset"])
        return self.file.read(s["length"]), s

    def bf16(self, name):
        raw, _ = self.raw(name)
        return bf16(raw)

    def int8(self, name):
        raw, spec = self.raw(name)
        rows, cols = spec["shape"]
        groups = cols // 64
        codes = np.frombuffer(raw[:rows * cols], dtype=np.uint8).reshape(rows, groups, 64)
        meta = bf16(raw[rows * cols:rows * cols + rows * groups * 4]).reshape(rows, groups, 2)
        return (codes * meta[:, :, 0:1] + meta[:, :, 1:2]).reshape(rows, cols)


def main():
    hidden_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hidden_q3.bin"
    routing_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/routing_q3.jsonl"
    root = sys.argv[3] if len(sys.argv) > 3 else "scratch/qwen3.gwt"

    trunk = Trunk(root)
    spec = trunk.manifest["spec"]
    width = spec["hiddenSize"]
    layers = trunk.manifest["layerCount"]
    experts = trunk.manifest["expertCount"]
    eps = spec["rmsNormEpsilon"]

    actual = [json.loads(l) for l in pathlib.Path(routing_path).read_text().splitlines() if l.strip()]
    top_k = len(actual[0]["experts"])

    h = np.fromfile(hidden_path, dtype=np.float32).reshape(-1, width)
    assert len(h) == len(actual), f"{len(h)} vectors vs {len(actual)} decisions"

    # BF16, not int8: the router is small and decides which experts fire, so
    # the installer keeps it at full width. Reading it as int8 produces a
    # matrix that predicts at exactly chance — which looks like a null result
    # about the idea rather than a bug in this script, so it is validated
    # against the true router input below before anything is concluded.
    routers = {n: trunk.bf16(f"layer{n}.router_w").reshape(experts, width)
               for n in range(layers)}
    norms = {n: trunk.bf16(f"layer{n}.post_norm") for n in range(layers)}
    # GPT-OSS's router carries a bias and Qwen3's does not. Omitting it cost
    # 10 points of agreement on the validation above — which is what that check
    # is for.
    have_bias = any(x["name"] == "layer0.router_b" for x in trunk.manifest["trunkSections"])
    biases = ({n: trunk.bf16(f"layer{n}.router_b") for n in range(layers)}
              if have_bias else {n: 0.0 for n in range(layers)})

    print(f"{trunk.manifest['model']}   {len(actual)} decisions, "
          f"top-{top_k} of {experts}, {layers} layers\n")

    # Validate before measuring. Applied to the vector the router actually
    # consumed, this must reproduce the engine's selection exactly; anything
    # less means the numbers below describe this script rather than the model.
    truth_path = pathlib.Path(hidden_path + ".router")
    if truth_path.exists():
        true_in = np.fromfile(truth_path, dtype=np.float32).reshape(-1, width)
        agree = 0
        for row, decision in zip(true_in, actual):
            n = decision["layer"]
            pick = set(np.argpartition(-(routers[n] @ row + biases[n]),
                                       top_k)[:top_k].tolist())
            agree += len(pick & set(decision["experts"]))
        rate = agree / (len(actual) * top_k)
        print(f"validation: router on its true input reproduces {rate:.1%} "
              f"of the engine's picks")
        if rate < 0.99:
            print("ABORT — the reimplementation is wrong, so nothing below means anything")
            return 1
        print()

    hits, total, per_layer = 0, 0, {}
    first_hits, first_total = 0, 0
    for row, decision in zip(h, actual):
        n = decision["layer"]
        # What the router would see if it ran before this layer's attention.
        normed = row / np.sqrt((row ** 2).mean() + eps) * norms[n]
        logits = routers[n] @ normed + biases[n]
        predicted = set(np.argpartition(-logits, top_k)[:top_k].tolist())
        truth = decision["experts"]
        overlap = len(predicted & set(truth))
        hits += overlap
        total += top_k
        per_layer.setdefault(n, [0, 0])
        per_layer[n][0] += overlap
        per_layer[n][1] += top_k
        # The top-ranked expert carries the most routing weight, so getting it
        # right matters more than the tail.
        if truth[0] in predicted:
            first_hits += 1
        first_total += 1

    chance = top_k / experts
    rate = hits / total
    print(f"one-layer-early prediction: {rate:.1%} of experts correct")
    print(f"chance:                     {chance:.1%}")
    print(f"lift over chance:           {rate / chance:.1f}x")
    print(f"top-1 expert recovered:     {first_hits / first_total:.1%}\n")

    depths = sorted(per_layer)
    print("by depth")
    print(f"{'layers':>12}{'accuracy':>11}")
    print("-" * 23)
    band = max(1, len(depths) // 6)
    for start in range(0, len(depths), band):
        chunk = depths[start:start + band]
        got = sum(per_layer[d][0] for d in chunk)
        want = sum(per_layer[d][1] for d in chunk)
        print(f"{f'{chunk[0]}-{chunk[-1]}':>12}{got / want:>10.1%}")
    print("-" * 23)

    # What it would buy. A prefetch is only useful if it lands before the read
    # would otherwise start; a wrong prediction costs a wasted read.
    print(f"\nAt {rate:.0%} accuracy, prefetching top-{top_k} a layer early lands")
    print(f"{rate * top_k:.1f} of {top_k} experts early and wastes "
          f"{(1 - rate) * top_k:.1f} reads unless verified before issue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
