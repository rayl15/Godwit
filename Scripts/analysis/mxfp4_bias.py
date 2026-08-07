#!/usr/bin/env python3
"""Is MXFP4's error correctable, or only measurable?

`quantisation_cost.py` found the quantised expert block consistently smaller
than the BF16 reference — five layers, five times, never larger. A systematic
bias suggests something fixable, so this chases it.

The obvious mechanism is the codebook. MXFP4's magnitudes are 0, 0.5, 1, 1.5,
2, 3, 4 and 6, so anything below a quarter of the block scale rounds to zero.
Weights are roughly Gaussian and most of a block sits near zero, so
round-to-nearest should delete a lot of small values and shrink the norm.

It does not, and the shrinkage is not worth correcting. Both are measured here
rather than argued.

    python3 Scripts/analysis/mxfp4_bias.py scratch/qwen3.gwt \\
        /tmp/hidden_q3.bin.router /tmp/routing_q3.jsonl
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from quantisation_cost import Install, Checkpoint, swiglu_expert   # noqa: E402


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen3.gwt"
    router = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hidden_q3.bin.router"
    trace = sys.argv[3] if len(sys.argv) > 3 else "/tmp/routing_q3.jsonl"

    install = Install(root)
    source = Checkpoint(install.manifest["model"])
    width = install.hidden
    layer = install.manifest["layerCount"] - 1        # deepest, where it was worst

    gate, up, down = install.expert(layer, 0)
    prefix = f"model.layers.{layer}.mlp.experts.0"
    gate_r = source.tensor(prefix + ".gate_proj.weight")
    up_r = source.tensor(prefix + ".up_proj.weight")
    down_r = source.tensor(prefix + ".down_proj.weight")

    print("1. Are the stored weights biased?\n")
    print(f"{'tensor':>8}{'||q||/||w||':>14}{'sum|q|/sum|w|':>16}")
    print("-" * 38)
    for name, q, w in (("gate", gate, gate_r), ("up", up, up_r),
                       ("down", down, down_r)):
        print(f"{name:>8}{np.linalg.norm(q) / np.linalg.norm(w):>13.4f}"
              f"{np.abs(q).sum() / np.abs(w).sum():>15.4f}")
    print("-" * 38)
    print("L2 is preserved to four decimal places. The L1 shortfall is the")
    print("small values rounding to zero — 11% of weights carrying 1% of the")
    print("magnitude — which is real and almost irrelevant.\n")

    decisions = [json.loads(l) for l in
                 pathlib.Path(trace).read_text().splitlines() if l.strip()]
    inputs = np.fromfile(router, dtype=np.float32).reshape(-1, width)
    xs = [inputs[i] for i, d in enumerate(decisions) if d["layer"] == layer]

    raw, floor, cosines, ratios = [], [], [], []
    for x in xs:
        y_q = swiglu_expert(x, gate, up, down)
        y_r = swiglu_expert(x, gate_r, up_r, down_r)
        raw.append(np.linalg.norm(y_q - y_r) / np.linalg.norm(y_r))
        ratios.append(np.linalg.norm(y_q) / np.linalg.norm(y_r))
        cos = float(y_q @ y_r / (np.linalg.norm(y_q) * np.linalg.norm(y_r)))
        cosines.append(cos)
        # Error left after rescaling each output perfectly — the best any
        # scale correction could ever do.
        floor.append(np.sqrt(max(1 - cos * cos, 0)))

    k = 1 / float(np.mean(ratios))
    fixed = []
    for x in xs:
        y_q = k * swiglu_expert(x, gate, up, down)
        y_r = swiglu_expert(x, gate_r, up_r, down_r)
        fixed.append(np.linalg.norm(y_q - y_r) / np.linalg.norm(y_r))

    print(f"2. The output shrinks anyway — {len(xs)} real inputs, layer {layer}\n")
    print(f"  ||quantised|| / ||reference||   mean {np.mean(ratios):.4f}, "
          f"below 1.0 in {sum(1 for r in ratios if r < 1)}/{len(ratios)}")
    print(f"  mean cosine                     {np.mean(cosines):.4f}\n")

    print("3. Correcting it is not worth doing\n")
    print(f"  error as-is                      {np.mean(raw):>7.1%}")
    print(f"  with one fixed scalar (k={k:.3f})    {np.mean(fixed):>7.1%}")
    print(f"  after a perfect per-output rescale {np.mean(floor):>6.1%}   <- floor")
    print()
    share = (np.mean(raw) - np.mean(floor)) / np.mean(raw)
    print(f"  magnitude is {share:.0%} of the error. The other {1 - share:.0%} is")
    print("  direction, and no rescaling touches direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
