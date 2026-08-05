#!/usr/bin/env python3
"""Would serving low-weight experts at lower precision be cheap or ruinous?

Expert reads are 71.6% of decode and run at device speed, so the only large
lever left is reading fewer bytes. HOBBIT (arXiv:2411.01433) proposes reading
cache-miss experts at reduced precision, on the argument that less critical
experts tolerate it. Before writing a 2-bit kernel it is worth knowing whether
that argument holds for the models here.

The MoE output is a weighted sum:

    y = sum_r  w_r * E_r(x)

so perturbing expert r by relative error e_r moves y by roughly w_r * e_r *
||E_r(x)||. Two things therefore decide whether the idea works, and both are
measured here rather than assumed:

  1. How lopsided are the router weights? If rank 8 carries 2% of the mass,
     degrading it is nearly free. If it carries 10%, it is not.
  2. What does coarser quantisation actually cost, per expert, on real tensors?

Combining them gives an estimated output error for any policy of the form
"serve ranks >= R at B bits", against the bytes that policy saves.

    python3 Scripts/analysis/precision_by_rank.py routing.jsonl model.gwt
"""

import json
import pathlib
import sys

import numpy as np

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
               dtype=np.float32)
MAG = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=np.float32)


def load_weights(path):
    """Router weights per decision, shape (decisions, top_k)."""
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line)["weights"])
    return np.array(rows, dtype=np.float64)


class Install:
    """Reads and dequantises one expert from a .gwt install."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        spec = self.manifest["spec"]
        self.hidden = spec["hiddenSize"]
        self.inner = spec["intermediateSize"]

    def section(self, layer, expert, name):
        span = self.manifest["expertSections"][name]
        offset = expert * self.manifest["expertStride"] + span["offset"]
        with open(self.root / "experts" / f"layer_{layer:02d}.bin", "rb") as handle:
            handle.seek(offset)
            return handle.read(span["length"])

    def gate_up(self, layer, expert):
        rows, cols = 2 * self.inner, self.hidden
        packed = np.frombuffer(self.section(layer, expert, "gate_up_blocks"),
                               dtype=np.uint8).reshape(rows, cols // 32, 16)
        scales = np.frombuffer(self.section(layer, expert, "gate_up_scales"),
                               dtype=np.uint8).reshape(rows, cols // 32)
        out = np.empty((rows, cols // 32, 32), dtype=np.float32)
        out[:, :, 0::2] = FP4[packed & 0x0F]
        out[:, :, 1::2] = FP4[packed >> 4]
        out *= np.exp2(scales.astype(np.int32) - 127)[:, :, None]
        return out.reshape(rows, cols)


def requantise(weights, bits, block=32):
    """Re-quantise already-MXFP4 weights to a coarser grid.

    Not a second MXFP4 pass: at 2 and 3 bits the codebook is a uniform subset
    of the magnitudes, sharing the same per-block power-of-two scale, which is
    what a low-precision tier would plausibly store.
    """
    if bits >= 4:
        return weights
    levels = MAG[:: (8 // (2 ** (bits - 1)))]        # 2 bit -> 2 mags, 3 -> 4
    flat = weights.reshape(-1, block)
    peak = np.abs(flat).max(axis=1, keepdims=True)
    scale = np.where(peak > 0, np.exp2(np.ceil(np.log2(np.maximum(peak, 1e-30) / 6.0))), 1.0)
    scaled = np.abs(flat) / scale
    index = np.abs(scaled[..., None] - levels[None, None, :]).argmin(axis=-1)
    return (np.sign(flat) * levels[index] * scale).reshape(weights.shape)


def gemv_error(exact, approx, trials=6, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        x = rng.standard_normal(exact.shape[1]).astype(np.float32)
        x /= np.linalg.norm(x)
        a, b = exact @ x, approx @ x
        worst = max(worst, float(np.linalg.norm(a - b) / np.linalg.norm(a)))
    return worst


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "/tmp/routing_q3.jsonl"
    root = sys.argv[2] if len(sys.argv) > 2 else "scratch/qwen3.gwt"

    w = load_weights(trace)
    decisions, top_k = w.shape
    install = Install(root)

    print(f"{install.manifest['model']}   {decisions} routing decisions, top-{top_k}\n")

    print("Router weight by rank")
    print(f"{'rank':>5}{'mean':>9}{'median':>9}{'share of mass':>16}")
    print("-" * 39)
    total = w.sum(axis=1).mean()
    for r in range(top_k):
        print(f"{r + 1:>5}{w[:, r].mean():>9.3f}{np.median(w[:, r]):>9.3f}"
              f"{w[:, r].mean() / total * 100:>15.1f}%")
    print("-" * 39)
    tail_from = top_k // 2
    tail = w[:, tail_from:].sum(axis=1).mean() / total
    print(f"ranks {tail_from + 1}-{top_k} carry {tail * 100:.1f}% of the routing mass\n")

    # What coarser precision costs on real expert tensors, measured not assumed.
    print("Quantisation error on real experts (relative error of W·x)")
    print(f"{'bits':>6}{'bits/weight':>13}{'gemv error':>13}")
    print("-" * 32)
    exact = install.gate_up(0, 0).astype(np.float32)
    costs = {}
    for bits in (4, 3, 2):
        approx = requantise(exact, bits)
        err = gemv_error(exact, approx)
        costs[bits] = err
        # 16 packed bytes + 1 scale byte per 32 weights at 4 bits.
        per_weight = bits + 8 / 32
        print(f"{bits:>6}{per_weight:>13.2f}{err:>12.1%}")
    print()

    # Policy: ranks >= R served at `bits`. Error into the residual is the
    # weighted sum of each degraded expert's own error.
    print("Estimated cost and benefit, per policy")
    print(f"{'policy':>28}{'added error':>14}{'bytes saved':>14}")
    print("-" * 56)
    baseline = costs[4]
    for bits in (3, 2):
        for cut in range(1, top_k):
            share = w[:, cut:].sum(axis=1).mean() / total
            # Extra error beyond what 4-bit already costs.
            added = share * max(costs[bits] - baseline, 0)
            saved = share * (1 - (bits + 8 / 32) / (4 + 8 / 32))
            label = f"ranks {cut + 1}-{top_k} at {bits} bit"
            flag = "  <-- worth trying" if saved > 0.15 and added < 0.02 else ""
            print(f"{label:>28}{added:>13.2%}{saved:>13.1%}{flag}")
    print("-" * 56)
    print("\nAdded error is relative to the residual contribution of the MoE block,")
    print("not to the whole model. Bytes saved is of expert traffic, which is")
    print("71.6% of decode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
