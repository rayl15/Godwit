#!/usr/bin/env python3
"""Does MXFP4 survive Qwen3's expert matrices?

GPT-OSS ships MXFP4, so the installer copies those bytes through untouched and
never has to ask this question. Qwen3 ships bf16, so we quantise it ourselves —
and the answer decides how much disk the install needs:

    MXFP4 experts   4.25 bit/weight   ~16.9 GB   fits alongside the 120B
    int8 experts    8.06 bit/weight   ~30.4 GB   requires deleting both

The worry is shape. GPT-OSS experts are 2880 wide; Qwen3's are 768. Narrower
rows give each 32-weight MXFP4 block less company to share an exponent with,
and a block whose values span a wide dynamic range is where 4 bits hurts.

Measured per tensor as signal-to-noise ratio in dB, and — because SNR on the
weights is not what anyone cares about — as the error in the output of an
actual matrix-vector product against a random unit input, which is what the
kernel will really compute.

    python3 Scripts/analysis/qwen3_quant_quality.py

Reads a handful of tensors over HTTP range requests. No full download.
"""

import json
import struct
import sys
import urllib.request

import numpy as np

REPO = "Qwen/Qwen3-30B-A3B"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

# OCP microscaling E2M1: the sixteen representable magnitudes, sign in bit 3.
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def get(url, start=None, end=None):
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(request) as response:
        return response.read()


def bf16_to_f32(raw):
    half = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (half << 16).view(np.float32) if False else \
        np.frombuffer((half << 16).astype(np.uint32).tobytes(), dtype=np.float32)


class Checkpoint:
    def __init__(self):
        index = json.loads(get(f"{BASE}/model.safetensors.index.json"))
        self.map = index["weight_map"]
        self.headers = {}

    def header(self, shard):
        if shard not in self.headers:
            url = f"{BASE}/{shard}"
            length = struct.unpack("<Q", get(url, 0, 7))[0]
            self.headers[shard] = (json.loads(get(url, 8, 8 + length - 1)), 8 + length)
        return self.headers[shard]

    def tensor(self, name):
        shard = self.map[name]
        header, start = self.header(shard)
        info = header[name]
        a, b = info["data_offsets"]
        raw = get(f"{BASE}/{shard}", start + a, start + b - 1)
        assert info["dtype"] == "BF16", info["dtype"]
        return bf16_to_f32(raw).reshape(info["shape"])


def mxfp4(weights, block=32):
    """Round-trip through OCP MXFP4: E8M0 shared exponent per 32 weights.

    Validated against ground truth: decoding GPT-OSS-20B's native MXFP4 bytes
    and re-encoding them here reproduces all 518,400 blocks bit-exactly,
    exponents included. So the numbers below are a property of 4-bit, not of
    this implementation. OpenAI evidently used the same max-fitting rule.

    Choosing the exponent to minimise squared error instead — legal, since E8M0
    may hold anything — was measured and buys 0.4 dB. Not enough to matter, and
    not worth diverging from what the format is conventionally written with.
    """
    flat = weights.reshape(-1, block)
    peak = np.abs(flat).max(axis=1, keepdims=True)
    # Shared scale is a power of two chosen so the block's largest value lands
    # on 6.0, the largest code. E8M0 carries the exponent and nothing else.
    # ceil, not floor. With floor the scale comes out below peak/6, so the
    # block's largest weight lands above 6.0 and clips to it — up to 2x low on
    # the very element the scale was chosen to protect.
    exponent = np.where(peak > 0, np.ceil(np.log2(peak / 6.0)), 0)
    exponent = np.clip(exponent, -127, 127)
    scale = np.exp2(exponent)
    scaled = np.abs(flat) / scale
    # Nearest representable magnitude.
    index = np.abs(scaled[..., None] - FP4[None, None, :]).argmin(axis=-1)
    return (np.sign(flat) * FP4[index] * scale).reshape(weights.shape)


def int8_affine(weights, group=64):
    """The trunk's scheme, for comparison: affine per group of 64."""
    flat = weights.reshape(-1, group)
    low, high = flat.min(axis=1, keepdims=True), flat.max(axis=1, keepdims=True)
    scale = np.where(high > low, (high - low) / 255.0, 1.0)
    codes = np.clip(np.round((flat - low) / scale), 0, 255)
    return (codes * scale + low).reshape(weights.shape)


def snr_db(original, approx):
    noise = float(np.sum((original - approx) ** 2))
    if noise == 0:
        return float("inf")
    return 10 * np.log10(float(np.sum(original ** 2)) / noise)


def gemv_error(original, approx, trials=8, seed=0):
    """Relative error of y = W x, which is what the kernel actually computes."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        x = rng.standard_normal(original.shape[1]).astype(np.float32)
        x /= np.linalg.norm(x)
        exact, got = original @ x, approx @ x
        worst = max(worst, float(np.linalg.norm(exact - got) / np.linalg.norm(exact)))
    return worst


def main():
    ckpt = Checkpoint()

    # Two layers apart, and the router, which is the one tensor where a small
    # error changes which experts are chosen rather than by how much.
    names = []
    for layer in (0, 24, 47):
        for kind in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"model.layers.{layer}.mlp.experts.0.{kind}.weight")
    names.append("model.layers.0.mlp.gate.weight")

    print(f"{REPO}\n")
    print(f"{'tensor':<52}{'shape':>14}{'MXFP4':>9}{'int8':>8}{'gemv':>9}")
    print("-" * 92)

    fp4_snrs, i8_snrs, fp4_gemv = [], [], []
    for name in names:
        try:
            w = ckpt.tensor(name)
        except Exception as error:                       # noqa: BLE001
            print(f"{name:<52}  {error}")
            continue
        q4, q8 = mxfp4(w), int8_affine(w)
        s4, s8 = snr_db(w, q4), snr_db(w, q8)
        g4 = gemv_error(w, q4)
        short = name.replace("model.layers.", "L").replace(".weight", "") \
                    .replace(".mlp.experts.0.", ".e0.").replace(".mlp.", ".")
        print(f"{short:<52}{str(w.shape):>14}{s4:>8.1f}dB{s8:>7.1f}dB{g4:>8.2%}")
        if "gate.weight" not in name:
            fp4_snrs.append(s4); i8_snrs.append(s8); fp4_gemv.append(g4)

    print("-" * 92)
    print(f"{'expert mean':<52}{'':>14}{np.mean(fp4_snrs):>8.1f}dB"
          f"{np.mean(i8_snrs):>7.1f}dB{np.mean(fp4_gemv):>8.2%}")
    print(f"\nMXFP4 costs {np.mean(i8_snrs) - np.mean(fp4_snrs):.1f} dB against int8, "
          f"and saves {1 - 4.25 / 8.06:.0%} of the disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
