#!/usr/bin/env python3
"""Check an installed Qwen3 against the checkpoint it came from.

`verify_install.py` does this for GPT-OSS, where the experts were copied
through as MXFP4 and must match bit for bit. Qwen3's were quantised on the way
in, so the test is different: the weights cannot match exactly, and the useful
question is whether they are off by exactly as much as MXFP4 costs and no more.

Three ways to be wrong that this is built to separate:

  quantisation   ~18.5 dB, expected, and the same for every expert
  interleaving   gate and up swapped — still ~18 dB per row, but the rows
                 are in the wrong places, so the control below collapses
  transposition  a matrix stored the wrong way round reads as noise

    python3 Scripts/analysis/verify_qwen3.py scratch/qwen3.gwt
"""

import json
import pathlib
import struct
import sys
import urllib.request

import numpy as np

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
               dtype=np.float32)


def get(url, start=None, end=None):
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(request) as response:
        return response.read()


class Source:
    def __init__(self, repo, revision="main"):
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}"
        self.map = json.loads(
            get(f"{self.base}/model.safetensors.index.json"))["weight_map"]
        self.headers = {}

    def tensor(self, name):
        shard = self.map[name]
        if shard not in self.headers:
            length = struct.unpack("<Q", get(f"{self.base}/{shard}", 0, 7))[0]
            self.headers[shard] = (
                json.loads(get(f"{self.base}/{shard}", 8, 8 + length - 1)), 8 + length)
        header, start = self.headers[shard]
        info = header[name]
        a, b = info["data_offsets"]
        raw = get(f"{self.base}/{shard}", start + a, start + b - 1)
        half = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return np.frombuffer((half << 16).tobytes(), dtype=np.float32).reshape(info["shape"])


class Installed:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            # Written last, so its absence usually means the install is still
            # running rather than that anything is wrong.
            done = len(list((self.root / "experts").glob("*.bin"))) \
                if (self.root / "experts").exists() else 0
            raise SystemExit(
                f"no manifest in {root} — install incomplete "
                f"({done} layer files so far)")
        self.manifest = json.loads(manifest.read_text())

    def section(self, layer, expert, name):
        span = self.manifest["expertSections"][name]
        offset = expert * self.manifest["expertStride"] + span["offset"]
        path = self.root / "experts" / f"layer_{layer:02d}.bin"
        with open(path, "rb") as handle:
            handle.seek(offset)
            return handle.read(span["length"])

    def dequantise(self, layer, expert, blocks, scales, rows, cols):
        packed = np.frombuffer(self.section(layer, expert, blocks), dtype=np.uint8)
        scale = np.frombuffer(self.section(layer, expert, scales), dtype=np.uint8)
        per_row = cols // 32
        packed = packed.reshape(rows, per_row, 16)
        scale = scale.reshape(rows, per_row)
        out = np.empty((rows, per_row, 32), dtype=np.float32)
        out[:, :, 0::2] = FP4[packed & 0x0F]
        out[:, :, 1::2] = FP4[packed >> 4]
        out *= np.exp2(scale.astype(np.int32) - 127)[:, :, None]
        return out.reshape(rows, cols)


def snr_db(original, approx):
    noise = float(np.sum((original - approx) ** 2))
    if noise == 0:
        return float("inf")
    return 10 * np.log10(float(np.sum(original ** 2)) / noise)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen3.gwt"
    installed = Installed(root)
    manifest = installed.manifest
    spec = manifest["spec"]
    hidden, inner = spec["hiddenSize"], spec["intermediateSize"]
    layers, experts = manifest["layerCount"], manifest["expertCount"]

    print(f"{manifest['model']} at {root}")
    print(f"  {layers} layers x {experts} experts, "
          f"hidden {hidden}, expert inner {inner}")
    print(f"  qk-norm {spec.get('queryKeyNorm')}, sinks {spec.get('attentionSinks')}, "
          f"activation {spec.get('activation')}\n")

    source = Source(manifest["model"], manifest.get("revision", "main"))

    # Spread across depth: an installer that loses its place shows up as one
    # region being fine and another being noise.
    picks = [(0, 0), (0, experts - 1), (layers // 2, 7), (layers - 1, experts - 1)]
    failures = 0

    print(f"{'layer':>6}{'expert':>8}{'gate_up':>10}{'down':>9}{'swapped':>10}")
    print("-" * 45)
    for layer, expert in picks:
        gate = source.tensor(f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight")
        up = source.tensor(f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight")
        down = source.tensor(f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight")

        want = np.empty((2 * inner, hidden), dtype=np.float32)
        want[0::2], want[1::2] = gate, up
        got = installed.dequantise(layer, expert, "gate_up_blocks", "gate_up_scales",
                                   2 * inner, hidden)
        got_down = installed.dequantise(layer, expert, "down_blocks", "down_scales",
                                        hidden, inner)

        # The control: if gate and up were interleaved the wrong way round the
        # per-weight error would still look reasonable, so compare against the
        # swapped arrangement and demand it be much worse.
        swapped = np.empty_like(want)
        swapped[0::2], swapped[1::2] = up, gate

        a, b, c = snr_db(want, got), snr_db(down, got_down), snr_db(swapped, got)
        flag = "" if (a > 15 and b > 15 and c < a - 10) else "  <-- WRONG"
        if flag:
            failures += 1
        print(f"{layer:>6}{expert:>8}{a:>9.1f}dB{b:>8.1f}dB{c:>9.1f}dB{flag}")

    print("-" * 45)
    if failures:
        print(f"\n{failures} of {len(picks)} sampled experts are wrong.")
        return 1
    print("\nEvery sampled expert is off by exactly what MXFP4 costs, and the")
    print("gate/up ordering is right rather than coincidentally plausible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
