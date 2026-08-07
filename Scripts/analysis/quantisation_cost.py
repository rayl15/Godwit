#!/usr/bin/env python3
"""What does our quantisation actually cost Qwen3's output?

The repo knows what MXFP4 costs in weight space — 18.5 dB against the
checkpoint — and it knows the model produces coherent text. It has never
measured the thing in between: how much the numbers the model computes move
because of quantisation.

This measures it directly, in output space, on real inputs. For a routing
decision taken from a real generation, it computes the mixture-of-experts block
twice from the same vector: once with the weights on disk, once with the
original BF16 weights fetched from the checkpoint. Same arithmetic, same
routing, same input — the only difference is the weights.

GPT-OSS needs no equivalent: it ships on the MXFP4 grid, so the installer
copies its experts through and quantisation costs it nothing. Qwen3 ships BF16
and we quantise it ourselves, so it is the one that can be damaged.

    python3 Scripts/analysis/quantisation_cost.py scratch/qwen3.gwt \\
        /tmp/hidden_q3.bin.router /tmp/routing_q3.jsonl
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


class Checkpoint:
    """The original BF16 weights, by HTTP range request."""

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
                json.loads(get(f"{self.base}/{shard}", 8, 8 + length - 1)),
                8 + length)
        header, start = self.headers[shard]
        info = header[name]
        a, b = info["data_offsets"]
        raw = get(f"{self.base}/{shard}", start + a, start + b - 1)
        half = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return np.frombuffer((half << 16).tobytes(),
                             dtype=np.float32).reshape(info["shape"])


class Install:
    """The quantised weights on disk."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        spec = self.manifest["spec"]
        self.hidden = spec["hiddenSize"]
        self.inner = spec["intermediateSize"]

    def _section(self, layer, expert, name):
        span = self.manifest["expertSections"][name]
        offset = expert * self.manifest["expertStride"] + span["offset"]
        with open(self.root / "experts" / f"layer_{layer:02d}.bin", "rb") as f:
            f.seek(offset)
            return f.read(span["length"])

    def _dequant(self, layer, expert, blocks, scales, rows, cols):
        packed = np.frombuffer(self._section(layer, expert, blocks),
                               dtype=np.uint8).reshape(rows, cols // 32, 16)
        scale = np.frombuffer(self._section(layer, expert, scales),
                              dtype=np.uint8).reshape(rows, cols // 32)
        out = np.empty((rows, cols // 32, 32), dtype=np.float32)
        out[:, :, 0::2] = FP4[packed & 0x0F]
        out[:, :, 1::2] = FP4[packed >> 4]
        out *= np.exp2(scale.astype(np.int32) - 127)[:, :, None]
        return out.reshape(rows, cols)

    def expert(self, layer, index):
        """gate, up, down as the runtime sees them."""
        gate_up = self._dequant(layer, index, "gate_up_blocks", "gate_up_scales",
                                2 * self.inner, self.hidden)
        down = self._dequant(layer, index, "down_blocks", "down_scales",
                             self.hidden, self.inner)
        return gate_up[0::2], gate_up[1::2], down


def swiglu_expert(x, gate_w, up_w, down_w):
    gate = gate_w @ x
    up = up_w @ x
    return down_w @ (gate / (1.0 + np.exp(-gate)) * up)


def moe_block(x, experts, weights, fetch):
    """Weighted sum over the selected experts, as the layer computes it."""
    total = np.zeros(down_width(fetch, experts[0]), dtype=np.float32)
    for index, weight in zip(experts, weights):
        gate_w, up_w, down_w = fetch(index)
        total += weight * swiglu_expert(x, gate_w, up_w, down_w)
    return total


def down_width(fetch, first):
    return fetch(first)[2].shape[0]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen3.gwt"
    router_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hidden_q3.bin.router"
    trace_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/routing_q3.jsonl"

    install = Install(root)
    source = Checkpoint(install.manifest["model"],
                        install.manifest.get("revision", "main"))
    width = install.hidden

    decisions = [json.loads(l) for l in
                 pathlib.Path(trace_path).read_text().splitlines() if l.strip()]
    inputs = np.fromfile(router_path, dtype=np.float32).reshape(-1, width)
    assert len(inputs) == len(decisions)

    layers = install.manifest["layerCount"]
    # Spread across depth: error may compound differently early and late.
    picks = [0, layers // 4, layers // 2, 3 * layers // 4, layers - 1]

    print(f"{install.manifest['model']} — MoE block output, quantised vs BF16\n")
    print(f"{'layer':>6}{'rel. error':>13}{'cosine':>10}{'‖y‖ ours':>11}"
          f"{'‖y‖ ref':>10}")
    print("-" * 50)

    errors = []
    for layer in picks:
        # First decision at this layer in the trace.
        i = next(k for k, d in enumerate(decisions) if d["layer"] == layer)
        decision, x = decisions[i], inputs[i]

        cache_ours, cache_ref = {}, {}

        def ours(index, layer=layer, cache=cache_ours):
            if index not in cache:
                cache[index] = install.expert(layer, index)
            return cache[index]

        def ref(index, layer=layer, cache=cache_ref):
            if index not in cache:
                p = f"model.layers.{layer}.mlp.experts.{index}"
                cache[index] = (source.tensor(f"{p}.gate_proj.weight"),
                                source.tensor(f"{p}.up_proj.weight"),
                                source.tensor(f"{p}.down_proj.weight"))
            return cache[index]

        y_ours = moe_block(x, decision["experts"], decision["weights"], ours)
        y_ref = moe_block(x, decision["experts"], decision["weights"], ref)

        rel = float(np.linalg.norm(y_ours - y_ref) / np.linalg.norm(y_ref))
        cos = float(y_ours @ y_ref /
                    (np.linalg.norm(y_ours) * np.linalg.norm(y_ref)))
        errors.append(rel)
        print(f"{layer:>6}{rel:>12.1%}{cos:>10.4f}"
              f"{np.linalg.norm(y_ours):>11.3f}{np.linalg.norm(y_ref):>10.3f}")

    print("-" * 50)
    mean = float(np.mean(errors))
    print(f"{'mean':>6}{mean:>12.1%}\n")

    print("Relative error is on the MoE block's own output, which is then added")
    print("to a residual stream carrying everything computed so far — so the")
    print("effect on the layer's output is smaller than this, and the effect on")
    print("the final logits smaller still. Cosine near 1.0 means the direction")
    print("survives and only the magnitude moves, which a following norm")
    print("largely absorbs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
