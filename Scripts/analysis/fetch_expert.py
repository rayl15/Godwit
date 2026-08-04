"""Extract one expert from GPT-OSS-120B and build a ground-truth fixture.

Downloads ~13 MB, not 60 GB: safetensors stores each tensor contiguously, so a
single expert is one byte range per sub-tensor.

Writes a directory holding the expert in Godwit's blob layout plus a reference
computed with NumPy, mirroring transformers' `convert_moe_packed_tensors`
exactly (FP4 lookup, low nibble to even positions, ldexp by scale-127). The
Swift side reads the same blob, runs the Metal kernel, and must agree.

usage: python3 fetch_expert.py <output-dir> [layer] [expert]
"""
import json
import os
import struct
import sys
import urllib.request

import numpy as np

REPO = "openai/gpt-oss-120b"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
PAGE = 16384

FP4_VALUES = np.array([
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def fetch(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def dequantize(blocks, scales):
    """blocks [rows, nblocks, 16] uint8, scales [rows, nblocks] uint8 -> [rows, cols]."""
    lo = FP4_VALUES[blocks & 0x0F]
    hi = FP4_VALUES[blocks >> 4]
    rows, nblocks, _ = blocks.shape
    out = np.empty((rows, nblocks, 32), dtype=np.float32)
    out[:, :, 0::2] = lo
    out[:, :, 1::2] = hi
    out *= np.exp2(scales.astype(np.int32) - 127)[:, :, None]
    return out.reshape(rows, nblocks * 32)


def main():
    out_dir = sys.argv[1]
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    expert = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(out_dir, exist_ok=True)

    print(f"reading index for layer {layer}, expert {expert}")
    weight_map = json.loads(fetch(f"{BASE}/model.safetensors.index.json"))["weight_map"]

    names = {
        "gate_up_blocks": f"model.layers.{layer}.mlp.experts.gate_up_proj_blocks",
        "gate_up_scales": f"model.layers.{layer}.mlp.experts.gate_up_proj_scales",
        "gate_up_bias":   f"model.layers.{layer}.mlp.experts.gate_up_proj_bias",
        "down_blocks":    f"model.layers.{layer}.mlp.experts.down_proj_blocks",
        "down_scales":    f"model.layers.{layer}.mlp.experts.down_proj_scales",
        "down_bias":      f"model.layers.{layer}.mlp.experts.down_proj_bias",
    }

    # Safetensors: 8-byte little-endian header length, then a JSON header, then data.
    headers, data_start = {}, {}
    for shard in sorted({weight_map[n] for n in names.values()}):
        url = f"{BASE}/{shard}"
        n = struct.unpack("<Q", fetch(url, 0, 7))[0]
        headers[shard] = json.loads(fetch(url, 8, 8 + n - 1))
        data_start[shard] = 8 + n

    parts, meta = {}, {}
    for key, name in names.items():
        shard = weight_map[name]
        info = headers[shard][name]
        shape = info["shape"]
        base = data_start[shard] + info["data_offsets"][0]

        # Expert index is the leading dimension, so one expert is one slice.
        per_expert = int(np.prod(shape[1:]))
        itemsize = 2 if info["dtype"] == "BF16" else 1
        span = per_expert * itemsize
        start = base + expert * span

        print(f"  {key:16} {info['dtype']:5} {str(shape):24} -> {span:>10,} bytes")
        parts[key] = fetch(f"{BASE}/{shard}", start, start + span - 1)
        assert len(parts[key]) == span, f"{key}: got {len(parts[key])} want {span}"
        meta[key] = {"shape": shape[1:], "dtype": info["dtype"]}

    # --- Write the blob: sections page-aligned so pread lands on page boundaries ---
    order = ["gate_up_blocks", "gate_up_scales", "down_blocks", "down_scales",
             "gate_up_bias", "down_bias"]
    layout, cursor = {}, 0
    with open(os.path.join(out_dir, "expert.bin"), "wb") as f:
        for key in order:
            pad = (-cursor) % PAGE
            f.write(b"\0" * pad)
            cursor += pad
            layout[key] = {"offset": cursor, "length": len(parts[key]), **meta[key]}
            f.write(parts[key])
            cursor += len(parts[key])
    print(f"\nwrote expert.bin ({cursor:,} bytes)")

    # --- Ground truth: y = dequant(gate_up) @ x ---
    gu_shape = meta["gate_up_blocks"]["shape"]          # [rows, nblocks, 16]
    rows, nblocks = gu_shape[0], gu_shape[1]
    cols = nblocks * 32
    blocks = np.frombuffer(parts["gate_up_blocks"], dtype=np.uint8).reshape(rows, nblocks, 16)
    scales = np.frombuffer(parts["gate_up_scales"], dtype=np.uint8).reshape(rows, nblocks)
    weights = dequantize(blocks, scales)

    rng = np.random.default_rng(20260804)
    x = rng.standard_normal(cols).astype(np.float16)
    y = (weights @ x.astype(np.float32)).astype(np.float32)

    x.tofile(os.path.join(out_dir, "x.f16"))
    y.tofile(os.path.join(out_dir, "y.f32"))

    manifest = {
        "model": REPO, "layer": layer, "expert": expert,
        "rows": rows, "cols": cols, "blockSize": 32, "pageAlignment": PAGE,
        "sections": layout,
        "reference": {"x": "x.f16", "y": "y.f32", "seed": 20260804},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"gate_up dequantised to [{rows}, {cols}]")
    print(f"  weight range [{weights.min():.3f}, {weights.max():.3f}], "
          f"nonzero {np.count_nonzero(weights) / weights.size:.1%}")
    print(f"  y range [{y.min():.3f}, {y.max():.3f}]")
    print(f"\nfixture ready in {out_dir}/")


if __name__ == "__main__":
    main()
