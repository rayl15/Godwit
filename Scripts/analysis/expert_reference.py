"""Reference implementation of one GPT-OSS expert, computed from a .gwt install.

Mirrors transformers' GptOssExperts.forward. Three details matter and none are
guessable from config.json:

  gate and up are interleaved   gate = gate_up[0::2], up = gate_up[1::2]
  the activation is not SiLU    (up + 1) * gate * sigmoid(1.702 * gate)
  clamping is asymmetric        gate above only, up on both sides

Reads the installed blob rather than the checkpoint, so it also confirms the
repacked layout is addressable exactly as the manifest describes.

usage: python3 expert_reference.py <gwt-dir> <layer> <expert> <out-dir>
"""
import json
import os
import sys

import numpy as np

FP4 = np.array([+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
ALPHA = 1.702
LIMIT = 7.0


def read_section(path, manifest, expert, name):
    span = manifest["expertSections"][name]
    offset = expert * manifest["expertStride"] + span["offset"]
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(span["length"])


def dequantize(blocks_raw, scales_raw, rows, cols):
    nblocks = cols // 32
    blocks = np.frombuffer(blocks_raw, dtype=np.uint8).reshape(rows, nblocks, 16)
    scales = np.frombuffer(scales_raw, dtype=np.uint8).reshape(rows, nblocks)
    out = np.empty((rows, nblocks, 32), dtype=np.float32)
    out[:, :, 0::2] = FP4[blocks & 0x0F]
    out[:, :, 1::2] = FP4[blocks >> 4]
    out *= np.exp2(scales.astype(np.int32) - 127)[:, :, None]
    return out.reshape(rows, cols)


def bf16(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def main():
    root, layer, expert, out_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)
    manifest = json.load(open(f"{root}/manifest.json"))
    spec = manifest["spec"]
    H, F = spec["hiddenSize"], spec["intermediateSize"]
    path = f"{root}/experts/layer_{layer:02d}.bin"

    gu_w = dequantize(read_section(path, manifest, expert, "gate_up_blocks"),
                      read_section(path, manifest, expert, "gate_up_scales"), 2 * F, H)
    gu_b = bf16(read_section(path, manifest, expert, "gate_up_bias"))
    dn_w = dequantize(read_section(path, manifest, expert, "down_blocks"),
                      read_section(path, manifest, expert, "down_scales"), H, F)
    dn_b = bf16(read_section(path, manifest, expert, "down_bias"))

    rng = np.random.default_rng(4080)
    x = (rng.standard_normal(H) * 0.5).astype(np.float16)

    gate_up = gu_w @ x.astype(np.float32) + gu_b
    gate, up = gate_up[0::2], gate_up[1::2]
    gate = np.minimum(gate, LIMIT)                 # no lower bound, on purpose
    up = np.clip(up, -LIMIT, LIMIT)
    glu = gate / (1.0 + np.exp(-ALPHA * gate)) * 1.0
    activated = ((up + 1.0) * glu).astype(np.float16)
    y = dn_w @ activated.astype(np.float32) + dn_b

    x.tofile(f"{out_dir}/x.f16")
    y.tofile(f"{out_dir}/y.f32")
    json.dump({"layer": layer, "expert": expert, "hiddenSize": H, "intermediateSize": F},
              open(f"{out_dir}/case.json", "w"), indent=2)

    print(f"layer {layer} expert {expert}")
    print(f"  gate_up  [{gu_w.shape[0]}, {gu_w.shape[1]}]  "
          f"range [{gu_w.min():.3f}, {gu_w.max():.3f}]")
    print(f"  pre-act  range [{gate_up.min():.3f}, {gate_up.max():.3f}]  "
          f"clamped {np.mean(gate_up[0::2] > LIMIT):.2%} of gate")
    print(f"  output   range [{y.min():.3f}, {y.max():.3f}]  |y| mean {np.abs(y).mean():.4f}")
    print(f"\nreference written to {out_dir}/")


if __name__ == "__main__":
    main()
