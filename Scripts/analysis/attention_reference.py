"""Reference implementation of one GPT-OSS attention block, from a .gwt install.

Mirrors transformers' GptOssAttention plus eager_attention_forward. Four details
matter and none are guessable from config.json alone:

  RoPE is NeoX half-split       halves rotate against each other, not adjacent pairs
  RoPE frequencies use YaRN     factor 32, corrected between beta_fast and beta_slow
  q/k/v/o all carry biases      attention_bias is true, unusually
  sinks join the softmax        a per-head logit appended, then dropped

usage: python3 attention_reference.py <gwt-dir> <layer> <tokens> <out-dir>
"""
import json
import math
import os
import sys

import numpy as np


def bf16(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class Trunk:
    def __init__(self, root):
        self.manifest = json.load(open(f"{root}/manifest.json"))
        self.sections = {s["name"]: s for s in self.manifest["trunkSections"]}
        self.file = open(f"{root}/trunk.bin", "rb")

    def raw(self, name):
        s = self.sections[name]
        self.file.seek(s["offset"])
        return self.file.read(s["length"]), s

    def bf16(self, name):
        raw, _ = self.raw(name)
        return bf16(raw)

    def int8(self, name):
        """Dequantises an affine int8 tensor: codes then BF16 scale/zero pairs."""
        raw, spec = self.raw(name)
        rows, cols = spec["shape"]
        groups = cols // 64
        codes = np.frombuffer(raw[:rows * cols], dtype=np.uint8).reshape(rows, groups, 64)
        meta = bf16(raw[rows * cols:rows * cols + rows * groups * 4]).reshape(rows, groups, 2)
        return (codes * meta[:, :, 0:1] + meta[:, :, 1:2]).reshape(rows, cols)


def yarn_inv_freq(dim, base, factor, orig, beta_fast, beta_slow):
    pos = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    extrapolation, interpolation = 1.0 / pos, 1.0 / (factor * pos)

    def correction(rotations):
        return (dim * math.log(orig / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction(beta_fast)), 0)
    high = min(math.ceil(correction(beta_slow)), dim - 1)
    ramp = np.clip((np.arange(dim // 2) - low) / max(high - low, 1e-3), 0, 1)
    weight = 1 - ramp
    inv = interpolation * (1 - weight) + extrapolation * weight
    return inv, 0.1 * math.log(factor) + 1.0


def rms_norm(x, weight, eps):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * weight


def main():
    root, layer, tokens, out_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)
    trunk = Trunk(root)
    spec = trunk.manifest["spec"]

    H = spec["hiddenSize"]
    heads, kv_heads, head_dim = spec["attentionHeads"], spec["keyValueHeads"], spec["headDimension"]
    sliding = spec["layers"][layer]["attention"] == "sliding"
    window = spec["layers"][layer]["window"] if sliding else 0

    wq, wk, wv, wo = (trunk.int8(f"layer{layer}.{p}_proj") for p in "qkvo")
    bq, bk, bv, bo = (trunk.bf16(f"layer{layer}.{p}_bias") for p in "qkvo")
    sinks = trunk.bf16(f"layer{layer}.sinks")
    norm_w = trunk.bf16(f"layer{layer}.input_norm")

    rng = np.random.default_rng(90210)
    hidden = (rng.standard_normal((tokens, H)) * 0.5).astype(np.float32)
    normed = rms_norm(hidden, norm_w, spec["rmsNormEpsilon"]).astype(np.float16)

    # The runtime holds q/k/v at FP16 -- k and v because that is the KV cache
    # format the memory budget assumes, q for symmetry with them. The reference
    # rounds at the same points so this test measures kernel correctness rather
    # than re-measuring FP16. The precision cost itself is reported separately
    # below, so it stays visible instead of being absorbed into a tolerance.
    def fp16(a):
        return a.astype(np.float16).astype(np.float32)

    q = fp16(normed.astype(np.float32) @ wq.T + bq).reshape(tokens, heads, head_dim)
    k = fp16(normed.astype(np.float32) @ wk.T + bk).reshape(tokens, kv_heads, head_dim)
    v = fp16(normed.astype(np.float32) @ wv.T + bv).reshape(tokens, kv_heads, head_dim)

    inv_freq, att_factor = yarn_inv_freq(head_dim, spec["ropeTheta"], 32.0, 4096, 32.0, 1.0)
    positions = np.arange(tokens)[:, None]
    angles = positions * inv_freq[None, :]
    cos, sin = np.cos(angles) * att_factor, np.sin(angles) * att_factor

    def rotate(x):                                  # NeoX: halves, not pairs
        first, second = x[..., :head_dim // 2], x[..., head_dim // 2:]
        return np.concatenate([first * cos[:, None, :] - second * sin[:, None, :],
                               second * cos[:, None, :] + first * sin[:, None, :]], axis=-1)

    q, k = fp16(rotate(q)), fp16(rotate(k))         # v is not rotated

    scale = head_dim ** -0.5
    groups = heads // kv_heads
    out = np.zeros((tokens, heads, head_dim), dtype=np.float32)
    for t in range(tokens):
        for h in range(heads):
            kv = h // groups
            lo = 0 if window == 0 else max(0, t - window + 1)
            keys = np.arange(lo, t + 1)
            logits = (k[keys, kv] @ q[t, h]) * scale
            # The sink is appended, softmaxed with the rest, then dropped: mass
            # that attends to nothing.
            combined = np.concatenate([logits, [sinks[h]]])
            combined = combined - combined.max()
            probs = np.exp(combined) / np.exp(combined).sum()
            out[t, h] = probs[:-1] @ v[keys, kv]

    # The runtime's attention output is FP16, because it feeds a residual add
    # that is FP16 too. Rounding here as well makes the comparison a test of the
    # kernel rather than a rediscovery of half precision.
    result = fp16(fp16(out.reshape(tokens, heads * head_dim)) @ wo.T + bo)

    normed.tofile(f"{out_dir}/hidden.f16")
    result.astype(np.float32).tofile(f"{out_dir}/y.f32")
    json.dump({"layer": layer, "tokens": tokens, "hiddenSize": H,
               "sliding": sliding, "window": window},
              open(f"{out_dir}/case.json", "w"), indent=2)

    print(f"layer {layer}, {tokens} tokens, {'sliding ' + str(window) if sliding else 'full'} attention")
    print(f"  sinks    range [{sinks.min():.3f}, {sinks.max():.3f}]")
    print(f"  output   range [{result.min():.3f}, {result.max():.3f}]  "
          f"|y| mean {np.abs(result).mean():.4f}, RMS {np.sqrt((result ** 2).mean()):.4f}")
    print(f"\nreference written to {out_dir}/")


if __name__ == "__main__":
    main()
