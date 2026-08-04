"""Reference for one complete transformer layer: attention + MoE with residuals.

Composes the attention and expert references, which are each verified
separately, so a mismatch here points at the residual wiring or the routing
rather than at either block.

usage: python3 layer_reference.py <gwt-dir> <layer> <tokens> <out-dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_reference import Trunk, yarn_inv_freq, rms_norm, bf16
from expert_reference import dequantize, read_section, FP4, ALPHA, LIMIT


def main():
    root, layer, tokens, out_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)
    trunk = Trunk(root)
    manifest, spec = trunk.manifest, trunk.manifest["spec"]

    H = spec["hiddenSize"]; F = spec["intermediateSize"]
    heads, kv_heads, hd = spec["attentionHeads"], spec["keyValueHeads"], spec["headDimension"]
    top_k = spec["layers"][layer]["expertsPerToken"]
    window = spec["layers"][layer]["window"] if spec["layers"][layer]["attention"] == "sliding" else 0
    eps = spec["rmsNormEpsilon"]

    wq, wk, wv, wo = (trunk.int8(f"layer{layer}.{p}_proj") for p in "qkvo")
    bq, bk, bv, bo = (trunk.bf16(f"layer{layer}.{p}_bias") for p in "qkvo")
    sinks = trunk.bf16(f"layer{layer}.sinks")
    in_norm, post_norm = trunk.bf16(f"layer{layer}.input_norm"), trunk.bf16(f"layer{layer}.post_norm")
    rw, rb = trunk.bf16(f"layer{layer}.router_w").reshape(-1, H), trunk.bf16(f"layer{layer}.router_b")

    fp16 = lambda a: a.astype(np.float16).astype(np.float32)
    rng = np.random.default_rng(31337)
    initial = fp16(rng.standard_normal((tokens, H)) * 0.5)
    stream = initial.copy()

    # --- attention ---
    normed = fp16(rms_norm(stream, in_norm, eps))
    q = fp16(normed @ wq.T + bq).reshape(tokens, heads, hd)
    k = fp16(normed @ wk.T + bk).reshape(tokens, kv_heads, hd)
    v = fp16(normed @ wv.T + bv).reshape(tokens, kv_heads, hd)
    inv, af = yarn_inv_freq(hd, spec["ropeTheta"], 32.0, 4096, 32.0, 1.0)
    ang = np.arange(tokens)[:, None] * inv[None, :]
    cos, sin = np.cos(ang) * af, np.sin(ang) * af
    def rot(x):
        a, b = x[..., :hd // 2], x[..., hd // 2:]
        return np.concatenate([a * cos[:, None, :] - b * sin[:, None, :],
                               b * cos[:, None, :] + a * sin[:, None, :]], -1)
    q, k = fp16(rot(q)), fp16(rot(k))

    groups = heads // kv_heads
    attn = np.zeros((tokens, heads, hd), np.float32)
    for t in range(tokens):
        for h in range(heads):
            kv = h // groups
            lo = 0 if window == 0 else max(0, t - window + 1)
            keys = np.arange(lo, t + 1)
            lg = (k[keys, kv] @ q[t, h]) * (hd ** -0.5)
            c = np.concatenate([lg, [sinks[h]]]); c -= c.max()
            p = np.exp(c) / np.exp(c).sum()
            attn[t, h] = p[:-1] @ v[keys, kv]
    stream = fp16(stream + fp16(fp16(attn.reshape(tokens, heads * hd)) @ wo.T + bo))

    # --- mixture of experts ---
    normed = fp16(rms_norm(stream, post_norm, eps))
    path = f"{root}/experts/layer_{layer:02d}.bin"
    cache = {}
    routing = []
    for t in range(tokens):
        x = normed[t]
        logits = rw @ x + rb
        chosen = np.argsort(-logits)[:top_k]
        top = logits[chosen]
        w = np.exp(top - top.max()); w /= w.sum()
        routing.append([int(c) for c in chosen])

        combined = np.zeros(H, np.float32)
        for pos, e in enumerate(chosen):
            if e not in cache:
                cache[e] = (
                    dequantize(read_section(path, manifest, e, "gate_up_blocks"),
                               read_section(path, manifest, e, "gate_up_scales"), 2 * F, H),
                    bf16(read_section(path, manifest, e, "gate_up_bias")),
                    dequantize(read_section(path, manifest, e, "down_blocks"),
                               read_section(path, manifest, e, "down_scales"), H, F),
                    bf16(read_section(path, manifest, e, "down_bias")))
            gw, gb, dw, db = cache[e]
            gu = gw @ x + gb
            gate, up = np.minimum(gu[0::2], LIMIT), np.clip(gu[1::2], -LIMIT, LIMIT)
            act = fp16((up + 1.0) * (gate / (1.0 + np.exp(-ALPHA * gate))))
            combined += (dw @ act + db) * w[pos]
        stream[t] = fp16(stream[t] + combined)

    np.asarray(stream, np.float16).tofile(f"{out_dir}/y.f16")
    initial.astype(np.float16).tofile(f"{out_dir}/hidden.f16")
    json.dump({"layer": layer, "tokens": tokens, "hiddenSize": H,
               "routing": routing}, open(f"{out_dir}/case.json", "w"), indent=2)

    print(f"layer {layer}, {tokens} tokens")
    print(f"  routing token 0: {routing[0]}")
    print(f"  output range [{stream.min():.3f}, {stream.max():.3f}], RMS {np.sqrt((stream**2).mean()):.4f}")
    print(f"\nreference written to {out_dir}/")


if __name__ == "__main__":
    main()
