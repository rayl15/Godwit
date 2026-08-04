"""Does requantising GPT-OSS-120B's trunk destroy quality?

The memory budget depends on it. `config.json` deliberately keeps embeddings,
the output head, attention, and the router in BF16 -- excluding exactly those
tensors from MXFP4 -- which leaves a 4.3 GiB resident trunk. Godwit needs 1.52
GiB, and gets there by quantising them ourselves. OpenAI may have had a reason
not to.

Rather than reconstruction error, which is easy to measure and hard to
interpret, this tests the thing that actually matters:

  output head  does the predicted token change? A logit matrix is only as good
               as the ranking it induces, so we compare top-1 and top-5 against
               the BF16 baseline over random hidden states.
  embeddings   do tokens stay distinguishable? Measured as cosine similarity
               between original and requantised rows.
  attention    plain reconstruction SNR, since its output feeds a long chain
               rather than a decision.

Downloads a few hundred MB of sampled rows, not the full checkpoint.
"""
import json
import struct
import sys
import urllib.request

import numpy as np

BASE = "https://huggingface.co/openai/gpt-oss-120b/resolve/main"


def get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def bf16_to_f32(raw):
    """BF16 is the top 16 bits of an FP32, so widening is a shift."""
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


class Checkpoint:
    def __init__(self):
        self.weight_map = json.loads(get(f"{BASE}/model.safetensors.index.json"))["weight_map"]
        self._headers, self._data_start = {}, {}

    def _header(self, shard):
        if shard not in self._headers:
            url = f"{BASE}/{shard}"
            n = struct.unpack("<Q", get(url, 0, 7))[0]
            self._headers[shard] = json.loads(get(url, 8, 8 + n - 1))
            self._data_start[shard] = 8 + n
        return self._headers[shard]

    def info(self, name):
        shard = self.weight_map[name]
        return shard, self._header(shard)[name]

    def rows(self, name, start_row, count):
        """Fetch `count` contiguous rows of a 2-D row-major BF16 tensor."""
        shard, info = self.info(name)
        assert info["dtype"] == "BF16", info["dtype"]
        cols = info["shape"][1]
        base = self._data_start[shard] + info["data_offsets"][0]
        stride = cols * 2
        raw = get(f"{BASE}/{shard}",
                  base + start_row * stride,
                  base + (start_row + count) * stride - 1)
        return bf16_to_f32(raw).reshape(count, cols)


def quantize_affine(w, bits, group=64):
    """MLX-style affine: per group of `group`, scale and zero-point, round-trip."""
    rows, cols = w.shape
    assert cols % group == 0
    levels = (1 << bits) - 1
    g = w.reshape(rows, cols // group, group)
    lo = g.min(axis=2, keepdims=True)
    hi = g.max(axis=2, keepdims=True)
    scale = np.maximum((hi - lo) / levels, 1e-12)
    q = np.clip(np.rint((g - lo) / scale), 0, levels)
    return (q * scale + lo).reshape(rows, cols)


def snr_db(original, approx):
    noise = np.sum((original - approx) ** 2)
    return 10 * np.log10(np.sum(original ** 2) / max(noise, 1e-30))


def sample_rows(ckpt, name, chunks=4, per_chunk=2500):
    """Spread samples across the vocabulary; low token ids are unrepresentative."""
    _, info = ckpt.info(name)
    total = info["shape"][0]
    step = total // chunks
    parts = [ckpt.rows(name, i * step, per_chunk) for i in range(chunks)]
    return np.concatenate(parts, axis=0)


def main():
    ckpt = Checkpoint()
    rng = np.random.default_rng(20260804)

    print("=== output head: does the predicted token change? ===")
    head = sample_rows(ckpt, "lm_head.weight")
    print(f"sampled {head.shape[0]:,} of 201,088 rows, {head.shape[1]} cols\n")

    # Hidden states scaled to a realistic magnitude; only the ranking matters.
    hidden = rng.standard_normal((512, head.shape[1])).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    baseline = hidden @ head.T
    base_top1 = baseline.argmax(axis=1)
    base_top5 = np.argsort(-baseline, axis=1)[:, :5]

    print(f"{'scheme':<18}{'SNR':>9}{'top-1 kept':>13}{'top-5 overlap':>16}")
    for bits in (8, 4):
        approx = quantize_affine(head, bits)
        logits = hidden @ approx.T
        top1 = logits.argmax(axis=1)
        top5 = np.argsort(-logits, axis=1)[:, :5]
        overlap = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(base_top5, top5)])
        print(f"{f'affine int{bits}':<18}{snr_db(head, approx):>8.1f}dB"
              f"{np.mean(top1 == base_top1):>12.1%}{overlap:>16.1%}")

    print("\n=== embeddings: do tokens stay distinguishable? ===")
    embed = sample_rows(ckpt, "model.embed_tokens.weight")
    print(f"{'scheme':<18}{'SNR':>9}{'mean cos':>12}{'worst cos':>12}")
    for bits in (8, 4):
        approx = quantize_affine(embed, bits)
        cos = np.sum(embed * approx, axis=1) / (
            np.linalg.norm(embed, axis=1) * np.linalg.norm(approx, axis=1) + 1e-12)
        print(f"{f'affine int{bits}':<18}{snr_db(embed, approx):>8.1f}dB"
              f"{cos.mean():>12.4f}{cos.min():>12.4f}")

    print("\n=== attention (layer 0) ===")
    print(f"{'tensor':<12}{'shape':>16}{'int8 SNR':>12}{'int4 SNR':>12}")
    for short, name in [("q_proj", "model.layers.0.self_attn.q_proj.weight"),
                        ("o_proj", "model.layers.0.self_attn.o_proj.weight")]:
        _, info = ckpt.info(name)
        w = ckpt.rows(name, 0, info["shape"][0])
        print(f"{short:<12}{str(list(w.shape)):>16}"
              f"{snr_db(w, quantize_affine(w, 8)):>11.1f}dB"
              f"{snr_db(w, quantize_affine(w, 4)):>11.1f}dB")

    print("""
Reading the numbers: the head is the decision-making tensor, so top-1 agreement
is the figure that matters. Embedding cosine below ~0.99 means tokens are
blurring into each other. SNR alone does not tell you whether generation
survives.""")


if __name__ == "__main__":
    main()
