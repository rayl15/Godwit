"""Check a .gwt installation byte-for-byte against the source checkpoint.

Repacking is only useful if it is lossless where it claims to be. Expert MXFP4
data is copied through untouched, so it must match the source exactly; the
trunk is deliberately quantised, so it is checked for accuracy instead.

Re-downloads a few slices to compare against, so it is independent of whatever
the installer believed it wrote.

usage: python3 verify_install.py <gwt-dir>
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


def bf16(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class Source:
    def __init__(self):
        self.map = json.loads(get(f"{BASE}/model.safetensors.index.json"))["weight_map"]
        self._h, self._start = {}, {}

    def header(self, shard):
        if shard not in self._h:
            url = f"{BASE}/{shard}"
            n = struct.unpack("<Q", get(url, 0, 7))[0]
            self._h[shard] = json.loads(get(url, 8, 8 + n - 1))
            self._start[shard] = 8 + n
        return self._h[shard]

    def slice(self, name, index):
        """One slice along the leading dimension."""
        shard = self.map[name]
        info = self.header(shard)[name]
        item = 2 if info["dtype"] == "BF16" else 1
        span = int(np.prod(info["shape"][1:])) * item
        base = self._start[shard] + info["data_offsets"][0] + index * span
        return get(f"{BASE}/{shard}", base, base + span - 1)

    def rows(self, name, start_row, count):
        shard = self.map[name]
        info = self.header(shard)[name]
        stride = info["shape"][1] * 2
        base = self._start[shard] + info["data_offsets"][0]
        raw = get(f"{BASE}/{shard}", base + start_row * stride,
                  base + (start_row + count) * stride - 1)
        return bf16(raw).reshape(count, info["shape"][1])


def main():
    root = sys.argv[1]
    manifest = json.load(open(f"{root}/manifest.json"))
    src = Source()
    failures = 0

    print(f"format v{manifest['formatVersion']}  {manifest['layerCount']} layers  "
          f"stride {manifest['expertStride']:,}\n")

    # --- Experts must be byte-identical: they are copied, not converted ---
    print("=== experts (must be byte-exact) ===")
    sections = {
        "gate_up_blocks": "gate_up_proj_blocks",
        "gate_up_scales": "gate_up_proj_scales",
        "down_blocks": "down_proj_blocks",
        "down_scales": "down_proj_scales",
        "gate_up_bias": "gate_up_proj_bias",
        "down_bias": "down_proj_bias",
    }
    layer = 0
    path = f"{root}/experts/layer_{layer:02d}.bin"
    with open(path, "rb") as f:
        # First, middle, and last: catches an off-by-one in the stride that a
        # single sample would miss.
        for expert in (0, 64, manifest["expertCount"] - 1):
            for key, suffix in sections.items():
                span = manifest["expertSections"][key]
                f.seek(expert * manifest["expertStride"] + span["offset"])
                got = f.read(span["length"])
                want = src.slice(f"model.layers.{layer}.mlp.experts.{suffix}", expert)
                ok = got == want
                failures += not ok
                if not ok or key == "gate_up_blocks":
                    print(f"  expert {expert:>3} {key:16} "
                          f"{'exact' if ok else 'MISMATCH'} ({len(got):,} bytes)")

    # --- Trunk is quantised, so check accuracy rather than equality ---
    print("\n=== trunk (int8 affine, lossy by design) ===")
    by_name = {s["name"]: s for s in manifest["trunkSections"]}
    with open(f"{root}/trunk.bin", "rb") as f:
        for name, tensor in [("embed", "model.embed_tokens.weight"),
                             ("head", "lm_head.weight"),
                             ("layer0.q_proj", "model.layers.0.self_attn.q_proj.weight")]:
            spec = by_name[name]
            rows, cols = spec["shape"]
            groups = cols // 64
            sample = 256

            f.seek(spec["offset"])
            codes = np.frombuffer(f.read(sample * cols), dtype=np.uint8).reshape(sample, cols)
            f.seek(spec["offset"] + rows * cols)
            meta = np.frombuffer(f.read(sample * groups * 4), dtype=np.uint16)
            meta = (meta.astype(np.uint32) << 16).view(np.float32).reshape(sample, groups, 2)

            restored = (codes.reshape(sample, groups, 64) * meta[:, :, 0:1]
                        + meta[:, :, 1:2]).reshape(sample, cols)
            original = src.rows(tensor, 0, sample)

            err = np.abs(restored - original)
            snr = 10 * np.log10(np.sum(original ** 2) / max(np.sum((original - restored) ** 2), 1e-30))
            ok = snr > 35
            failures += not ok
            print(f"  {name:16} SNR {snr:5.1f} dB  max abs err {err.max():.2e}  "
                  f"{'ok' if ok else 'TOO LOSSY'}")

    print("\nPASS — installation matches the checkpoint" if not failures
          else f"\nFAIL — {failures} check(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
