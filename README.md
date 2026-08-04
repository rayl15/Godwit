# Godwit

**Run mixture-of-experts models that do not fit in your machine's memory.**

A bar-tailed godwit flies roughly 13,500 km without landing, without eating, on
a body that weighs about half a pound. Maximum range, minimum payload. That is
the whole idea here.

## The problem

Modern open-weight MoE models are enormous on disk but activate only a small
fraction of their parameters per token:

| Model | Total | Active per token | 4-bit size | Fits in 16 GB? |
| --- | ---: | ---: | ---: | :---: |
| GPT-OSS-120B | 117B | ~5.1B | ~60 GB | no |
| Qwen3-235B-A22B | 235B | ~22B | ~120 GB | no |

The weights you need for any single token are small. The weights you might need
are not. Every mainstream runtime resolves this by requiring all of them to be
resident, so the disk footprint sets the memory requirement.

Godwit does not. It keeps the shared trunk — embeddings, attention, routers,
norms, shared experts — resident, and streams routed experts from NVMe as the
router selects them, against a bounded per-layer slot cache.

The cost is throughput; you are trading tokens per second for the ability to run
the model at all. The bet is that "slow" beats "impossible."

## Status

**Pre-alpha — no inference yet, but the foundations are verified against real
weights.** See [docs/DESIGN.md](docs/DESIGN.md) for the architecture and
[docs/ESTIMATE.md](docs/ESTIMATE.md) for the measured feasibility case.

Verified end to end on an M4: one expert pulled from the real GPT-OSS-120B
checkpoint by HTTP range request, written in Godwit's blob layout, read back
with `pread` into GPU-visible memory, and multiplied by the Metal kernel —
matching a NumPy reference to **7.2e-07** relative error.

```
$ godwit verify-expert scratch/expert-l0-e0
expert  5760 x 2880
read    8.40 MiB in 3.2 ms (2.56 GiB/s)
compute 2.55 ms
error   max 7.209e-07  mean 6.262e-08

PASS — Metal output matches the NumPy reference on real weights
```

What exists:

- `ArchitectureSpec` — model-agnostic transformer description; kernels take
  these as specialisation constants rather than hardcoding a family
- `MXFP4` — reference CPU decoder for the OCP microscaling block format, the
  ground truth every kernel is validated against
- `ExpertCachePlanner` — LFU-with-LRU-tiebreak slot planner, pure logic and
  fully testable without a GPU
- `ExpertBlobReader` — `pread` into page-aligned, zero-copy GPU memory
- Fused MXFP4 dequant-GEMV kernels, benchmarked against affine int4
- **Installer** — streams the checkpoint and repacks it into `.gwt`, never
  materialising a shard. Verified byte-exact against the source.

- **`ExpertRunner`** — a complete expert forward pass on GPU from an installed
  model, verified against NumPy to ~1e-5 across experts 0, 17, 64 and 127

- `Router` — top-4 selection with softmax over the selection only
- `RoutingTrace` — measures real router skew and replays it through the planner
- `RoPE` — YaRN-corrected frequencies, matching transformers to 1e-5
- `Attention` — grouped-query attention with sinks and sliding windows, verified
  against NumPy at 1, 8, 32 and 200 tokens

- **`TransformerLayer`** — a complete layer, attention and MoE joined by
  residuals, verified against NumPy with routing matching exactly
- `KVCache` — FP16, ring-buffered for sliding layers so memory stays bounded

Not started: the loop over 36 layers, tokenizer, sampling.

### Installing

```bash
godwit install --output model.gwt            # ~57 GiB, hours
godwit install --output model.gwt --layers 1 # one layer, for testing the pipeline
python3 Scripts/analysis/verify_install.py model.gwt
```

Experts are copied through as MXFP4 without ever being dequantised; only their
arrangement changes. The trunk arrives as BF16 and is quantised to int8 on the
way to disk, which is lossy by design and measured.

## Requirements

- Apple Silicon Mac, macOS 26+, Swift 6.2+
- Fast NVMe storage. This is not optional — the design is I/O bound by
  construction, and the whole approach collapses on network or spinning storage.

## Build

```bash
swift build
Scripts/test.sh
```

Use `Scripts/test.sh` rather than `swift test` directly. On a machine with only
the Command Line Tools installed, Swift Testing needs framework and rpath flags
that the script supplies; without them the test bundle fails to load.

## Reproducing the verification

Downloads ~13 MB — one expert, not the 60 GB checkpoint. Safetensors stores
tensors contiguously, so a single expert is one byte range per sub-tensor.

```bash
python3 Scripts/analysis/fetch_expert.py scratch/expert-l0-e0
swift build -c release
.build/release/godwit verify-expert scratch/expert-l0-e0
```

## Prior art and acknowledgements

Godwit is an independent, clean-room implementation. It is not a fork.

It was, however, directly inspired by
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) by Andrey Mikhaylov,
which demonstrated that streamed-expert inference is genuinely practical on
Apple Silicon and published an unusually honest record of 103 experiments —
including the failures. Several findings there saved us from repeating dead
ends, and they are credited individually in
[docs/DESIGN.md](docs/DESIGN.md#inherited-findings).

Godwit differs in aim: it targets multiple model families and quantisation
formats rather than one pinned checkpoint, and it treats memory-constrained
execution of very large models as the goal rather than small-machine execution
of a mid-sized one.

## License

[Apache License 2.0](LICENSE).

Model weights are not included and remain governed by their own terms.
