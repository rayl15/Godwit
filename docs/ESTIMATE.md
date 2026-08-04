# Feasibility estimate: GPT-OSS-120B on a 16 GB M4

Answers open question #2 in [DESIGN.md](DESIGN.md): what throughput does a 30:1
stream-to-resident ratio actually produce?

**Verdict: decode is viable at 3-5 tok/s. Prefill costs a full pass over the
routed pool per chunk, which makes chunk size the single most important
parameter in the system.**

Done before writing kernels, deliberately — the answer could have killed the
project and it cost an afternoon.

## Inputs

Model geometry from
[`openai/gpt-oss-120b/config.json`](https://huggingface.co/openai/gpt-oss-120b):
36 layers, 128 experts, **top-4** (not top-8), hidden 2880, intermediate 2880,
64 Q heads / 8 KV heads / head_dim 64, vocab 201088, sliding window **128**,
alternating sliding and full attention.

Storage measured on a 13" M4 MacBook Air, internal NVMe, using `pread` at
12.6 MiB random blob-aligned offsets with `F_NOCACHE` and `F_RDAHEAD` disabled:

| Threads | Throughput |
| ---: | ---: |
| 1 | 3.08 GiB/s |
| 4 | 4.24 GiB/s |
| 8 | 5.08 GiB/s |
| 12 | 7.62 GiB/s (short run) |

Control: the same benchmark with the page cache enabled reached 4.28 GiB/s
single-threaded, only ~40% above the bypassed figure and identical at 8 threads,
so the bypass is working and these are device numbers.

## Derived geometry

Each expert holds `2 × 2880 × 2880` (gate+up) plus `2880 × 2880` (down) =
24.9M parameters. MXFP4 costs 4.25 bits/weight including the one E8M0 scale byte
per 32 weights, giving **12.61 MiB per expert** and a **56.7 GiB routed pool**.
That reproduces the published ~60 GB checkpoint size, which validates the
accounting.

`config.json` excludes attention, router, embeddings, and head from MXFP4 — they
ship as bf16, which would put the resident trunk at ~4.3 GiB. Requantising them
ourselves is what makes this fit:

| Component | Precision | Size |
| --- | --- | ---: |
| Embedding | 4-bit | 0.29 GiB |
| Output head (untied) | 4-bit | 0.29 GiB |
| Attention | 8-bit | 0.92 GiB |
| Router | bf16 (precision-sensitive, and tiny) | 0.03 GiB |
| **Resident trunk** | | **1.52 GiB** |

KV is cheap because the sliding window is only 128 tokens: **0.29 GiB at 8K
context**, 1.13 GiB at 32K.

## Cache hit rates

Simulated with Godwit's LFU-plus-LRU policy, one independent cache per layer,
Zipf-distributed expert popularity drawn independently per layer.

The Zipf exponent is the one free parameter, so it is **calibrated against a
published measurement** rather than guessed: TurboFieldfare reports expert I/O
falling 166 → 88 ms/token at 16 slots with top-8 of 128, a 47% hit rate.
That fits α = 0.90.

| Slots/layer | Cache RAM | Hit rate | Reads/token |
| ---: | ---: | ---: | ---: |
| 4 | 1.77 GiB | 13.6% | 124.4 |
| 6 | 2.66 GiB | 27.7% | 104.1 |
| 8 | 3.55 GiB | 35.5% | 92.9 |
| 12 | 5.32 GiB | 45.3% | 78.8 |
| 16 | 7.09 GiB | 51.6% | 69.7 |

## Decode

Assumes 8-thread read bandwidth, and that overlap hides only half the compute —
**GPT-OSS has no shared-expert branch**, so there is less resident GPU work to
hide reads behind than TurboFieldfare had.

| Slots | Total RAM | I/O time | Compute | tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 3.57 GiB | 301 ms | 27 ms | **3.17** |
| 6 | 4.46 GiB | 252 ms | 27 ms | **3.76** |
| 8 | 5.35 GiB | 225 ms | 27 ms | **4.19** |
| 12 | 7.12 GiB | 191 ms | 27 ms | **4.89** |
| 16 | 8.89 GiB | 169 ms | 27 ms | **5.48** |

8-12 slots is the sweet spot: 5.4-7.1 GiB total leaves real headroom on a 16 GB
machine, and the returns above 12 slots are thin.

## Prefill

A chunk of C tokens makes 4C routing draws per layer over 128 experts. By
coupon-collector, **C ≥ 256 touches essentially every expert**, so the cost floor
is one full pass over the 56.7 GiB routed pool per chunk — **11.2 s at measured
bandwidth** — regardless of how few tokens the chunk holds.

Chunk size is therefore the whole game, and activations are almost free:

| Chunk | Activations | TTFT, 4K prompt |
| ---: | ---: | ---: |
| 1,024 | 16.9 MiB | 44.7 s |
| 2,048 | 33.8 MiB | 22.3 s |
| 4,096 | 67.5 MiB | 11.2 s |
| 8,192 | 135.0 MiB | 11.2 s |

At 67 MiB for a 4,096-token chunk there is no memory reason to chunk small. If
the chunk spans the whole prompt, **TTFT is a flat ~11 s for any prompt up to
several thousand tokens.** Intra-chunk attention adds roughly 1-2 s at C=4096
(≈5 TFLOP across the 18 full-attention layers), which does not change the shape.

### This model reproduces TurboFieldfare's measurement

Their 1,017-token prompt at chunk 128 is 8 chunks over a 12.9 GB pool = ~103 GB
of reads. At an M2 Air's ~2.5-3 GB/s that predicts 34-41 s; they measured
**36.7 s**. The model lands inside the range without being fitted to it, which
is the main reason to trust the GPT-OSS numbers.

It also implies their own prefill would improve severalfold with larger chunks.
They ran 17 prefill experiments, so there may be a constraint not visible from
outside — but nothing in the published record rules it out, and it is worth
testing.

## What this changes

1. **Build it.** 4-5 tok/s on a 120B model on a 16 GB laptop is a real
   capability, and it is in the same range TurboFieldfare achieves on a model a
   quarter the size.
2. **Chunk size is a first-class design parameter, not a tuning knob.** Size the
   prefill path for whole-prompt chunks from the start; retrofitting is painful.
3. **Positioning follows the numbers.** ~11 s to first token and ~4 tok/s after
   is not a chat experience. It is good for long-form generation from short
   prompts, batch work, and anything asynchronous.
4. **Losing the shared expert hurts.** Worth spending real design effort on what
   else can overlap the read window, since the obvious candidate is gone.

## Weakest assumptions

In the order most likely to be wrong.

- **The Zipf exponent.** MoE routers are trained with a load-balancing loss that
  actively suppresses the skew caching depends on, so α = 0.90 may be optimistic
  even though it fits their measurement. At α = 0.50, the 8-slot hit rate falls
  35.5% → 16.2% and decode drops to ~3.3 tok/s. Replace this with a real routing
  trace as soon as one can be captured.
- **MXFP4 dequantisation cost.** The 27 ms compute figure is a RAM-bandwidth
  model that ignores dequant entirely. If MXFP4's 32-weight block is slow on GPU
  — open question #1 — compute could exceed I/O and invalidate the whole table.
- **The 50% overlap recovery factor.** A guess, not a measurement.
- **Sustained vs burst bandwidth.** The 12-thread 7.62 GiB/s came from a short
  run and is likely burst. Estimates use the 8-thread figure; sustained
  multi-minute reads under thermal load on a fanless Air are unmeasured.
- **Uniform expert size.** Assumes no per-expert compression variance.
