# Measured results

Everything here was measured on the machine described below. Where a number
contradicts [ESTIMATE.md](ESTIMATE.md), this page is the one that is true —
that document is the pre-build projection, kept for the reasoning rather than
the figures.

## Two models, one binary

| | GPT-OSS-120B | GPT-OSS-20B |
| --- | ---: | ---: |
| Layers × experts | 36 × 128 | 24 × 32 |
| Install | 58.9 GiB | 11.2 GiB |
| Resident (8 slots) | 5.7 GiB | 4.1 GiB |
| Decode | 1.4-1.5 tok/s | 2.8-3.2 tok/s |
| Prefill | ~2.2 tok/s | ~4 tok/s |

The 20B ran first time with no code change beyond declaring its spec, which
demonstrates that dimensions really are read from `ArchitectureSpec` rather
than assumed. It is a narrow demonstration: same family, so tensor naming,
RoPE, activation, sinks and tokeniser were untested.

The speed difference is entirely reads. Both models fetch four experts per
layer at 12.61 MiB each, so a 20B token costs 24 layers of reads against the
120B's 36, and a smaller pool means a warmer cache.

## Machine

13" MacBook Air, Apple M4, 16 GB unified memory, **base 256 GB SSD**, macOS 26.
The SSD capacity matters more than anything else here; see below.

## Throughput

| | |
| --- | ---: |
| Decode | **1.4-1.5 tok/s**, flat across the generation |
| Prefill | ~2.2 tok/s |
| Time to first token | 10-13 s for a short prompt |
| Resident | 2.12 GiB trunk + 3.56 GiB expert slots |
| On disk | 58.93 GiB |

Decode is linear in output length, not quadratic — the KV cache holds every
previous token's keys and values.

## Where the time goes

From `godwit generate --profile`, 24 tokens at eight expert slots:

| Phase | Wall | GPU | Share |
| --- | ---: | ---: | ---: |
| **expert reads** | 11.90 s | — | **71.6%** |
| expert arithmetic | 2.78 s | 1.77 s | 16.8% |
| attention | 1.11 s | 0.73 s | 6.7% |
| output head | 0.37 s | 0.37 s | 2.3% |
| router | 0.26 s | 0.06 s | 1.6% |
| CPU norm + residual | 0.02 s | — | 0.1% |

**The GPU is busy 17.6% of the time.** This is an I/O-bound design and the
profile says so plainly.

## The SSD is the constraint

Reads run at ~2 GiB/s and the runtime's read path is *at device speed*. Five
independent measurements agree: `nvmebench` against four different layer files
at one and eight threads, and the runtime's own profiler.

An earlier benchmark reported 5.08 GiB/s and anchored the whole feasibility
case. It was wrong: the test file had just been written by `dd`, and macOS
`F_NOCACHE` prevents *new* caching without evicting pages already present, so
the benchmark read largely from RAM. Anyone measuring disk throughput on macOS
should know this.

Small Apple SSDs use fewer NAND dies and are genuinely slower. A 256 GB module
reads ~2 GiB/s; 512 GB–2 TB modules reach 3–6. **The same code should reach
roughly 3-5 tok/s on a Mac with a larger SSD**, and that has not been verified
because no such machine was available.

## What did not work

Recorded because they cost as much to find out as the wins.

| Attempt | Result |
| --- | --- |
| Prefetch by reusing the previous layer's expert IDs | 4.8% hit against 3.1% for chance. This rules out *that* method, not prefetching — see Routing |
| Concurrent miss reads | No effect; one and eight threads both reach ~2 GiB/s |
| Splitting each read into chunks | Slightly *worse* (1.54 → 1.46 tok/s) |
| More cache slots | Hit rate rises, throughput does not; 24 slots swaps and collapses to 0.12 tok/s |
| Staging activations in threadgroup memory | No effect |
| Codebook in constant address space | No effect |
| File defragmentation | No effect; a forced byte copy reads at the same speed |

## What did work

| Change | Effect |
| --- | ---: |
| Expert cache (8 slots, LFU) | 0.91 → 1.21 tok/s |
| Batching GPU submissions (504 → 108 per token) | 1.21 → 1.42 tok/s |
| `F_NOCACHE` on expert files | 1.43 → 1.54 tok/s |
| 4 rows per SIMD group in the MoE kernel | +18% wide, +11% narrow, p < 0.0001 |
| Expert-major layer ordering | Prefill 14.2 → 10.4 s |
| Read/compute overlap | ~2-3% decode, 4-5% prefill |

## Routing

Measured on real layers with `godwit trace-layers`, 36 layers × 64 tokens:

- Cache hit rate at 8 slots: **38-42%** in real generation
- Adjacent *layers* share **4.8%** of their experts (chance: 3.1%)
- Adjacent *tokens* at one layer share **54%**

That asymmetry is why caching along the token axis works.

**A correction.** This section used to end "caching works, prefetching does
not", which the 4.8% does not support. Expert-set overlap between adjacent
layers answers one question: can you predict layer *n+1* by reusing layer *n*'s
chosen expert IDs? No. It says nothing about running layer *n+1*'s router early
on layer *n*'s hidden state, which computes the gate rather than assuming the
choices repeat. That is what colibrì's lookahead thread does, and what
[Pre-Attention Expert Prediction](https://arxiv.org/abs/2511.10676) does with a
learned linear probe, reporting 94.69% accuracy on Qwen3-30B.

Neither is implemented here, and neither has been ruled out. The measurement
above is correct; the conclusion drawn from it was not.

Worth noting what the ceiling would be on this machine. Lookahead hides read
latency behind compute, and the GPU is busy 17.6% of decode — so overlapping
read(n+1) with compute(n) recovers at most that, roughly 1.2x. The larger lever
is reducing bytes read, which is 71.6% of decode and untouched by any prefetch.

## Quantisation

`Scripts/analysis/requant_quality.py`. The trunk ships as BF16 and must be
quantised to fit; 4-bit does not survive contact with the data.

| Tensor | Scheme | Result |
| --- | --- | --- |
| Output head | int8 | 98.4% top-1 agreement |
| Output head | int4 | **81.1%** — a different model |
| Embeddings | int8 | cosine 1.0000 |
| Embeddings | int4 | cosine 0.9880 |

## Correctness

Every stage is checked against a NumPy reference built from the same installed
bytes:

| Check | Result |
| --- | --- |
| MXFP4 decode vs OpenAI's tensors | byte-exact |
| Expert feed-forward | 3e-06 to 6e-05 relative |
| Attention (1, 8, 32, 200 tokens) | RMS 3e-04 |
| Full layer | RMS 6e-04, routing matches exactly |
| Tokeniser vs `tiktoken` | 14/14 exact |
| Installer vs checkpoint | 18/18 expert sections byte-exact |

Activations are FP16 except the residual stream, which is FP32 — a range
requirement, not a precision preference. GPT-OSS was trained in bfloat16
(~3e38); FP16 stops at 65504, and some prompts drive activations past it,
producing NaN logits and a model that emits one token forever.
