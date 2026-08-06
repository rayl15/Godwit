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
| Serving low-weight experts at lower precision | Costs more accuracy than it saves bytes — see Precision by router rank |
| Pre-warming the cache from the range map | Real signal, no benefit — the cache finds the same experts in two tokens |
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

## Precision by router rank

`Scripts/analysis/precision_by_rank.py`, on real routing dumps from both
families.

[HOBBIT](https://arxiv.org/abs/2411.01433) reads cache-miss experts at reduced
precision, arguing that less critical experts tolerate it, and reports up to
9.93x. Expert reads are 71.6% of decode here and run at device speed, so
reading fewer bytes is the only large lever left. It was worth an hour to find
out whether the argument holds before writing a 2-bit kernel.

It does not. Two measurements kill it, and neither is close.

**The routers are not peaky enough for rank to isolate the damage.**

| | rank 1 | last rank | ratio | mass in the bottom half |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B, top-8 | 23.5% | 6.9% | 3.4x | 33.4% |
| GPT-OSS-120B, top-4 | 37.6% | 17.5% | 2.1x | 37.6% |

A third of the routing mass sits in the ranks that would be degraded. HOBBIT's
argument needs a tail that contributes almost nothing; neither model has one.

**The precision headroom is already spent.** HOBBIT's gain comes from the
FP16 → INT4 range. These experts are stored at 4.25 bits already, so what
remains is 4 → 2, which is where the format collapses:

| bits/weight | error on W·x, Qwen3 | GPT-OSS |
| ---: | ---: | ---: |
| 4.25 | baseline | baseline |
| 3.25 | 26.6% | 27.7% |
| 2.25 | 50.9% | 53.2% |

Combining the two, every policy of the form "serve ranks ≥ R at B bits" costs
slightly *more* added error than the bytes it saves — a ratio near 1.1:1, on
both models, at every cut point. Serving Qwen3's ranks 5-8 at 2 bits saves
15.7% of expert traffic for 17.0% added error in the MoE block.

Not implemented. The paper is sound; the precondition is a model whose router
concentrates its mass and whose experts are still at 16 bits, and neither holds
here.

## Lookahead routing

`Scripts/analysis/lookahead_accuracy.py`, on real decode traces from both
families.

The question the 4.8% figure above does *not* answer: if you run a layer's own
router on the residual **entering** that layer — before its attention block has
run — how much of the real selection do you recover?

| | accuracy | chance | lift | top-1 recovered |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B, top-8 | **90.6%** | 6.2% | 14.5x | 99.6% |
| GPT-OSS-120B, top-4 | **87.1%** | 3.1% | 27.9x | 99.1% |

Accuracy is lowest in the first few layers — 80.3% and 75.5% — and settles above
90% for the rest of the depth, which is consistent with early layers changing
the residual most.

The top-ranked expert, which carries the largest routing weight, is recovered
**over 99%** of the time in both models. So the errors are concentrated in the
tail, where a miss costs the least.

This is the measurement that should have been made before writing "prefetching
does not work". Adjacent layers sharing 4.8% of their experts is true and
irrelevant: a lookahead scheme does not reuse the previous layer's choices, it
computes the gate early. colibrì reports 71.6% one layer ahead and ships it;
these numbers are higher because the prediction here is made within the layer,
skipping only attention rather than a whole block.

**What it would buy is smaller than the accuracy suggests.** Prefetching hides
read latency behind compute, and the GPU is busy 17.6% of decode — so the
ceiling on this machine is roughly 1.2x, not the multiples reported by projects
running on hardware where compute is the larger share. A wrong prediction also
costs a wasted read, at 0.8 of 8 experts for Qwen3, unless the speculative
fetch is cancelled once the true routing is known.

### Implemented, and worth less than the accuracy suggests

`godwit chat --lookahead`. Off by default.

Interleaved A/B on Qwen3, 9 pairs, nothing else touching the disk:

| | mean | sd | range |
| --- | ---: | ---: | ---: |
| off | 2.222 tok/s | 0.079 | 2.09–2.38 |
| on | 2.292 tok/s | 0.049 | 2.21–2.41 |

**+3.1%, faster in 9 of 9 pairs, sign test p = 0.002.** Real, consistent, and
far below the ~1.2x this section estimated. It also halves the spread, which is
worth as much as the mean: the slow runs are the ones it rescues.

The reported cache hit rate jumps from 45% to 89.6% with it on, and that number
is a trap. Speculation does not avoid a read, it *moves* it earlier, so an
expert fetched on a guess counts as a hit when the real routing arrives. Bytes
off disk barely change. Only wall clock is evidence here.

The gap between 3.1% and the estimate is the settle: the real acquire must wait
for any speculative read still in flight, or a real read lands under a
speculative one in the same slot. So the win is only what actually overlapped
during attention, and attention is a small share of the 17.6% the GPU is busy
at all.

Decode only. In prefill the union of every token's experts routinely exceeds
the eight slots, so speculating there evicts faster than it reads — the first
implementation hung on exactly that.

### A note on how this was measured

The first version of this script read the router as int8 and reported 6.3%
against 6.2% chance — a perfect null, and exactly the answer the repo already
believed. The router is stored BF16, because it decides which experts fire and
is small enough to keep at full width.

The script now reproduces the engine's own selection from the true router input
before it measures anything, and aborts below 99%. That check caught the int8
error, and then caught a second one: GPT-OSS's router carries a bias that Qwen3's
does not, worth 10 points of agreement on its own.

## Pre-warming the cache from the range map

`Scripts/analysis/topic_prior.py` and `prewarm_sim.py`.

Cache-aware routing ([arXiv:2412.00099](https://arxiv.org/html/2412.00099v2)
and others) raises hit rate by biasing selection toward resident experts. It
works, and it changes which experts run, so it buys speed with output quality.

The range map suggested something better: if it can predict which experts a
generation will use, the cache could be seeded from the prompt's topic —
changing only what is resident, never what runs, leaving output bit-identical.

**The prediction is real.** Weighted by how often each expert fires, against
three topically distinct generations on Qwen3:

| generation | own-topic experts | enrichment |
| --- | ---: | ---: |
| Python | 31.1% of fires vs 15.6% of map | **2.00x** |
| Chinese | 41.3% vs 7.4% | **5.61x** |
| Maths | 38.6% vs 13.0% | **2.98x** |

Off-topic experts are suppressed in step — `medical` at 0.20x during Python,
`sql` at 0.11x during maths. Two things fell out that were not asked for:
Chinese text pulls `japanese` experts at 3.98x, so the map has captured
something real about CJK, and `chat` experts are enriched in all three traces,
which is what a general-purpose expert should look like.

**The benefit is nil anyway.** Simulated against the real policy — per-layer,
8 slots, LFU with LRU tie-break:

| decisions | baseline | pre-warmed | gain |
| ---: | ---: | ---: | ---: |
| 48 (one token) | 0.0% | 9.4% | +9.4% |
| 96 | 14.1% | 18.8% | +4.7% |
| 240 | 25.5% | 27.3% | +1.9% |
| 960 | 19.6% | 20.0% | +0.5% |
| 1488 | 20.8% | 21.1% | +0.3% |

Pre-warming is a cold-start optimisation, and the cold start is two tokens
long. LFU with LRU finds the same experts unaided almost immediately, after
which the seed is just history. The ceiling is structural regardless of how
good the prior gets: 384 seeded slots against 11,904 expert accesses caps the
saving at 3.2% even if every guess were right and never re-read.

Not implemented. The measurement is kept because the enrichment numbers are
worth having on their own — they are the first direct evidence that the range
map predicts behaviour rather than merely describing it.

One caveat on the baseline: these simulations run from `dump-routing`, which
records only the last token's decision per step, so they miss the expert reuse
within a prefill batch. That is why the hit rates here sit near 20% while real
generation measures 38-45%. The comparison between columns is unaffected.
