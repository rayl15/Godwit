# Design

## Thesis

For a mixture-of-experts model, disk footprint and working-set size are very
different numbers. GPT-OSS-120B is ~60 GB at 4 bits but activates ~5.1B
parameters per token. If the shared trunk stays resident and routed experts are
fetched on demand, the memory requirement tracks the working set instead of the
checkpoint.

This is only worth doing where the ratio is extreme. Running a 14 GB model on a
16 GB machine is a rounding error; MLX already does it, faster. Running a 60 GB
model on the same machine is a capability that does not otherwise exist.

## Resource split

| Component | Residency | Rationale |
| --- | --- | --- |
| Embeddings / output head | Resident, 8-bit | Touched every token; 4-bit measurably breaks the head |
| Attention projections | Resident | Touched every layer, every token |
| Router | Resident | Must run before we know what to fetch |
| Norms, scalars | Resident | Negligible |
| Shared experts | Resident where present | Useful GPU work to overlap reads against — but **GPT-OSS has none** |
| **Routed experts** | **Streamed** | Bulk of the checkpoint, sparsely used |
| KV cache | Resident | Grows with context; bounded for sliding-window layers |

## Per-token loop

```
for each layer:
    attention + router          (GPU, resident weights)
    plan top-k against cache    (CPU, pure)
    dispatch cached experts     (GPU) ─┐ overlapped
    fetch missing experts       (CPU) ─┘
    combine routed outputs      (GPU)
```

Overlap is deliberately coarse: one dispatch against one batch of reads.
Finer-grained schemes are covered under inherited findings below.

TurboFieldfare overlapped reads against its resident shared-expert branch.
**GPT-OSS has no shared expert**, so the only work available to hide reads
behind is the experts already in cache — which is exactly the work that shrinks
as the cache does. What else can fill the read window is an open problem.

## Inherited findings

TurboFieldfare published measurements that are expensive to reproduce. These are
facts about the problem, not code, and we take them as priors — each still needs
re-validation on our own workload, since our expert-to-resident ratio is far
higher and some conclusions may not transfer.

| Finding | Their measurement | How we use it |
| --- | --- | --- |
| `pread` beats `mmap` for cold experts | 2.79 ms vs 9.88 ms; 3.97 vs 0.50 tok/s end-to-end | Explicit reads from day one. Do not revisit. |
| Persistent workgroups beat SIMD-cooperative MoE | routed phase 239 → 60 ms; decode +51% | Start with persistent-workgroup dispatch. |
| LFU beats LRU for expert eviction | 72.6 → 64.8 ms/token | Implemented in `ExpertCachePlanner`. |
| Zipf-modelled routing skew | fitted α = 0.90 | **Superseded twice.** Real layers give 73.6% at 8 slots, far above both the model and the embedding proxy; see [ESTIMATE.md](ESTIMATE.md). |
| Bounded slot cache pays for itself | 166 → 88 ms/token at 16 slots/layer | Slot count is a tuning knob, default 16. |
| Coarse overlap beats fine-grained | 4.404 → 4.736 tok/s coarse; 4.799 → 4.648 fine | One batch per layer. Resist cleverness. |
| Cross-layer expert prediction does not work | 7% hit rate predicting next layer from current | **Confirmed on GPT-OSS: 4.8% against 3.1% chance.** Do not build speculative prefetch. |
| Speculative reads hurt | probe hit 10.63 GB/s; decode fell 4.937 → 4.742 tok/s | Do not prefetch on idle bandwidth. |
| 4-bit KV cache fails quality, and is larger at long context | rejected after full eval | FP16 KV. Do not revisit without a quality harness. |

Their prefill work is the least transferable: 9 of 17 experiments were rejected,
and their time-to-first-token remains poor (23 s on a 3,000-token prompt). We
should expect to solve prefill ourselves rather than inherit it.

## Open questions

These are the decisions that determine whether the project works. In rough
order of how badly a wrong answer hurts.

### 1. Is MXFP4 dequantisation fast enough on GPU? — ANSWERED, with a caveat

**MXFP4 is not the problem.** Measured on an M4 with fused dequant-GEMV kernels
of identical structure (`godwit bench dequant`, median of 7 rounds):

| Shape | MXFP4 | affine int4 |
| --- | ---: | ---: |
| 5760 × 2880 | 27.8-33.2 G weights/s | 22.3-26.1 |
| 2880 × 2880 | 12.9-16.8 G weights/s | 12.2-15.1 |

MXFP4 was ahead in five of six paired comparisons. The smaller block and
codebook lookup cost nothing measurable, so the format choice does not decide
the project. GPU output is validated against `MXFP4.swift` to a relative error
below 1e-3, and that check is a test.

**The caveat is the kernel, not the format.** Against the 15.9 G weights/s
decode budget, the narrow shape lands at 0.81-1.06× — marginal or short. But
these naive kernels reach only 6-16 GiB/s where the M4 has roughly 120 GB/s of
memory bandwidth, so they are latency- and occupancy-bound, not bandwidth-bound.
The wide shape is consistently ~2× the narrow one at the same total work, which
points at threadgroup count rather than arithmetic.

That is the same wall TurboFieldfare hit, and persistent workgroups took their
routed phase from 239 to 60 ms. Applying that here is the next kernel task, and
there is a large amount of headroom to claim.

### 2. What is the real ceiling at a 30:1 stream-to-resident ratio? — ANSWERED

**Decode 3-5 tok/s; prefill costs a full pass over the routed pool per chunk.**
Worked through in [ESTIMATE.md](ESTIMATE.md) against measured NVMe bandwidth and
a hit-rate simulation calibrated to TurboFieldfare's published numbers.

Two consequences carry into the design:

- **Chunk size is a first-class parameter.** Any chunk past ~256 tokens touches
  every expert in a layer, so prefill costs 56.7 GiB of reads whether the chunk
  holds 256 tokens or 8,192. Activations are ~67 MiB at 4,096 tokens, so there
  is no reason to chunk small. Whole-prompt chunks give a flat ~11 s TTFT.
- **There is no shared expert to overlap against.** TurboFieldfare hid read time
  behind its resident shared-MLP branch; GPT-OSS has no such branch. Finding
  something else to overlap is now an open design problem in its own right.

### 3. Does external Thunderbolt NVMe hold up?

Internal Apple NVMe measured 5.08 GiB/s at 8 threads; a good TB4 enclosure does
~3 GB/s with higher latency. Since reads are the bottleneck, this scales
throughput directly.

Not on the critical path for development — the checkpoint fits internally — but
it decides whether people without large internal disks can use the result, so it
still needs measuring rather than assuming.

### 4. How much does unified memory carry us?

`makeBuffer(bytesNoCopy:)` with shared storage means the CPU reads into a page
the GPU then reads directly — no copy. A discrete GPU has no equivalent; a CUDA
port would need pinned memory plus a PCIe transfer, or GPUDirect Storage to DMA
NVMe straight into VRAM.

Not a problem today, but the streaming interface should not assume zero-copy in
its *shape*, so a future port is a new backend rather than a rewrite.

## How model-specific is this?

Structurally generic, practically single-model, and worth being precise about.

| | Files | Lines |
| --- | ---: | ---: |
| Model-agnostic | 17 | 2,209 (78%) |
| GPT-OSS-specific | 4 | 806 (22%) |

The generic side is everything load-bearing: cache planner, streaming, MXFP4 and
int8 codecs, `ModelReader`, `Router`, `KVCache`, `Attention`, `TransformerLayer`.
These read dimensions from `ArchitectureSpec` and do not know what model they are
running.

What remains model-specific:

- **Tensor names.** `Installer` and `InstallLayout` know that experts live at
  `model.layers.N.mlp.experts.gate_up_proj_blocks`. Qwen3 names them differently
  *and* stores them per-expert rather than stacked — and the second part matters
  more, because "one expert is one contiguous byte range" depends on expert being
  the leading dimension.
- **RoPE scaling.** YaRN only; no linear, NTK, or unscaled variant.
- **Quantisation assumption.** MXFP4 experts with a BF16 trunk.

Two constants that *were* hardcoded are now parameters: head dimension is a
Metal function constant (64 for GPT-OSS, 128 for Qwen3), and the activation's
alpha and clamp come from the spec. Both were the dangerous kind of hardcoding —
a wrong head dimension computes confidently wrong numbers rather than failing.

**Deliberately not generalised further yet.** Abstracting against one example
produces the wrong abstraction; a second model would show what actually varies
rather than what we imagine does. GPT-OSS-20B is the obvious candidate — same
family, different dimensions, and small enough to run alongside for comparison.

## Measuring on a fanless machine

The M4 Air's GPU throughput drifts by up to 2x under sustained load, which is
larger than most differences worth detecting. Three methodology failures were
hit and fixed here, in order:

1. **All variants in one process.** Each heated the GPU for the next: the
   *unchanged* naive baseline measured 27 G w/s with five configs ahead of it
   and 13 with eight. Position in the run was worth 2x.
2. **Separate processes, seconds apart.** Better, but the machine still drifted
   between the two runs of a pair; ratios on identical code swung 0.62 to 2.02.
3. **A four-round sign test.** 4 of 4 looks convincing and is not — chance
   produces it one time in eight. A promising 4/4 result evaporated to 6/12.

What works: `godwit ab-kernel` alternates two kernels inside one process, ~40
pairs milliseconds apart, and reports the median B/A ratio with a sign test.
Both variants see the same thermal state, so the ratio means something even
when neither absolute number does.

**Report ratios, not absolutes, and never trust fewer than ~20 pairs.**

## Kernel findings

Measured with the harness above. Negative results included, because they cost
as much to obtain as the positive ones.

| Change | Effect | Verdict |
| --- | --- | --- |
| Persistent workgroups, grid-stride | +2.2%, p=0.0016 | Real but negligible |
| Staging x in threadgroup memory | none | Rejected |
| Codebook in constant address space | none, p=0.75 | Rejected |
| **4 rows per SIMD group** | **+18.2% wide, +11.2% narrow, p<0.0001** | **Production** |

The diagnosis mattered more than any individual attempt. The kernel is neither
bandwidth-bound (19% of the M4's) nor ALU-bound (2%) — it is latency-bound.
Each lane walked three blocks for its row, one dependent load at a time, with
nothing to overlap the memory latency against. Processing four rows at once
gives four independent load streams and reuses each loaded x value across all
of them.

That also explains why the first three attempts failed: launch overhead, memory
placement, and the lookup were never the constraint.

## Where decode time actually goes

Measured, not estimated. `godwit generate --profile` records wall time per
phase and GPU busy time per command buffer.

| Phase | Calls | Wall | GPU | Share |
| --- | ---: | ---: | ---: | ---: |
| **expert reads** | 1,897 | **11.90 s** | — | **71.6%** |
| experts (compute) | 828 | 2.78 s | 1.77 s | 16.8% |
| attention | 828 | 1.11 s | 0.73 s | 6.7% |
| output head | 23 | 0.37 s | 0.37 s | 2.3% |
| router | 828 | 0.26 s | 0.06 s | 1.6% |
| CPU norm + residual | 3,312 | 0.02 s | — | 0.1% |

**GPU busy: 17.6% of wall time.**

Two earlier attempts to attribute this arithmetically put reads at 24% and then
28%. They are 71.6%. Both estimates were wrong, and the second was wrong in the
same direction as the first, which is what estimating instead of measuring
tends to produce.

The reads run at **1.97 GiB/s**, against 5.08 GiB/s for the same 12.6 MiB reads
at eight threads in the standalone benchmark. Reading the misses concurrently
was tried and did **not** help: at eight slots a token has only two or three
misses per layer, so there is never enough outstanding work to saturate NVMe.

Bypassing the page cache with `F_NOCACHE` is worth about 8% (1.43 → 1.54
tok/s). We hold the experts we want in our own slots, so the page cache only
buys a second copy of 59 GiB it cannot hold anyway, and evicts something useful
 time. Worth having, not a fix.

### More cache slots do not help

The obvious lever, measured:

| Slots | Cache RAM | Hit rate | Decode |
| ---: | ---: | ---: | ---: |
| 8 | 3.56 GiB | 42.7% | **1.54 tok/s** |
| 12 | 5.35 GiB | 49.8% | 1.50 tok/s |
| 16 | 7.13 GiB | 55.6% | 1.48 tok/s |
| 24 | 10.69 GiB | 64.1% | **0.12 tok/s** |

Hit rate rises monotonically and throughput does not. From 8 to 16 slots misses
fall 23% while read time falls 6% — per-read latency dominates, not bytes — and
the larger cache starves the rest of the system of exactly the memory that
makes reads fast.

At 24 slots it collapses outright: 10.7 GiB of slots plus 2.1 GiB of trunk on a
16 GB machine swaps, and GPU utilisation drops to 1.5%. **Memory pressure, not
hit rate, sets the ceiling**, which the feasibility estimate did not model at
all — it treated slots as a free lever bounded only by RAM.

Eight slots is the setting. It is also the cheapest.

### What is actually left

Reads run at roughly 2 GiB/s inside the runtime. Three explanations were tested
and none held:

**Not request concurrency.** Reading misses in parallel did nothing, and
benchmarking the real model file directly showed one thread and eight threads
both reach ~2 GiB/s. Splitting each expert read into four concurrent chunks
made it slightly worse (1.54 → 1.46 tok/s).

**Not fragmentation.** The installer writes tensor-major with scattered
`pwrite`s, which looked like an obvious culprit. A forced byte copy of a layer
file reads at the same speed as the original, so the physical layout is not the
problem. An earlier note here claimed otherwise on the strength of a single
unpaired measurement against a `cp` that APFS had silently turned into a
copy-on-write clone; that claim was wrong and is retracted.

**Not the page cache, mostly.** `F_NOCACHE` is worth about 8% and is kept, but
it is not the difference between 2 and 5 GiB/s.

The gap against the 5.08 GiB/s the standalone benchmark reached remains
unexplained. What differs between them is not yet isolated: the benchmark used
a freshly written 6 GiB file with the machine otherwise idle, while the runtime
reads a 59 GiB installation while holding 3.6 GiB of slots and 2.1 GiB of trunk
resident. Measuring that properly needs a cold page cache, which needs `purge`
and root.

The structural constraint stands regardless. Routing must finish before a read
starts and the read before the expert runs, 36 times a token, with two or three
misses outstanding at a time. Prefetch would break the chain and was measured
not to work here.

## Numerical precision## Numerical precision

Activations are FP16 throughout: q, k, v, the attention output, and the residual
stream. That is the KV cache format the memory budget assumes, so it is a
constraint rather than a preference.

Measured cost, from `check-attention` against an FP32 reference: **RMS relative
error ~3e-4**, stable from 1 to 200 tokens. Peak element error tracks one FP16
ULP on the largest output, which for this layer is ~1.3e-2 relative to RMS
because the output's peak is ~26x its RMS.

The peak figure is a property of the format, not of the kernel, and tolerances
in the checks are derived from it rather than chosen. A fixed threshold tight
enough to look impressive would fail a correct implementation.

## Non-goals

- Beating MLX or llama.cpp on machines where the model already fits. We lose
  that comparison by construction and should say so plainly.
- Training, fine-tuning, or LoRA.
- Vision, audio, or any non-text modality.
- Serving multiple concurrent requests. Single-stream until the single stream is
  good.
