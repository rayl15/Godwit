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
| Bounded slot cache pays for itself | 166 → 88 ms/token at 16 slots/layer | Slot count is a tuning knob, default 16. |
| Coarse overlap beats fine-grained | 4.404 → 4.736 tok/s coarse; 4.799 → 4.648 fine | One batch per layer. Resist cleverness. |
| Cross-layer expert prediction does not work | 7% hit rate predicting next layer from current | Do not build speculative prefetch. |
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

## Non-goals

- Beating MLX or llama.cpp on machines where the model already fits. We lose
  that comparison by construction and should say so plainly.
- Training, fine-tuning, or LoRA.
- Vision, audio, or any non-text modality.
- Serving multiple concurrent requests. Single-stream until the single stream is
  good.
