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

**Pre-alpha. Nothing runs yet.** Currently in the ground-truth and
feasibility-check stage. See [docs/DESIGN.md](docs/DESIGN.md) for the
architecture and the open questions.

What exists:

- `ArchitectureSpec` — model-agnostic transformer description; kernels take
  these as specialisation constants rather than hardcoding a family
- `MXFP4` — reference CPU decoder for the OCP microscaling block format, the
  ground truth every kernel is validated against
- `ExpertCachePlanner` — LFU-with-LRU-tiebreak slot planner, pure logic and
  fully testable without a GPU

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
