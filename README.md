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

**It generates text.** The full 58.93 GiB model runs on a 16 GB M4 Air,
streaming experts from disk.

```
$ python3 Scripts/chat.py model.gwt "What is the capital of France?"

<|channel|>analysis<|message|>The user asks: "What is the capital of France?"
This is a straightforward factual question. The answer: Paris.<|end|>
<|start|>assistant<|channel|>final<|message|>The capital of France is **Paris**.<|return|>
```

On a raw completion the model puts 58.2% on " Paris" for "The capital of France
is", with " a", ":" and " London" behind it.

Verified against NumPy references at every level: MXFP4 decode, expert
feed-forward, attention with sinks, and a complete layer including routing.

**Decode is linear**, at 1.21 tok/s on an M4 Air with an eight-slot expert
cache, flat across the generation rather than degrading. Prefill runs about
1.6 tok/s.

Still short of the ~5 tok/s projected, and the reason has moved. Reads are now
24% of decode time and expert arithmetic 11%; **65% is command-buffer round
trips** — each expert submits its own and blocks on it, 144 times per token.
Batching those is the next piece of work.

There is no Swift tokeniser yet either; `Scripts/chat.py` uses tiktoken's
`o200k_harmony`, which matches the model's vocabulary exactly.

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
