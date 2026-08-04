# Contributing

Issues and pull requests are welcome. This file covers the one convention that
is unusual enough to be worth stating.

## Claims are measured

If a change is meant to make something faster, the commit message should say by
how much and how that was established. "Optimised the expert loop" is not a
claim anyone can check; "1.21 → 1.42 tok/s, 24 tokens, eight slots" is.

This is not pedantry. Three conclusions in this repository's history were
confidently wrong and had to be retracted:

- A kernel reported as 1.75× faster was measuring **run order**, because every
  variant ran in one process and each heated the GPU for the next.
- Slow reads were attributed to **file fragmentation** on the strength of one
  unpaired measurement against a `cp` that APFS had silently turned into a
  copy-on-write clone.
- The bandwidth figure underpinning the entire feasibility estimate was
  **page cache**, because macOS `F_NOCACHE` does not evict pages already
  resident.

Each looked reasonable. Each survived until someone measured it properly.

## Negative results stay

`docs/DESIGN.md` records things that did not work — staging activations in
threadgroup memory, moving the codebook to constant address space, splitting
reads into chunks, adding cache slots beyond eight. They cost as much to
establish as the wins, and deleting them means the next person pays again.

If your change makes an experiment obsolete, update the entry rather than
removing it.

## Benchmarking on Apple laptops

GPU throughput on a fanless machine drifts by up to 2×, which is larger than
most effects worth detecting. Use `godwit ab-kernel`: it alternates two kernels
inside one process, milliseconds apart, and reports a median ratio with a sign
test. A single before/after timing is not evidence, and four samples in one
direction is not either — chance produces that one time in eight.

## Correctness

Every numerical stage has a NumPy reference in `Scripts/analysis/`, built from
the same installed bytes the runtime reads. If you touch a kernel, run:

```bash
Scripts/test.sh
godwit check-expert    model.gwt reference-dir
godwit check-attention model.gwt reference-dir
godwit check-layer     model.gwt reference-dir
```

`check-layer` verifies that expert *selection* matches exactly, not just that
the numbers are close. Routing is a discrete choice — drift there means a
different set of experts ran, which no numerical tolerance would catch.

Tolerances are derived from the number format rather than chosen. FP16 has an
11-bit significand, so a correct kernel can differ by one ULP of the largest
element; a threshold tighter than that fails working code.

## Style

Match the surrounding code. Comments explain *why*, especially where the
obvious approach is wrong — Metal function constants over buffer parameters,
`pread` over `mmap`, FP32 for the residual stream. Those notes exist because
each was learned the hard way.

## Running the tests

```bash
Scripts/test.sh
```

Use the script rather than `swift test`. On a machine with only the Command
Line Tools installed, Swift Testing needs a framework search path and two
rpaths that the script supplies; without them the test bundle compiles and
then fails to load.
