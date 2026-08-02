# Architecture

A pipeline: read the hardware, enumerate every combination, score them, print
the best. Each stage is a package, and the two that hold decisions hold the
tests.

```
hw ──► advisor ──► fit ──► Estimate
        ▲   ▲       ▲
   catalog  engine  quant
   hfapi
```

## Packages

| Package | Responsibility |
|---|---|
| `internal/arch` | A model's shape. GQA/MLA/MoE are distinct cases, not scaling factors. |
| `internal/quant` | Bits per weight per format, calibrated against published GGUF file sizes. |
| `internal/hw` | Detection: `nvidia-smi`, `rocm-smi`, `sysctl`, `/proc`, plus the GPU spec table and the memory-bandwidth measurement. |
| `internal/engine` | What each runtime can load and where it can run. |
| `internal/fit` | The arithmetic. Pure. Memory, KV cache, decode and prefill speed. |
| `internal/catalog` | The embedded model list. |
| `internal/hfapi` | Reads any model's shape from Hugging Face. |
| `internal/advisor` | Search and ranking — the tool's opinion. |

## Why the split matters

`internal/fit` takes numbers and returns numbers. No AWS of the LLM world, no
clock, no filesystem, no network. That is what lets the memory model be checked
against real published file sizes rather than against itself, which is the
difference between an estimate and a guess.

`internal/advisor` is where judgement lives, and it is separated deliberately.
The scoring curves — how much a quantization level costs, how quickly extra
speed stops mattering — are opinions. Keeping them in one small package means
they can be argued with and changed without touching the arithmetic.

## Two things that look like details and are not

**Bandwidth is looked up, not reported.** No driver exposes memory bandwidth, so
`internal/hw/gpudb.go` is a table matched by substring, longest match first.
`nvidia-smi` says "NVIDIA GeForce RTX 4070 Ti SUPER"; matching the shorter "RTX
4070 Ti" row gives a card with a third less bandwidth and a decode estimate to
match.

**Host memory bandwidth is measured.** A table keyed on the CPU would be
guessing about the memory fitted — the same processor runs one DDR4 stick or
four DDR5 ones. `internal/hw/membench.go` copies buffers far larger than L3
across several threads and takes the best of three rounds.

## Adding things

**A model**: an entry in `internal/catalog/models.json`. Every field the maths
divides by is asserted present by `TestCatalogueEntriesAreComplete`.

**A quantization format**: a row in `internal/quant`. If it is a GGUF K-quant,
add a case to `TestWeightBytesMatchesPublishedGGUFSizes` with a real file size.

**A runtime**: an entry in `internal/engine`. The constraints — which formats,
which vendors, whether it offloads to CPU — are what change the answer.

**A GPU**: a row in `internal/hw/gpudb.go`. Bandwidth is the number that
matters; TFLOPS only affects prefill.
