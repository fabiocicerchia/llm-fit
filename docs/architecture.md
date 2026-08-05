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

## Fitting is the easy half

Most calculators answer "does it fit". That is the question that matters least.
A 70B at Q4_K_M *fits* on a 12GB card with enough system RAM — llama.cpp will
load it, spill sixty layers into DDR, and generate at two tokens per second,
which is slower than you read.

So every answer here carries a speed estimate, and the speed estimate is what
decides whether a model is suggested at all.

## The arithmetic

**Decode is memory-bandwidth bound.** Generating one token reads every active
weight and the entire KV cache, once. So tokens per second is bandwidth divided
by bytes-read-per-token — not FLOPs. It is why a 4090 and an A100 are far
closer in single-stream chat than their compute suggests, and why a second GPU
buys capacity rather than speed.

**Prefill is compute bound.** Ingesting a prompt is a matrix-matrix product over
many tokens at once, so it scales with FLOPs. A card can be slow to chat and
quick to read a document.

Four corrections do most of the work, and skipping any one of them produces
confidently wrong advice:

| | |
|---|---|
| **Quantization is not the number in its name** | Q4_K_M is 4.83 bits per weight, not 4. Block scales, and higher-precision embeddings, are the difference between predicting 35GB for a 70B and the 42.5GB it really is. |
| **Grouped-query attention** | Llama-3-70B has 64 query heads and 8 KV heads. Sizing its cache off query heads overstates it 8×. DeepSeek's MLA is a different formula again — roughly a fourteenth of the GQA equivalent. |
| **Mixture of experts** | Qwen3-30B-A3B occupies 30B of memory and reads 3.3B per token. Memory of a 30B, speed of a 3B — which is why it is often the best answer on a small card, and why it tops the list above. |
| **The KV cache is not a rounding error** | At 128k context an 8B model's cache exceeds its weights, and generation slows as the conversation grows. Quantizing the cache to `q8_0` usually buys more than dropping a quantization level. |

The memory model is checked against published GGUF file sizes rather than
against itself — see `internal/fit/fit_test.go`, which asserts predictions land
within 3% of real releases for Llama-2, Llama-3 and Mistral across seven
quantizations.

## Runtimes

llama.cpp, Ollama, vLLM, SGLang, ExLlamaV2, MLX, TensorRT-LLM. The differences
that change the answer are modelled, not just listed:

- **vLLM and SGLang do not usefully offload.** On a 12GB card a 70B is out of
  reach whatever the system RAM. `llm-fit` says so rather than reporting a
  number for a configuration that will fail to load.
- **ExLlamaV2 is NVIDIA-only**, MLX is Apple-only, FP8 needs Ada or Hopper.
  Engines the machine cannot run are shown with the reason.
- **Serving is a different question.** `-serving` switches the ranking from
  single-stream latency to how many concurrent requests the leftover memory
  holds, and drops the runtimes that are not built for it.

## What the numbers are, and are not

Estimates from bandwidth and compute, not measurements. On hardware in the
built-in table they are usually within about 20% — the right ballpark for
choosing, not a benchmark. Three things push them off:

- A GPU not in the spec table has no known bandwidth. It is flagged, and the
  speed figures that follow are guesses.
- Speculative decoding, prefix caching and batch-of-one assumptions all move
  real throughput.
- The quality ranking (`Score`) is a judgement, not a measurement. It encodes
  that a 32B at Q4_K_M beats an 8B at Q8_0, and that neither is beaten by a
  109B at 1.75 bits. Those trades are pinned in `advisor_test.go`.
