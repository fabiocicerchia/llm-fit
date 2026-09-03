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
| `internal/hw` | Detection: `nvidia-smi`, `rocm-smi`, `sysctl`, `/proc`, plus the GPU spec table (`gpus.json`) and the memory-bandwidth measurement. |
| `internal/engine` | What each runtime can load and where it can run. |
| `internal/fit` | The arithmetic. Pure. Memory, KV cache, decode and prefill speed. |
| `internal/catalog` | The embedded model list. |
| `internal/hfapi` | Reads any model's shape from Hugging Face. |
| `internal/gguf` | Reads the header and tensor directory of a GGUF file on disk. |
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
`internal/hw/gpus.json` is a table matched by substring, longest match first;
`gpudb.go` embeds it, rejects a row missing a divisor, and does the matching.
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

**A GPU**: a row in `internal/hw/gpus.json`. Bandwidth is the number that
matters; TFLOPS only affects prefill. `vram_gib`, `bandwidth_gbs` and `vendor`
are required — the loader panics at startup rather than let a missing key
become a zero the speed estimate divides by.

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
buys capacity rather than speed *under a layer split* — see below, because that
last clause is doing more work than it looks.

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

### Multi-GPU: the two strategies are different arithmetic

"A second GPU buys capacity rather than speed" is true of a **layer split** and
false of **tensor parallelism**, and modelling only the first made vLLM look
worse than it is.

| | Layer split | Tensor parallel |
|---|---|---|
| Who | llama.cpp, Ollama | vLLM, SGLang, ExLlamaV2, TensorRT-LLM |
| Where the weights go | whole layers per card | every matrix sharded across all cards |
| Effective bandwidth | **harmonic** mean — one card at a time, so for identical cards it is one card's | **sum** — all cards read their shard at once |
| Prefill FLOPs | one card's | all cards' |
| Interconnect | nothing crosses it per token | an all-reduce per layer per token |

Two identical 3090s therefore decode at ~936 GB/s under llama.cpp and ~1872
GB/s under vLLM, and that is the difference the estimate now shows. `-parallel`
overrides it; the default is what the chosen runtime actually does.

The all-reduce is modelled as a ring: `2(N-1)/N × hidden × 2 bytes × batch` per
layer per token, over `-interconnect` (default 32 GB/s, one direction of PCIe
4.0 x16). On two desktop cards at batch 1 that is tens of microseconds against
a decode step of tens of milliseconds — **nearly free, and the model says so
rather than inflating it**. It stops being free as the card count and the batch
grow, which is when people notice, so the estimate warns when the interconnect
comes within 4× of being the limit.

### Speculative decoding is a trade, not a speedup

A draft model proposes `K` tokens, the target verifies all `K` in one forward
pass, and everything up to the first rejection is kept. With acceptance
probability `a` the expected yield per step is the geometric sum
`(1 − a^(K+1)) / (1 − a)`, capped at `K+1` — the target's own token is free when
every proposal is accepted.

Both passes are bandwidth-bound, so their relative cost is just the ratio of
weights read, and the multiplier is `E[accepted] / (1 + K·r)`. **It goes below
1** when the draft is large or rarely accepted, and reporting that is the point:
a badly matched pair is slower than not using it at all.

The draft is resident for the whole run, so its weights *and its own KV cache*
are added to the memory total. Counting the speedup without counting the memory
is how speculative decoding gets recommended onto a card it no longer fits on.

The default acceptance rate is **0.7**, the middle of the 0.6–0.8 range
Leviathan et al. (2023), *Fast Inference from Transformers via Speculative
Decoding*, measure for a draft from the same family as its target. It is printed
in the output rather than buried, because the answer is more sensitive to it
than to anything else in the estimate — override with `-acceptance`.

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

Estimates from bandwidth and compute, not measurements.

**The accuracy band is unknown, and this document used to claim otherwise.** It
said "usually within about 20%", which was an assertion about arithmetic rather
than a result: exactly ONE end-to-end number here has ever been held up against
a stopwatch. On that one — Qwen3-30B-A3B at Q4_K_M, 26 of 48 layers on an RTX
3060 — the tool predicts 2.8 tok/s against a measured 2.4, and the measured
figure is explicitly a floor because the machine was still swapping. One point
cannot distinguish a systematic bias from a machine having a bad afternoon.

`llm-fit validate` is how that stops being true. It reads
`bench/observations.tsv` — one row per measured run, with the machine and the
settings written down beside the number — and prints predicted against
measured, per row and as a geometric mean:

```console
$ llm-fit validate
model                  quant    engine     predicted  measured    ratio
------------------------------------------------------------------------
Qwen/Qwen3-30B-A3B     q4_k_m   llama.cpp        2.8       2.4    1.18x

1 observation(s) compared
geometric mean ratio  1.18x  (>1 means the tool is optimistic)

NOT AN ACCURACY BAND. 1 observation(s) cannot separate a systematic
bias from one machine that was swapping...
```

It refuses to state a band below three observations, and says so rather than
quoting a figure — the false precision the old sentence had. The mean is
geometric because these are ratios: a 2x overestimate and a 2x underestimate
have to cancel, and an arithmetic mean reports them as a 25% optimistic bias
that is not there. Above three, a mean ratio over 1.3 is called out as a
**systematic overestimate**, which is the direction that matters: an
overestimate ranks a model first on a machine where it is unusable, while an
underestimate merely hides one that would have worked.

Adding a row is a stopwatch and four numbers from `llm-fit detect`. Two more
runs on different hardware and the band above stops being a blank.

Three things push the estimates off:

- A GPU not in the spec table has no known bandwidth. It is flagged, and the
  speed figures that follow are guesses.
- Prefix caching and batch-of-one assumptions move real throughput. Speculative
  decoding is now modelled (`-draft`), but on an assumed acceptance rate rather
  than a measured one, and the acceptance rate is the whole answer.
- **An MoE with experts on the CPU is the one case measured against a stopwatch,
  and it needed its own constant.** Scaling bytes by the active fraction is
  right on the GPU and wrong on the CPU, where each layer re-gathers its experts
  every token. Under a single shared efficiency factor, Qwen3-30B-A3B on a 12GB
  card predicted 19 tok/s and delivered 2.4 — enough to rank it the best option
  on a machine it is unusable on. `cpuGatherEfficiency` in `internal/fit/speed.go`
  now separates the two, but it is fitted to one machine. Treat MoE offload
  figures as the least trustworthy numbers here until there are more.
- The quality ranking (`Score`) is a judgement, not a measurement. It encodes
  that a 32B at Q4_K_M beats an 8B at Q8_0, and that neither is beaten by a
  109B at 1.75 bits. Those trades are pinned in `advisor_test.go`.
