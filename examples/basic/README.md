# Basic Example

What it shows: the same model on the same card is a different answer at 8k
context than at 128k — because the KV cache is not a rounding error.

Nothing to install beyond the binary, and nothing to download.

## Run

```sh
llm-fit detect
llm-fit suggest
```

`detect` measures host memory bandwidth (a few hundred milliseconds, ~1.5GB
transiently). That number cannot be read without root and every CPU-offload
estimate divides by it, so it is measured rather than looked up in a table
keyed on the CPU — the same Ryzen runs one DDR4 stick or four DDR5 ones, a 4×
spread.

## Watch context change the answer

```sh
llm-fit suggest -ctx 8192
llm-fit suggest -ctx 131072
```

Models drop off the list between those two runs, and the ones that survive get
slower. At 128k an 8B model's KV cache exceeds its own weights — decode reads
the whole cache for every token, so generation slows as the conversation grows.

Then try paying for it in cache precision instead of model size:

```sh
llm-fit suggest -ctx 131072 -kv q8_0
```

Usually a better trade than dropping a quantization level. That is the sort of
comparison the tool exists to make cheap.

## Why the tool refuses to just answer "does it fit"

```sh
llm-fit check qwen3-30b
```

Every quantization × runtime for one model, with a decode estimate on each. A
70B at Q4_K_M *fits* on a 12GB card with enough system RAM — llama.cpp will
load it, spill sixty layers into DDR, and generate slower than you read. The
speed estimate is what decides whether a plan is suggested at all.

Look at the MoE entries while you are there: Qwen3-30B-A3B occupies 30B of
memory and reads ~3.3B per token. Memory of a 30B, speed of a 3B — which is
why it often wins on a small card.

## Find out why an engine is missing

```sh
llm-fit engines
```

```text
llama.cpp     available
Ollama        available
ExLlamaV2     unavailable — NVIDIA only
MLX           unavailable — Apple silicon only
vLLM          available    (no useful CPU offload: a 70B is out of reach here)
```

The constraints are modelled, not just listed. vLLM and SGLang do not usefully
offload, so on a 12GB card a 70B is out of reach whatever the system RAM —
`llm-fit` says so rather than printing a number for a configuration that will
fail to load.

## Plan for a card you do not own

```sh
llm-fit suggest -gpu "RTX 4090"
llm-fit suggest -gpu "A100 80GB" -ram 256 -serving
```

`-gpu` takes VRAM, bandwidth **and** compute together from the spec table.
Overriding `-vram` alone would model your card with someone else's memory
capacity, which is nobody's hardware.

`-serving` switches the ranking from single-stream latency to how many
concurrent requests the leftover memory holds, and drops the runtimes that are
not built for it. It is a different question and it gets a different answer.

## Before you trust a number

These are estimates from bandwidth and compute, not measurements — usually
within about 20% on hardware in the spec table. Three things push them off:

- A GPU not in the table has no known bandwidth. It is flagged, and every speed
  figure after that is a guess.
- Speculative decoding, prefix caching and batch-of-one assumptions all move
  real throughput.
- `Score` is a judgement about quality, not a measurement.

And the honest caveat from the README: the **memory** model is calibrated
against published GGUF file sizes and asserted to within 3%, but the **speed**
estimates have not been checked against a stopwatch. Use them to choose between
options, not to predict a benchmark.
