# llm-fit

**Which LLMs this machine can actually run, and how fast.** Reads the hardware,
does the memory and bandwidth arithmetic, and recommends a model, a
quantization and a runtime — with an estimate of how many tokens per second
you will get.

```console
$ llm-fit suggest
Context 8k, KV f16. Ranked by capability among plans that run usable or better.

MODEL                              RUNTIME     QUANT         SIZE    DECODE  CTXMAX
Qwen3 30B-A3B (MoE)                llama.cpp   Q4_K_S    17.1 GiB     20/s     32k  good  29/48 layers on GPU
Qwen2.5 14B Instruct               ExLlamaV2   EXL2-4.0   8.6 GiB     33/s     14k  excellent
Phi-4 14B                          ExLlamaV2   EXL2-4.0   8.6 GiB     33/s     13k  excellent
Mistral Small 24B                  llama.cpp   IQ3_M     11.3 GiB     13/s     32k  usable  37/40 layers on GPU
Mistral Nemo 12B                   ExLlamaV2   EXL2-4.0   7.2 GiB     40/s     24k  excellent
```

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

## Use

```sh
llm-fit detect                    # what is here, and what it implies
llm-fit suggest                   # models that run well, best first
llm-fit suggest -ctx 32768 -kv q8_0
llm-fit suggest -serving          # optimise for concurrency
llm-fit check qwen3-30b           # every quant × runtime for one model
llm-fit check -hf Qwen/Qwen3-14B  # any model, read from Hugging Face
llm-fit engines                   # what this machine can run, and why not
llm-fit models                    # the built-in catalogue
```

`-batch N` for concurrent sequences, `-engine NAME` to restrict to one runtime,
`-top N` for list length, and `-min LEVEL` / `-min-quality N` for the floors
below which a result is not worth showing (default: `usable`, and no sub-3-bit).

Plan for hardware you do not have yet with `-gpu`, which takes VRAM, bandwidth
and compute together from the spec table — overriding VRAM alone would model
your card with someone else's memory capacity, which is nobody's hardware:

```sh
llm-fit suggest -gpu "RTX 4090"
llm-fit suggest -gpu "A100 80GB" -ram 256 -serving
llm-fit suggest -gpu "M4 Max"
```

`-vram`, `-ram` and `-ram-bandwidth` override individual figures. `-json` for
everything.

### System RAM bandwidth is measured, not assumed

It cannot be read without root, and it is what every CPU-offload estimate
divides by. A table keyed on the CPU would be guessing about the memory fitted —
the same Ryzen runs one DDR4 stick or four DDR5 ones, a 4× spread. So `llm-fit`
measures it: a few hundred milliseconds of multi-threaded copy over buffers far
larger than L3. On the machine this was written on that returns 43 GB/s against
a 51.2 GB/s theoretical peak, where the assumption would have been 80.

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

## Status

- [x] Memory model calibrated against published GGUF sizes
- [x] GQA, MLA and MoE handled as distinct cases, not scaling factors
- [x] Seven runtimes with their real constraints
- [x] Measured host memory bandwidth
- [x] NVIDIA, AMD, Apple and CPU-only detection
- [x] Any Hugging Face model via `-hf`
- [ ] **Validate the speed estimates against real runs.** Nothing here has been
      checked against a stopwatch; the arithmetic is sound and the constants are
      from published figures, but the end-to-end numbers are unverified.
- [ ] Read an actual GGUF header, so a specific file is measured rather than a
      format assumed
- [ ] Speculative decoding and draft-model pairs
- [ ] Multi-GPU tensor parallelism (currently modelled as layer split, which is
      right for llama.cpp and pessimistic for vLLM)

## Development

```sh
make test   # go test ./...
make lint   # vet + gofmt
make demo   # detect, then suggest, on this machine
```

No dependencies outside the standard library. `internal/fit` is the arithmetic
and holds most of the tests; `internal/advisor` is the ranking and holds the
rest.

## License

Apache-2.0 — see [LICENSE](LICENSE).
