# llm-fit

[![CI](https://github.com/fabiocicerchia/llm-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/fabiocicerchia/llm-fit/actions/workflows/ci.yml)
[![Code Quality](https://github.com/fabiocicerchia/llm-fit/actions/workflows/code-quality.yml/badge.svg)](https://github.com/fabiocicerchia/llm-fit/actions/workflows/code-quality.yml)
[![Security](https://github.com/fabiocicerchia/llm-fit/actions/workflows/security.yml/badge.svg)](https://github.com/fabiocicerchia/llm-fit/actions/workflows/security.yml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/fabiocicerchia/llm-fit/badge)](https://securityscorecards.dev/viewer/?uri=github.com/fabiocicerchia/llm-fit)
[![CI carbon](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/fabiocicerchia/llm-fit/gh-pages/badge.json)](.github/workflows/carbon-badge.yml)

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

## Install

```sh
go install github.com/fabiocicerchia/llm-fit/cmd/llm-fit@latest
```

Or from a checkout:

```sh
make build      # -> ./bin/
```

## Use

```sh
llm-fit detect                    # what is here, and what it implies
llm-fit suggest                   # models that run well, best first
llm-fit suggest -ctx 32768 -kv q8_0
llm-fit suggest -serving          # optimise for concurrency
llm-fit check qwen3-30b           # every quant × runtime for one model
llm-fit check -hf Qwen/Qwen3-14B  # any model, read from Hugging Face
llm-fit check ~/models/q4.gguf    # the file on disk: its own shape and quant
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

## Documentation

Full docs live in [`docs/`](docs/). Runnable examples live in [`examples/`](examples/).

## License

Apache-2.0 — see [LICENSE](LICENSE).
