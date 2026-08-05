# Getting Started

## Prerequisites

Go 1.24 or newer to build. Nothing else — there are no dependencies outside the
standard library, and no runtime requirements on the machine being inspected.

`nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) on `PATH` if you want GPUs detected;
without them the machine is treated as CPU-only and says so.

## Build

```sh
make build      # ./bin/llm-fit
```

## First run

```sh
./bin/llm-fit detect
```

This measures host memory bandwidth, which takes a few hundred milliseconds and
allocates around 1.5GB transiently. It is worth it: that number cannot be read
without root, and every CPU-offload speed estimate divides by it.

Then:

```sh
./bin/llm-fit suggest
```

## Common questions

**"Nothing reaches usable on this machine."** Try a shorter context, a quantized
KV cache, or lower the bar:

```sh
./bin/llm-fit suggest -ctx 4096 -kv q8_0
./bin/llm-fit suggest -min sluggish
```

**"Why is that model not listed?"** `suggest` only shows what runs acceptably.
To see every option for one model, including the ones that do not fit:

```sh
./bin/llm-fit check qwen3-8b
```

**"My model is not in the catalogue."** Read it from Hugging Face instead:

```sh
./bin/llm-fit check -hf mistralai/Mistral-Small-24B-Instruct-2501
```

Gated repositories (Llama, Gemma) need `HF_TOKEN` set after accepting the
licence.

**"I am buying hardware."** Plan against a card you do not own:

```sh
./bin/llm-fit suggest -gpu "RTX 4090"
./bin/llm-fit suggest -gpu "A100 80GB" -ram 256 -serving
```

## Development

```sh
make test   # go test ./...
make lint   # vet + gofmt
make demo   # detect, then suggest, on this machine
```

No dependencies outside the standard library. `internal/fit` is the arithmetic
and holds most of the tests; `internal/advisor` is the ranking and holds the
rest.
