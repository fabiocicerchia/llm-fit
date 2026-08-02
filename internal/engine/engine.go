// Package engine describes what each runtime can actually load and where it
// can run, which is half the advice: a plan that is arithmetically sound and
// names a format the runtime cannot read is not a plan.
//
// The differences that matter are not performance trivia. vLLM will not
// meaningfully offload to system RAM, so on a 12GB card a 70B is simply out of
// reach no matter how much DDR is installed — while llama.cpp will run it, and
// run it at two tokens per second. ExLlamaV2 is NVIDIA-only. MLX is Apple-only.
// FP8 needs Ada or Hopper silicon, and asking for it on an A100 fails at load.
package engine

import (
	"strconv"
	"strings"

	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/fit"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/hw"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/quant"
)

type Engine struct {
	fit.Engine
	Formats []quant.Family
	// Vendors that can run it at all. Empty means anywhere.
	Vendors []hw.Vendor
	// MinComputeCapability gates NVIDIA-only features; 0 means no constraint.
	MinComputeCapability float64
	Serving              bool // built for concurrent requests rather than one chat
	Summary              string
	InstallHint          string
}

var All = []Engine{
	{
		Engine: fit.Engine{
			Name: "llama.cpp", MBU: 0.80, MFU: 0.32,
			RuntimeOverheadBytes: 250 * fit.MiB,
			CanOffloadCPU:        true, MemoryFraction: 0.95,
		},
		Formats:     []quant.Family{quant.GGUF},
		Summary:     "Runs anywhere, splits layers between GPU and CPU, best single-user speed per gigabyte.",
		InstallHint: "brew install llama.cpp  # or: https://github.com/ggml-org/llama.cpp",
	},
	{
		Engine: fit.Engine{
			Name: "ollama", MBU: 0.76, MFU: 0.30,
			RuntimeOverheadBytes: 400 * fit.MiB,
			CanOffloadCPU:        true, MemoryFraction: 0.92,
		},
		Formats:     []quant.Family{quant.GGUF},
		Summary:     "llama.cpp with model management and automatic layer splitting. Slightly slower, much less to configure.",
		InstallHint: "curl -fsSL https://ollama.com/install.sh | sh",
	},
	{
		Engine: fit.Engine{
			Name: "vLLM", MBU: 0.72, MFU: 0.55,
			// A CUDA context, the PyTorch runtime and captured CUDA graphs,
			// before a single weight is loaded.
			RuntimeOverheadBytes: 2200 * fit.MiB,
			CanOffloadCPU:        false, MemoryFraction: 0.90,
		},
		Formats:              []quant.Family{quant.AWQ, quant.GPTQ, quant.FP8, quant.BnB, quant.Native},
		Vendors:              []hw.Vendor{hw.NVIDIA, hw.AMD},
		MinComputeCapability: 7.0,
		Serving:              true,
		Summary:              "Throughput serving with paged attention. Everything must be in VRAM; in exchange it holds many concurrent requests.",
		InstallHint:          "pip install vllm",
	},
	{
		Engine: fit.Engine{
			Name: "SGLang", MBU: 0.73, MFU: 0.57,
			RuntimeOverheadBytes: 2200 * fit.MiB,
			CanOffloadCPU:        false, MemoryFraction: 0.90,
		},
		Formats:              []quant.Family{quant.AWQ, quant.GPTQ, quant.FP8, quant.Native},
		Vendors:              []hw.Vendor{hw.NVIDIA, hw.AMD},
		MinComputeCapability: 7.5,
		Serving:              true,
		Summary:              "Like vLLM, with prefix caching that pays off when many requests share a long system prompt.",
		InstallHint:          "pip install \"sglang[all]\"",
	},
	{
		Engine: fit.Engine{
			Name: "ExLlamaV2", MBU: 0.85, MFU: 0.45,
			RuntimeOverheadBytes: 900 * fit.MiB,
			CanOffloadCPU:        false, MemoryFraction: 0.93,
		},
		Formats:              []quant.Family{quant.EXL2, quant.GPTQ},
		Vendors:              []hw.Vendor{hw.NVIDIA},
		MinComputeCapability: 7.5,
		Summary:              "The fastest single-user option on consumer NVIDIA, and EXL2 lets you pick the exact bits per weight.",
		InstallHint:          "pip install exllamav2",
	},
	{
		Engine: fit.Engine{
			Name: "MLX", MBU: 0.78, MFU: 0.40,
			RuntimeOverheadBytes: 300 * fit.MiB,
			CanOffloadCPU:        true, MemoryFraction: 0.90,
		},
		Formats:     []quant.Family{quant.MLX},
		Vendors:     []hw.Vendor{hw.Apple},
		Summary:     "Apple silicon native. Unified memory means no split to reason about, and it beats llama.cpp's Metal backend on prefill.",
		InstallHint: "pip install mlx-lm",
	},
	{
		Engine: fit.Engine{
			Name: "TensorRT-LLM", MBU: 0.88, MFU: 0.68,
			RuntimeOverheadBytes: 2600 * fit.MiB,
			CanOffloadCPU:        false, MemoryFraction: 0.90,
		},
		Formats:              []quant.Family{quant.AWQ, quant.FP8, quant.Native},
		Vendors:              []hw.Vendor{hw.NVIDIA},
		MinComputeCapability: 8.0,
		Serving:              true,
		Summary:              "Fastest on NVIDIA if you are willing to compile an engine per model per GPU per batch shape.",
		InstallHint:          "pip install tensorrt-llm",
	},
}

func ByName(name string) (Engine, bool) {
	for _, e := range All {
		if strings.EqualFold(e.Name, name) {
			return e, true
		}
	}
	return Engine{}, false
}

// Supports reports whether the engine can load the format.
func (e Engine) Supports(f quant.Format) bool {
	for _, fam := range e.Formats {
		if fam == f.Family {
			return true
		}
	}
	return false
}

// RunsOn reports whether the machine can run the engine at all, and why not
// when it cannot. The reason is the useful part: "vLLM needs an NVIDIA or AMD
// GPU" is advice, a missing row in a table is not.
func (e Engine) RunsOn(m hw.Machine) (bool, string) {
	if len(e.Vendors) == 0 {
		return true, ""
	}
	var best hw.Vendor = hw.Unknown
	var cc float64
	for _, g := range m.GPUs {
		for _, v := range e.Vendors {
			if g.Vendor == v {
				best = v
				if g.ComputeCapability > cc {
					cc = g.ComputeCapability
				}
			}
		}
	}
	if best == hw.Unknown {
		return false, e.Name + " needs " + vendorList(e.Vendors) + "; this machine has none"
	}
	if best == hw.NVIDIA && e.MinComputeCapability > 0 && cc > 0 && cc < e.MinComputeCapability {
		return false, e.Name + " needs compute capability " + trim(e.MinComputeCapability) + " or newer; this GPU is " + trim(cc)
	}
	return true, ""
}

func vendorList(vs []hw.Vendor) string {
	parts := make([]string, 0, len(vs))
	for _, v := range vs {
		parts = append(parts, string(v))
	}
	return strings.Join(parts, " or ")
}

// trim prints a compute capability without trailing zeros: 8.6 stays "8.6",
// 9.0 becomes "9". The shortest representation that round-trips is what a
// version-like number wants — truncating to one decimal by hand turned 8.6
// into "8.5", because 8.6-8 is 0.5999999999999996 in binary floating point.
func trim(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}
