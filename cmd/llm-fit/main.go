// llm-fit — which models this machine can actually run, and how well.
//
// This file is the command line: the flags, the validation that has to happen
// before anything divides by a value, and the dispatch. What each verb does
// lives in commands.go, what it looks like in render.go.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/fabiocicerchia/llm-fit/internal/advisor"
	"github.com/fabiocicerchia/llm-fit/internal/fit"
	"github.com/fabiocicerchia/llm-fit/internal/hw"
	"github.com/fabiocicerchia/llm-fit/internal/quant"
)

const usage = `llm-fit — which LLMs this machine can run, and how fast

  llm-fit detect                 what hardware is here, and what it implies
  llm-fit suggest                models that run well, best first
  llm-fit check <model>          every quantization and runtime for one model
  llm-fit check <file.gguf>      read the shape and quant from a local GGUF file
  llm-fit engines                runtime support matrix for this machine
  llm-fit models                 the built-in catalogue

Flags
  -ctx N              context length to plan for (default 8192)
  -kv TYPE            KV cache precision: f16, q8_0, q4_0 (default f16)
  -batch N            concurrent sequences (default 1)
  -engine NAME        restrict to one runtime
  -serving            optimise for concurrent throughput, not chat latency
  -min LEVEL          unusable|sluggish|usable|good|excellent (default usable)
  -min-quality N      quantization quality floor 0-100 (default 55: no sub-3-bit)
  -top N              how many to list (default 12)
  -hf                 read the architecture from Hugging Face instead of the
                      built-in catalogue: llm-fit check -hf Qwen/Qwen3-14B
  -json               machine-readable output
  -gpu NAME           plan for a different card ("RTX 4090", "A100 80GB", "M4 Max"):
                      takes its VRAM, bandwidth and compute from the spec table
  -vram GiB           override VRAM only, keeping the detected bandwidth
  -ram GiB            override detected system RAM
  -ram-bandwidth GBs  override system RAM bandwidth

Speed figures are estimates from memory bandwidth and compute, not measurements.
They are usually within about 20% on hardware in the built-in table; treat them
as "this is the right ballpark" rather than a benchmark.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(2)
	}

	fs := flag.NewFlagSet("llm-fit", flag.ExitOnError)
	ctx := fs.Int("ctx", 8192, "")
	kv := fs.String("kv", "f16", "")
	batch := fs.Int("batch", 1, "")
	engineName := fs.String("engine", "", "")
	serving := fs.Bool("serving", false, "")
	minLevel := fs.String("min", "usable", "")
	minQuality := fs.Int("min-quality", 0, "")
	top := fs.Int("top", 12, "")
	asJSON := fs.Bool("json", false, "")
	useHF := fs.Bool("hf", false, "")
	gpuName := fs.String("gpu", "", "")
	vramOverride := fs.Float64("vram", 0, "")
	ramOverride := fs.Float64("ram", 0, "")
	ramBW := fs.Float64("ram-bandwidth", 0, "")
	fs.Usage = func() { fmt.Print(usage) }

	cmd := os.Args[1]
	positional := parsePositional(fs, os.Args[2:])

	// Validate before anything divides by it. An unknown -kv silently fell back
	// to 2.0 bytes at three separate call sites, so "-kv fp16" or "-kv Q8_0"
	// quietly sized every cache as f16 and reported a context ceiling roughly
	// half of what the flag asked for. Wrong answers are worse than no answer.
	if _, ok := quant.KVCacheBytesPerElement[*kv]; !ok {
		fmt.Fprintf(os.Stderr, "unknown -kv %q; valid values are %s\n",
			*kv, strings.Join(quant.KVTypes(), ", "))
		os.Exit(2)
	}

	machine := hw.Detect()
	applyOverrides(&machine, *gpuName, *vramOverride, *ramOverride, *ramBW)

	req := advisor.Request{
		Ctx: *ctx, KVType: *kv, Batch: *batch, Serving: *serving,
		MinVerdict: parseVerdict(*minLevel),
		MinQuality: *minQuality,
	}
	if *engineName != "" {
		req.Engines = []string{*engineName}
	}

	switch cmd {
	case "detect":
		cmdDetect(machine, *asJSON)
	case "suggest":
		cmdSuggest(machine, req, *top, *asJSON)
	case "check":
		if len(positional) == 0 {
			fmt.Fprintln(os.Stderr, "check needs a model name, e.g. llm-fit check qwen3-8b")
			os.Exit(2)
		}
		cmdCheck(machine, req, strings.Join(positional, " "), *asJSON, *useHF)
	case "engines":
		cmdEngines(machine)
	case "models":
		cmdModels(*asJSON)
	default:
		fmt.Print(usage)
		os.Exit(2)
	}
}

// parsePositional peels the non-flag arguments out of args, in order.
//
// flag.Parse stops at the first non-flag argument, so a single pass would
// silently ignore everything after the model name — `check qwen3-8b -ctx 32768`
// would plan for the default context and say nothing. Parse repeatedly, taking
// one positional off the front each time.
func parsePositional(fs *flag.FlagSet, args []string) []string {
	var positional []string
	for len(args) > 0 {
		if err := fs.Parse(args); err != nil {
			os.Exit(2)
		}
		args = fs.Args()
		if len(args) > 0 {
			positional = append(positional, args[0])
			args = args[1:]
		}
	}
	return positional
}

func applyOverrides(m *hw.Machine, gpuName string, vram, ram, ramBW float64) {
	// Naming a card replaces the whole device. Overriding VRAM alone would
	// otherwise model "my RTX 3060, but with 80GB" — the memory of an A100 at
	// the bandwidth of a 3060, which is nobody's hardware and produces decode
	// figures 5x too low.
	if gpuName != "" {
		spec, ok := hw.Lookup(gpuName)
		if !ok {
			fmt.Fprintf(os.Stderr, "unknown GPU %q — pass -vram and -ram-bandwidth instead\n", gpuName)
			os.Exit(2)
		}
		b := int64(spec.VRAMGiB * (1 << 30))
		m.GPUs = []hw.GPU{{
			Name: gpuName, Vendor: spec.Vendor, TotalBytes: b, FreeBytes: b,
			BandwidthGBs: spec.BandwidthGBs, TFLOPS: spec.TFLOPS,
			ComputeCapability: spec.ComputeCapability,
		}}
	}
	if vram > 0 {
		b := int64(vram * (1 << 30))
		if len(m.GPUs) == 0 {
			m.GPUs = []hw.GPU{{Name: "GPU (specified)", Vendor: hw.NVIDIA, BandwidthGBs: 900, TFLOPS: 100, Estimated: true}}
		}
		// Override applies to the whole GPU pool, split evenly.
		per := b / int64(len(m.GPUs))
		for i := range m.GPUs {
			m.GPUs[i].TotalBytes, m.GPUs[i].FreeBytes = per, per
		}
	}
	if ram > 0 {
		b := int64(ram * (1 << 30))
		m.RAMTotal, m.RAMFree = b, b
	}
	if ramBW > 0 {
		m.RAMBandwidth, m.RAMEstimated = ramBW, false
	}
}

func parseVerdict(s string) fit.Verdict {
	switch strings.ToLower(s) {
	case "unusable":
		return fit.Unusable
	case "sluggish":
		return fit.Sluggish
	case "good":
		return fit.Good
	case "excellent":
		return fit.Excellent
	}
	return fit.Usable
}

// isFile reports whether the query names a readable file rather than a model
// id. Directories do not count: "check ./models" is a mistake worth an error
// about an unknown model, not an attempt to parse a directory as GGUF.
func isFile(query string) bool {
	st, err := os.Stat(query)
	return err == nil && st.Mode().IsRegular()
}
