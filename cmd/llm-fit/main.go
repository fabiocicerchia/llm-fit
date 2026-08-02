// llm-fit — which models this machine can actually run, and how well.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/advisor"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/arch"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/catalog"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/engine"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/fit"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/hfapi"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/hw"
)

const usage = `llm-fit — which LLMs this machine can run, and how fast

  llm-fit detect                 what hardware is here, and what it implies
  llm-fit suggest                models that run well, best first
  llm-fit check <model>          every quantization and runtime for one model
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
	// flag.Parse stops at the first non-flag argument, so a single pass would
	// silently ignore everything after the model name — `check qwen3-8b -ctx
	// 32768` would plan for the default context and say nothing. Parse
	// repeatedly, peeling off one positional each time.
	var positional []string
	rest := os.Args[2:]
	for len(rest) > 0 {
		if err := fs.Parse(rest); err != nil {
			os.Exit(2)
		}
		rest = fs.Args()
		if len(rest) > 0 {
			positional = append(positional, rest[0])
			rest = rest[1:]
		}
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

func cmdDetect(m hw.Machine, asJSON bool) {
	if asJSON {
		emit(m)
		return
	}
	fmt.Printf("%s/%s, %d cores\n", m.OS, m.Arch, m.CPUCores)
	if m.CPUModel != "" {
		fmt.Printf("  cpu    %s\n", m.CPUModel)
	}
	fmt.Printf("  ram    %s free of %s", gib(m.RAMFree), gib(m.RAMTotal))
	if m.RAMBandwidth > 0 {
		fmt.Printf(" @ %.0f GB/s", m.RAMBandwidth)
		switch {
		case m.RAMMeasured:
			fmt.Print(" (measured)")
		case m.RAMEstimated:
			fmt.Print(" (assumed)")
		}
	}
	fmt.Println()

	if len(m.GPUs) == 0 {
		fmt.Println("  gpu    none detected — CPU inference only, expect single-digit tokens/sec above 7B")
	}
	for _, g := range m.GPUs {
		fmt.Printf("  gpu    %s — %s free of %s", g.Name, gib(g.FreeBytes), gib(g.TotalBytes))
		if g.BandwidthGBs > 0 {
			fmt.Printf(" @ %.0f GB/s", g.BandwidthGBs)
		}
		if g.ComputeCapability > 0 {
			fmt.Printf(", cc %.1f", g.ComputeCapability)
		}
		if g.Estimated {
			fmt.Print("  [not in the spec table: speed estimates unreliable]")
		}
		fmt.Println()
	}
	if m.UnifiedMem {
		fmt.Println("\n  Unified memory: the GPU figure above is the wired limit (~75-80% of RAM),")
		fmt.Println("  not a separate pool. Exceeding it swaps rather than failing.")
	}
	for _, w := range m.Warnings {
		fmt.Printf("\n  ! %s\n", w)
	}
}

func cmdSuggest(m hw.Machine, req advisor.Request, top int, asJSON bool) {
	opts := advisor.Suggest(m, req)
	if asJSON {
		emit(opts)
		return
	}
	if len(opts) == 0 {
		fmt.Printf("Nothing in the catalogue reaches %q on this machine at %s context.\n",
			req.MinVerdict, thousands(req.Ctx))
		fmt.Println("Try a shorter -ctx, a quantized KV cache (-kv q8_0), or -min sluggish to see what merely fits.")
		return
	}
	if len(opts) > top {
		opts = opts[:top]
	}

	fmt.Printf("Context %s, KV %s%s. Ranked by capability among plans that run %s or better.\n\n",
		thousands(req.Ctx), req.KVType, servingNote(req.Serving), req.MinVerdict)

	fmt.Printf("%-34s %-11s %-9s %8s %9s %7s  %s\n",
		"MODEL", "RUNTIME", "QUANT", "SIZE", "DECODE", "CTXMAX", "")
	for _, o := range opts {
		e := o.Estimate
		note := verdictMark(e.Verdict)
		if req.Serving && e.Concurrency > 0 {
			note += fmt.Sprintf("  %d concurrent", e.Concurrency)
		} else if !e.FullyOnGPU && e.LayersOnGPU < e.TotalLayers {
			note += fmt.Sprintf("  %d/%d layers on GPU", e.LayersOnGPU, e.TotalLayers)
		}
		fmt.Printf("%-34s %-11s %-9s %8s %6.0f/s %7s  %s\n",
			truncate(o.Model.Name, 34), o.Engine.Name, o.Format.Name,
			gib(e.WeightBytes+e.KVBytes), e.DecodeTPS, thousands(e.MaxCtxAtMem), note)
	}

	best := opts[0]
	fmt.Printf("\nStart here: %s on %s at %s\n", best.Model.Name, best.Engine.Name, best.Format.Name)
	if best.Format.Notes != "" {
		fmt.Printf("  %s\n", best.Format.Notes)
	}
	for _, r := range best.Estimate.Reasons {
		fmt.Printf("  %s\n", r)
	}
	fmt.Printf("  %s\n", best.Engine.InstallHint)
}

func cmdCheck(m hw.Machine, req advisor.Request, query string, asJSON, useHF bool) {
	var model arch.Model
	var ok bool
	if useHF {
		fetched, err := hfapi.Fetch(query)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		model, ok = fetched, true
	} else {
		model, ok = catalog.Find(query)
	}
	if !ok {
		hits := catalog.Matches(query)
		if len(hits) == 0 {
			fmt.Fprintf(os.Stderr, "no model matching %q. Try: llm-fit models\n", query)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "%q matches several models:\n", query)
		for _, h := range hits {
			fmt.Fprintf(os.Stderr, "  %s\n", h.ID)
		}
		os.Exit(1)
	}

	opts := advisor.Inspect(model, m, req)
	if asJSON {
		emit(opts)
		return
	}

	fmt.Printf("%s\n  %s parameters", model.Name, params(model.Params))
	if model.IsMoE() {
		fmt.Printf(", %s active per token (%d of %d experts)", params(model.Active()), model.ExpertsActive, model.Experts)
	}
	fmt.Printf("\n  %d layers, %d heads / %d KV heads", model.Layers, model.Heads, model.KVHeads)
	if model.Attention == arch.MLA {
		fmt.Print(", MLA compressed cache")
	}
	fmt.Printf("\n  %s vocab, trained to %s context\n", thousands(model.Vocab), thousands(model.MaxCtx))
	if model.Notes != "" {
		fmt.Printf("  %s\n", model.Notes)
	}

	kvAt := fit.KVBytes(model, req.Ctx, req.Batch, req.KVType)
	fmt.Printf("\nKV cache at %s context, %s: %s", thousands(req.Ctx), req.KVType, gib(kvAt))
	if alt := fit.KVBytes(model, req.Ctx, req.Batch, "q8_0"); req.KVType == "f16" && alt < kvAt {
		fmt.Printf("  (%s at q8_0 — usually free quality-wise)", gib(alt))
	}
	fmt.Println()
	fmt.Println()

	fmt.Printf("%-11s %-9s %8s %8s %9s %9s  %s\n", "RUNTIME", "QUANT", "WEIGHTS", "TOTAL", "DECODE", "PREFILL", "")
	for _, o := range opts {
		e := o.Estimate
		status := verdictMark(e.Verdict)
		if !e.Fits {
			status = "does not fit"
			if len(e.Reasons) > 0 {
				status = e.Reasons[0]
			}
			fmt.Printf("%-11s %-9s %8s %8s %9s %9s  %s\n",
				o.Engine.Name, o.Format.Name, gib(e.WeightBytes), "—", "—", "—", status)
			continue
		}
		if !e.FullyOnGPU && e.LayersOnGPU < e.TotalLayers {
			status += fmt.Sprintf("  %d/%d layers on GPU", e.LayersOnGPU, e.TotalLayers)
		}
		fmt.Printf("%-11s %-9s %8s %8s %6.0f/s %6.0f/s  %s\n",
			o.Engine.Name, o.Format.Name, gib(e.WeightBytes),
			gib(e.WeightBytes+e.KVBytes+e.OverheadBytes), e.DecodeTPS, e.PrefillTPS, status)
	}
}

func cmdEngines(m hw.Machine) {
	for _, e := range engine.All {
		ok, why := e.RunsOn(m)
		mark := "yes"
		if !ok {
			mark = "no"
		}
		fmt.Printf("%-14s %-4s %s\n", e.Name, mark, e.Summary)
		if !ok {
			fmt.Printf("%18s %s\n", "", why)
		} else {
			fmt.Printf("%18s formats: %s\n", "", families(e))
		}
	}
}

func cmdModels(asJSON bool) {
	all := catalog.All()
	if asJSON {
		emit(all)
		return
	}
	for _, m := range all {
		extra := ""
		if m.IsMoE() {
			extra = fmt.Sprintf("  (MoE, %s active)", params(m.Active()))
		}
		fmt.Printf("%-46s %8s%s\n", m.ID, params(m.Params), extra)
	}
}

// --- helpers -----------------------------------------------------------------

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

func verdictMark(v fit.Verdict) string {
	switch v {
	case fit.Excellent:
		return "excellent"
	case fit.Good:
		return "good"
	case fit.Usable:
		return "usable"
	case fit.Sluggish:
		return "sluggish"
	}
	return "unusable"
}

func servingNote(s bool) string {
	if s {
		return ", serving"
	}
	return ""
}

func families(e engine.Engine) string {
	parts := make([]string, 0, len(e.Formats))
	for _, f := range e.Formats {
		parts = append(parts, string(f))
	}
	return strings.Join(parts, ", ")
}

func gib(b int64) string {
	if b <= 0 {
		return "—"
	}
	g := float64(b) / (1 << 30)
	if g < 1 {
		return fmt.Sprintf("%.0f MiB", float64(b)/(1<<20))
	}
	return fmt.Sprintf("%.1f GiB", g)
}

func params(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.0fM", float64(n)/1e6)
	}
	return fmt.Sprintf("%d", n)
}

func thousands(n int) string {
	switch {
	case n >= 1000000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%dk", n/1000)
	}
	return fmt.Sprintf("%d", n)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func emit(v any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}
