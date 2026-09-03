package main

// Every line the tool prints for a human. Results go to stdout; the commands in
// commands.go decide what to show, this file decides how it looks.

import (
	"fmt"

	"github.com/fabiocicerchia/llm-fit/internal/advisor"
	"github.com/fabiocicerchia/llm-fit/internal/arch"
	"github.com/fabiocicerchia/llm-fit/internal/engine"
	"github.com/fabiocicerchia/llm-fit/internal/fit"
	"github.com/fabiocicerchia/llm-fit/internal/hw"
)

func printMachine(m hw.Machine) {
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
		printGPU(g)
	}
	if m.UnifiedMem {
		fmt.Println("\n  Unified memory: the GPU figure above is the wired limit (~75-80% of RAM),")
		fmt.Println("  not a separate pool. Exceeding it swaps rather than failing.")
	}
	for _, w := range m.Warnings {
		fmt.Printf("\n  ! %s\n", w)
	}
}

func printGPU(g hw.GPU) {
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

// printNothingFits says what to relax, because an empty table looks like a bug.
func printNothingFits(req advisor.Request) {
	fmt.Printf("Nothing in the catalogue reaches %q on this machine at %s context.\n",
		req.MinVerdict, thousands(req.Ctx))
	fmt.Println("Try a shorter -ctx, a quantized KV cache (-kv q8_0), or -min sluggish to see what merely fits.")
}

func printSuggestions(m hw.Machine, req advisor.Request, opts []advisor.Option) {
	// "capability" alone would be a lie: Score multiplies parameter count by
	// quantization quality and by a speed term, so a model that is bigger but
	// slower can and does lose.
	fmt.Printf("Context %s, KV %s%s. Ranked by capability and speed among plans that run %s or better.\n\n",
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
	fmt.Printf("  %s\n", best.Engine.HintFor(m.OS))
}

func printModelShape(model arch.Model) {
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
}

func printKVCache(model arch.Model, req advisor.Request) {
	kvAt := fit.KVBytes(model, req.Ctx, req.Batch, req.KVType)
	fmt.Printf("\nKV cache at %s context, %s: %s", thousands(req.Ctx), req.KVType, gib(kvAt))
	if alt := fit.KVBytes(model, req.Ctx, req.Batch, "q8_0"); req.KVType == "f16" && alt < kvAt {
		fmt.Printf("  (%s at q8_0 — usually free quality-wise)", gib(alt))
	}
	fmt.Println()
	fmt.Println()
}

func printPlanTable(opts []advisor.Option) {
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

// printEngine reports one runtime and, when it cannot run here, why not. The
// reason is the useful part: a missing row in a table is not advice.
func printEngine(e engine.Engine, runs bool, why string) {
	mark := "yes"
	if !runs {
		mark = "no"
	}
	fmt.Printf("%-14s %-4s %s\n", e.Name, mark, e.Summary)
	if !runs {
		fmt.Printf("%18s %s\n", "", why)
	} else {
		fmt.Printf("%18s formats: %s\n", "", families(e))
	}
}

func printCatalogueEntry(m arch.Model) {
	extra := ""
	if m.IsMoE() {
		extra = fmt.Sprintf("  (MoE, %s active)", params(m.Active()))
	}
	fmt.Printf("%-46s %8s%s\n", m.ID, params(m.Params), extra)
}
