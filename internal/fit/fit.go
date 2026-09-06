// Package fit is the arithmetic the tool exists for: does this model fit, and
// if it fits, is it fast enough to be worth running.
//
// Fitting is the easy half. The half people get wrong is that a model can fit
// and still be useless — a 70B spilling eight layers into system RAM fits, and
// generates at two tokens per second, which is slower than reading. So every
// answer here carries a speed estimate, and the speed estimate is what decides
// whether a suggestion is made at all.
//
// The physics, in one line each:
//
//	Decode is memory-bandwidth bound. Generating one token reads every active
//	weight and the whole KV cache exactly once, so tokens/sec is bandwidth
//	divided by bytes-read-per-token. Not FLOPs. A 4090 and an A100 differ far
//	less in single-stream decode than their compute suggests.
//
//	Prefill is compute bound. Processing a prompt is a matrix-matrix product
//	over many tokens at once, so it scales with FLOPs, and it is why a slow-
//	bandwidth card with tensor cores still ingests documents quickly.
package fit

import (
	"fmt"
	"math"
	"strconv"
)

// MiB and GiB are the units every memory figure in this package is in.
// Binary, not decimal: so is every number a driver or a GGUF header reports.
const (
	MiB = 1 << 20
	GiB = 1 << 30
)

const (
	// maxUbatch is the micro-batch every engine here caps at, so the activation
	// working set stops growing past it however large the batch.
	maxUbatch = 512

	// cpuPrefillShare: CPU prefill is roughly two orders of magnitude slower
	// than a GPU's, so CPU-resident layers contribute almost nothing rather
	// than pretending to help.
	cpuPrefillShare = 0.02
)

// EstimatePlan runs the whole calculation for one plan.
func EstimatePlan(p Plan, e Engine) Estimate {
	var est Estimate
	m := p.Model

	// A caller that says nothing gets the strategy the runtime would really
	// use, rather than whichever one happens to be the zero value.
	par := p.Parallelism
	if !p.ParallelismSet {
		par = e.DefaultParallelism
	}
	if gpuCount(p.Devices) < 2 {
		// With one card the distinction does not exist, and reporting a
		// strategy would suggest a choice that was never made.
		par = LayerSplit
	}
	est.Parallelism = par

	est.WeightBytes = WeightBytes(m, p.Format)
	est.KVBytes = KVBytes(m, p.Ctx, p.Batch, p.KVType)

	// A draft model is resident for the whole run: its weights and its own KV
	// cache come out of the same VRAM the target wanted. Counting the speedup
	// without counting the memory is how speculative decoding gets recommended
	// onto a card it no longer fits on.
	if p.Speculative != nil {
		d := *p.Speculative
		df := draftFormat(d, p.Format)
		dw := WeightBytes(d.Draft, df)
		dkv := KVBytes(d.Draft, p.Ctx, p.Batch, p.KVType)
		est.DraftBytes = dw + dkv
		est.WeightBytes += dw
		est.KVBytes += dkv
	}

	gpuTotal, cpuTotal := splitCapacity(p.Devices, e.MemoryFraction)

	// Decide the split: how many layers land on the GPU. Weights divide by
	// layer; the KV cache follows its layers.
	perLayerWeight := float64(est.WeightBytes) / math.Max(float64(m.Layers), 1)
	perLayerKV := float64(est.KVBytes) / math.Max(float64(m.Layers), 1)
	perLayer := perLayerWeight + perLayerKV

	hasGPU := gpuTotal > 0
	est.OverheadBytes = Overhead(m, e, p.Ctx, p.Batch, hasGPU)
	est.TotalBytes = est.WeightBytes + est.KVBytes + est.OverheadBytes

	usableGPU := float64(gpuTotal) - float64(est.OverheadBytes)
	layersFit := m.Layers
	if perLayer > 0 && usableGPU < float64(est.WeightBytes+est.KVBytes) {
		layersFit = int(math.Floor(usableGPU / perLayer))
	}
	layersFit = clamp(layersFit, 0, m.Layers)

	if layersFit < m.Layers && !e.CanOffloadCPU {
		// vLLM and friends need the whole thing resident. Partial is not a
		// slower option, it is not an option.
		est.Fits = false
		est.LayersOnGPU = layersFit
		est.Reasons = append(est.Reasons, e.Name+" needs the whole model in VRAM; it does not usefully offload to system RAM")
		est.Verdict = Unusable
		return est
	}

	est.LayersOnGPU = layersFit
	est.TotalLayers = m.Layers
	cpuLayers := m.Layers - layersFit
	est.GPUBytes = int64(float64(layersFit)*perLayer) + est.OverheadBytes
	est.CPUBytes = int64(float64(cpuLayers) * perLayer)
	est.FullyOnGPU = cpuLayers == 0 && hasGPU

	if est.CPUBytes > cpuTotal {
		est.Fits = false
		est.Reasons = append(est.Reasons, "does not fit even with system RAM")
		est.Verdict = Unusable
		return est
	}
	est.Fits = true

	gpuShare := float64(layersFit) / math.Max(float64(m.Layers), 1)
	cpuShare := 1 - gpuShare
	est.DecodeTPS = decodeTPS(p, e, est, gpuShare, cpuShare)
	est.PrefillTPS = prefillTPS(p, e, gpuShare, cpuShare, par)
	if par == TensorParallel {
		if reduce := allReduceSecondsPerToken(
			m, p.Batch, gpuCount(p.Devices), interconnect(p)); reduce > 0 {
			est.InterconnectTPS = 1 / reduce
		}
	}

	// What the other strategy would have given, so the delta is visible rather
	// than implied. This is the number that says whether a second card is worth
	// buying for *this* runtime.
	if gpuCount(p.Devices) > 1 {
		est.AltDecodeTPS = altDecodeTPS(p, e, est, gpuShare, cpuShare)
	}

	// Speculative decoding multiplies the decode rate, so it is applied to the
	// bandwidth-bound figure above rather than replacing it.
	if p.Speculative != nil {
		est.AcceptanceRate, est.AcceptedPerStep, est.Speedup = speculativeSpeedup(
			m, *p.Speculative, p.Format)
		est.DecodeTPS *= est.Speedup
		est.AltDecodeTPS *= est.Speedup
	}

	// Report the context the cache can reach in the memory the weights actually
	// occupy. A GPU-resident model could nominally cache far more by spilling
	// into system RAM, but a KV cache in DDR is read in full on every token, so
	// that number describes a configuration nobody would choose.
	capacity := gpuTotal
	if !est.FullyOnGPU {
		capacity = gpuTotal + cpuTotal
	}
	est.MaxCtxAtMem = maxContext(m, p, e, capacity)
	est.Concurrency = concurrency(m, p, e, gpuTotal, est.WeightBytes, est.OverheadBytes)

	est.Verdict = verdictFor(est.DecodeTPS)
	est.Reasons = append(est.Reasons, explain(est, p, e, cpuLayers, cpuTotal)...)
	return est
}

// cpuCachePressure is the share of free RAM the CPU-side weights must hold to
// stay resident. Past this, treat the plan as at risk rather than merely slow.
const cpuCachePressure = 0.7

func explain(est Estimate, p Plan, e Engine, cpuLayers int, cpuTotal int64) []string {
	var out []string
	if cpuLayers > 0 {
		out = append(out, layerClause(cpuLayers)+" running on the CPU: every token waits on system RAM, which is why this is slow")
	}
	// Fitting is not the same as staying resident. llama.cpp mmaps the weight
	// file, so CPU-side layers live in reclaimable page cache — the same pages
	// MemAvailable counted as free for everything else. When they are most of
	// what is free, ordinary memory pressure evicts them and every token starts
	// faulting back off the SSD. That is not a gentle slowdown: measured on a
	// 30GB box it was the difference between 1.4 and 2.4 tok/s, and the machine
	// spent the gap in sustained major faults with swap pinned at 100%.
	if est.CPUBytes > 0 && cpuTotal > 0 &&
		float64(est.CPUBytes) > cpuCachePressure*float64(cpuTotal) {
		out = append(out, "the CPU-side weights need most of the free RAM to stay cached; close other memory-hungry programs or expect it to fault off disk mid-generation")
	}
	if est.KVBytes > est.WeightBytes {
		out = append(out, "the KV cache is larger than the weights at this context — quantize the cache before dropping to a smaller quant")
	}
	if p.Ctx > p.Model.MaxCtx {
		out = append(out, "requested context exceeds what the model was trained for")
	}
	if est.FullyOnGPU && est.Verdict >= Good {
		out = append(out, "entirely in VRAM")
	}
	if n := gpuCount(p.Devices); n > 1 && est.AltDecodeTPS > 0 {
		alt := LayerSplit
		if est.Parallelism == LayerSplit {
			alt = TensorParallel
		}
		out = append(out, fmt.Sprintf(
			"%d GPUs, %s (%s default): ~%.0f tok/s here, ~%.0f under %s",
			n, est.Parallelism, e.Name, est.DecodeTPS, est.AltDecodeTPS, alt))
		// The thing people actually want to know before buying a second card.
		if est.Parallelism == LayerSplit && est.AltDecodeTPS > est.DecodeTPS*1.2 {
			out = append(out, "a layer split makes extra cards buy capacity, not speed — a tensor-parallel runtime would be faster on this hardware")
		}
		if est.Parallelism == TensorParallel && est.InterconnectTPS > 0 &&
			est.InterconnectTPS < est.DecodeTPS*4 {
			out = append(out, fmt.Sprintf(
				"the interconnect is close to being the limit here (~%.0f tok/s of all-reduce alone at %.0f GB/s) — NVLink would move it",
				est.InterconnectTPS, interconnect(p)))
		}
	}
	if p.Speculative != nil {
		verb := "gains"
		if est.Speedup < 1 {
			verb = "LOSES"
		}
		out = append(out, fmt.Sprintf(
			"speculative decoding %s %.2fx at an assumed %.0f%% acceptance rate (%.1f tokens per verify step); the draft also costs %s of VRAM",
			verb, est.Speedup, est.AcceptanceRate*100, est.AcceptedPerStep, humanBytes(est.DraftBytes)))
		if est.Speedup < 1 {
			out = append(out, "the draft is too expensive or too rarely accepted to pay for itself — try a smaller draft, or drop it")
		}
	}
	return out
}

// layerClause names the offloaded layers in a sentence, singular or plural.
// humanBytes renders a byte count the way the reasons read it.
func humanBytes(b int64) string {
	switch {
	case b >= int64(GiB):
		return fmt.Sprintf("%.1f GiB", float64(b)/GiB)
	case b >= int64(MiB):
		return fmt.Sprintf("%.0f MiB", float64(b)/MiB)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func layerClause(n int) string {
	if n == 1 {
		return "1 layer is"
	}
	return strconv.Itoa(n) + " layers are"
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
