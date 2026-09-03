package fit

// How fast: decode is bandwidth bound, prefill is compute bound, and a
// layer-split plan pays both rates for their share of the layers.

import (
	"math"

	"github.com/fabiocicerchia/llm-fit/internal/arch"
	"github.com/fabiocicerchia/llm-fit/internal/quant"
)

// CPU decode efficiency, as a fraction of measured memcpy bandwidth.
//
// Dense models stream: the CPU-resident layers are read one contiguous weight
// matrix after another. Fewer, wider cores and no coalesced access cost roughly
// half of what a GPU manages, hence 0.55.
//
// MoE models do not stream, and this is the single largest error in the model
// if you let them share a constant. Scaling bytes by the active fraction is
// correct on the GPU and badly optimistic on the CPU: routing is re-decided per
// layer per token, so every CPU layer performs a fresh gather of a handful of
// experts out of the full resident bank, with no reuse or prefetch between
// tokens, and per-expert call overhead dominates a matmul this small. The CPU
// side stops being bandwidth-bound at all.
//
// 0.08 is measured, not derived. Qwen3-30B-A3B at Q4_K_M, 26 of 48 layers on an
// RTX 3060 and 22 on a Ryzen 3700X with 43 GB/s memcpy bandwidth, 16k context
// with a q8_0 cache, ran at 2.4 tok/s — 3.3 GB/s effective, or 0.078 of
// measured. Under the old flat 0.55 the same plan predicted 19 tok/s: a 7x
// overestimate, and enough to rank that model first on a machine where it is
// unusable. It is a floor, not a best case; swap was still draining during the
// run that produced it.
//
// Caveats worth keeping honest: one machine is one data point, and it replaces
// a constant that had none. It is also applied to the KV term, which really
// does stream — folding that into a single number is what makes the constant
// reproduce the measurement, and splitting the two is not justified until there
// is more than one machine to fit against.
const (
	cpuStreamEfficiency = 0.55
	cpuGatherEfficiency = 0.08
)

// cpuEfficiency picks between them. Only reached when layers actually land on
// the CPU, so an MoE that fits entirely in VRAM is unaffected.
func cpuEfficiency(m arch.Model) float64 {
	if m.IsMoE() {
		return cpuGatherEfficiency
	}
	return cpuStreamEfficiency
}

// decodeTPS is the bandwidth-bound half, in tokens per second.
//
// One token reads every active weight once, plus the entire KV cache once. The
// cache term is not a rounding error: at 128k context an 8B model spends more
// time reading cache than weights, which is why long-context generation slows
// down as the conversation grows.
//
// A layer-split plan pays both rates, each for its share of the layers, and the
// host's is 10-40x lower — which is the whole reason CPU offload is slow.
func decodeTPS(p Plan, e Engine, est Estimate, gpuShare, cpuShare float64) float64 {
	m := p.Model
	activeFraction := float64(m.Active()) / math.Max(float64(m.Params), 1)
	bytesPerToken := float64(est.WeightBytes)*activeFraction + float64(est.KVBytes)

	gpuBW, cpuBW := bandwidths(p.Devices, est.Parallelism)
	var secondsPerToken float64
	if gpuShare > 0 && gpuBW > 0 {
		secondsPerToken += bytesPerToken * gpuShare / (e.MBU * gpuBW * 1e9)
	}
	if cpuShare > 0 && cpuBW > 0 {
		secondsPerToken += bytesPerToken * cpuShare / (cpuEfficiency(m) * cpuBW * 1e9)
	}
	// Tensor parallelism pays an all-reduce per layer per token. Added to the
	// bandwidth term rather than modelled as a separate ceiling, because the
	// two are serial: the card cannot start the next layer until the reduce for
	// this one has landed.
	if est.Parallelism == TensorParallel {
		secondsPerToken += allReduceSecondsPerToken(
			m, p.Batch, gpuCount(p.Devices), interconnect(p))
	}
	if secondsPerToken <= 0 {
		return 0
	}
	return 1 / secondsPerToken
}

// bytesReadPerToken is the weights-plus-cache figure both decode paths read.
func bytesReadPerToken(m arch.Model, est Estimate) float64 {
	activeFraction := float64(m.Active()) / math.Max(float64(m.Params), 1)
	return float64(est.WeightBytes)*activeFraction + float64(est.KVBytes)
}

// altDecodeTPS is the decode rate the *other* parallelism strategy would give,
// with everything else held equal. Reported so the choice is visible: under a
// layer split a second identical card changes nothing about speed, and under
// tensor parallelism it roughly doubles it, and no single number says that.
func altDecodeTPS(p Plan, e Engine, est Estimate, gpuShare, cpuShare float64) float64 {
	m := p.Model
	alt := TensorParallel
	if est.Parallelism == TensorParallel {
		alt = LayerSplit
	}
	bytesPerToken := bytesReadPerToken(m, est)
	gpuBW, cpuBW := bandwidths(p.Devices, alt)
	var spt float64
	if gpuShare > 0 && gpuBW > 0 {
		spt += bytesPerToken * gpuShare / (e.MBU * gpuBW * 1e9)
	}
	if cpuShare > 0 && cpuBW > 0 {
		spt += bytesPerToken * cpuShare / (cpuEfficiency(m) * cpuBW * 1e9)
	}
	if alt == TensorParallel {
		spt += allReduceSecondsPerToken(m, p.Batch, gpuCount(p.Devices), interconnect(p))
	}
	if spt <= 0 {
		return 0
	}
	return 1 / spt
}

// interconnect is the link bandwidth this plan pushes activations over.
func interconnect(p Plan) float64 {
	if p.InterconnectGBs > 0 {
		return p.InterconnectGBs
	}
	return DefaultInterconnectGBs
}

// draftFormat is the quantization the draft is stored at. A caller that names
// no format for the draft means "the same as the target" — the common case, and
// the one that silently dropped the draft's weights from the VRAM total when
// only speculativeSpeedup applied the fallback.
func draftFormat(s Speculative, target quant.Format) quant.Format {
	if s.Format.Name == "" {
		return target
	}
	return s.Format
}

// speculativeSpeedup returns the acceptance rate used, the expected tokens per
// verification step, and the resulting multiplier on decode throughput.
//
// One step is K draft forward passes plus one target pass that verifies all K
// proposals at once. Every token up to the first rejection is kept, so with
// acceptance probability a the expected yield is the geometric sum
//
//	E[accepted] = (1 - a^(K+1)) / (1 - a)
//
// capped at K+1 — the target's own token is free when every proposal is
// accepted. The step costs one target pass plus K draft passes, and both are
// bandwidth-bound, so their relative cost is just the ratio of weights read.
//
// The multiplier is therefore E[accepted] / (1 + K*r) with r that ratio. It
// goes *below* 1 when the draft is large or rarely accepted, which is the
// answer that matters: speculative decoding is a trade, and a badly matched
// pair is slower than not using it.
func speculativeSpeedup(target arch.Model, s Speculative, f quant.Format) (rate, accepted, speedup float64) {
	rate = s.AcceptanceRate
	if rate <= 0 {
		rate = DefaultAcceptanceRate
	}
	rate = math.Min(rate, 0.999) // a == 1 would divide by zero below
	k := s.Lookahead
	if k <= 0 {
		k = DefaultLookahead
	}

	accepted = (1 - math.Pow(rate, float64(k)+1)) / (1 - rate)
	accepted = math.Min(accepted, float64(k)+1)

	df := draftFormat(s, f)
	targetRead := float64(WeightBytes(target, f)) *
		float64(target.Active()) / math.Max(float64(target.Params), 1)
	draftRead := float64(WeightBytes(s.Draft, df)) *
		float64(s.Draft.Active()) / math.Max(float64(s.Draft.Params), 1)
	var ratio float64
	if targetRead > 0 {
		ratio = draftRead / targetRead
	}

	speedup = accepted / (1 + float64(k)*ratio)
	return rate, accepted, speedup
}

// prefillTPS is the compute-bound half: a matrix-matrix product over the whole
// prompt at once, so it scales with FLOPs rather than bandwidth.
func prefillTPS(p Plan, e Engine, gpuShare, cpuShare float64, par Parallelism) float64 {
	m := p.Model
	flopsPerToken := 2 * float64(m.Active())
	// Attention adds a term that grows with how much context is already there.
	flopsPerToken += 4 * float64(m.Layers) * float64(m.Hidden) * float64(p.Ctx) / 2
	tflops := effectiveTFLOPS(p.Devices, gpuShare, cpuShare, par)
	if flopsPerToken <= 0 || tflops <= 0 {
		return 0
	}
	return e.MFU * tflops * 1e12 / flopsPerToken
}

// cpuCachePressure is the share of free RAM the CPU-side weights must hold to
// bandwidths returns the aggregate GPU bandwidth and the host's.
//
// Aggregate, not minimum, because layer-split inference across cards runs each
// layer on the card holding it — the cards work in sequence, each at its own
// bandwidth, so total time is the sum and the effective rate is the harmonic
// combination. For the common case of identical cards this is just one card's
// bandwidth, which is why adding a second 3090 buys capacity, not speed.
func bandwidths(devs []Device, par Parallelism) (gpu, cpu float64) {
	var invSum, sum float64
	var n int
	for _, d := range devs {
		if d.IsCPU {
			cpu = d.BandwidthGBs
			continue
		}
		if d.BandwidthGBs > 0 {
			invSum += 1 / d.BandwidthGBs
			sum += d.BandwidthGBs
			n++
		}
	}
	switch {
	case n == 0:
	case par == TensorParallel:
		// Every card holds a shard of every layer and reads it at the same
		// time, so the bandwidths add. This is the whole reason the two
		// strategies are worth modelling apart: two identical cards are twice
		// the decode rate here and exactly the same rate under a layer split.
		gpu = sum
	default:
		// Harmonic mean: n cards each holding 1/n of the layers take
		// sum(1/bw_i) * bytes/n seconds, so the effective rate is n/sum(1/bw).
		gpu = float64(n) / invSum
	}
	return gpu, cpu
}

// gpuCount is how many non-CPU devices a plan has.
func gpuCount(devs []Device) int {
	n := 0
	for _, d := range devs {
		if !d.IsCPU {
			n++
		}
	}
	return n
}

// allReduceSecondsPerToken is what tensor parallelism pays the interconnect.
//
// A ring all-reduce moves 2*(N-1)/N * S bytes per card, and there is one per
// layer per token, with S the activation vector: hidden * 2 bytes at fp16,
// times the batch. On two desktop cards over PCIe this is microseconds against
// a decode step of tens of milliseconds — which is the honest answer, and the
// reason tensor parallelism is worth it there. It stops being negligible as the
// card count and the batch grow, which is exactly when people notice.
func allReduceSecondsPerToken(m arch.Model, batch, gpus int, linkGBs float64) float64 {
	if gpus < 2 || linkGBs <= 0 {
		return 0
	}
	if batch < 1 {
		batch = 1
	}
	activation := float64(m.Hidden) * 2 * float64(batch)
	n := float64(gpus)
	bytesPerToken := float64(m.Layers) * 2 * (n - 1) / n * activation
	return bytesPerToken / (linkGBs * 1e9)
}

func effectiveTFLOPS(devs []Device, gpuShare, cpuShare float64, par Parallelism) float64 {
	var gpuT float64
	var n int
	for _, d := range devs {
		if !d.IsCPU && d.TFLOPS > 0 {
			gpuT += d.TFLOPS
			n++
		}
	}
	if n > 0 && par != TensorParallel {
		// Layer-split runs one card at a time, so extra cards add capacity
		// rather than prefill throughput. Under tensor parallelism every card
		// works on the same layer at once, so the FLOPs add and gpuT stands.
		gpuT /= float64(n)
	}
	return gpuT*gpuShare + gpuT*cpuShare*cpuPrefillShare
}
