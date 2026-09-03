package fit

// How fast: decode is bandwidth bound, prefill is compute bound, and a
// layer-split plan pays both rates for their share of the layers.

import (
	"math"

	"github.com/fabiocicerchia/llm-fit/internal/arch"
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

	gpuBW, cpuBW := bandwidths(p.Devices)
	var secondsPerToken float64
	if gpuShare > 0 && gpuBW > 0 {
		secondsPerToken += bytesPerToken * gpuShare / (e.MBU * gpuBW * 1e9)
	}
	if cpuShare > 0 && cpuBW > 0 {
		secondsPerToken += bytesPerToken * cpuShare / (cpuEfficiency(m) * cpuBW * 1e9)
	}
	if secondsPerToken <= 0 {
		return 0
	}
	return 1 / secondsPerToken
}

// prefillTPS is the compute-bound half: a matrix-matrix product over the whole
// prompt at once, so it scales with FLOPs rather than bandwidth.
func prefillTPS(p Plan, e Engine, gpuShare, cpuShare float64) float64 {
	m := p.Model
	flopsPerToken := 2 * float64(m.Active())
	// Attention adds a term that grows with how much context is already there.
	flopsPerToken += 4 * float64(m.Layers) * float64(m.Hidden) * float64(p.Ctx) / 2
	tflops := effectiveTFLOPS(p.Devices, gpuShare, cpuShare)
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
func bandwidths(devs []Device) (gpu, cpu float64) {
	var invSum float64
	var n int
	for _, d := range devs {
		if d.IsCPU {
			cpu = d.BandwidthGBs
			continue
		}
		if d.BandwidthGBs > 0 {
			invSum += 1 / d.BandwidthGBs
			n++
		}
	}
	if n > 0 {
		// Harmonic mean: n cards each holding 1/n of the layers take
		// sum(1/bw_i) * bytes/n seconds, so the effective rate is n/sum(1/bw).
		gpu = float64(n) / invSum
	}
	return gpu, cpu
}

func effectiveTFLOPS(devs []Device, gpuShare, cpuShare float64) float64 {
	var gpuT float64
	var n int
	for _, d := range devs {
		if !d.IsCPU && d.TFLOPS > 0 {
			gpuT += d.TFLOPS
			n++
		}
	}
	if n > 0 {
		// Layer-split runs one card at a time, so extra cards add capacity
		// rather than prefill throughput.
		gpuT /= float64(n)
	}
	return gpuT*gpuShare + gpuT*cpuShare*cpuPrefillShare
}
