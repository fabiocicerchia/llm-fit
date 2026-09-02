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
	"math"
	"strconv"

	"github.com/fabiocicerchia/llm-fit/internal/arch"
	"github.com/fabiocicerchia/llm-fit/internal/quant"
)

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

// Device is one place weights can live. A machine is a list of these: usually
// some GPUs and always the host.
type Device struct {
	Name string
	// BytesFree is what is actually available, after the display server and
	// anything else already resident.
	BytesFree int64
	// BandwidthGBs is the number that decides decode speed. For system RAM this
	// is the DDR channel bandwidth, which is 10-40x lower than a GPU's — the
	// entire reason CPU offload is slow.
	BandwidthGBs float64
	// TFLOPS is dense fp16 throughput, no sparsity. Decides prefill.
	TFLOPS float64
	IsCPU  bool
}

// Plan is a proposed way to run one model: which quantization, how much
// context, and how the layers are split across devices.
type Plan struct {
	Model   arch.Model
	Format  quant.Format
	Ctx     int
	KVType  string
	Batch   int
	Engine  string
	Devices []Device
}

// Estimate is what the tool reports for one plan.
type Estimate struct {
	WeightBytes   int64
	KVBytes       int64
	OverheadBytes int64
	TotalBytes    int64

	// GPUBytes and CPUBytes are where that total ends up.
	GPUBytes int64
	CPUBytes int64

	// LayersOnGPU of TotalLayers. Anything short of all of them puts host
	// memory bandwidth in the critical path for every token generated.
	LayersOnGPU int
	TotalLayers int

	Fits        bool
	FullyOnGPU  bool
	DecodeTPS   float64
	PrefillTPS  float64
	MaxCtxAtMem int

	// Concurrency is how many simultaneous requests of this context length the
	// leftover memory can hold. Only meaningful for serving engines.
	Concurrency int

	Verdict Verdict
	Reasons []string
}

type Verdict int

const (
	Unusable Verdict = iota
	Sluggish
	Usable
	Good
	Excellent
)

func (v Verdict) String() string {
	switch v {
	case Excellent:
		return "excellent"
	case Good:
		return "good"
	case Usable:
		return "usable"
	case Sluggish:
		return "sluggish"
	}
	return "unusable"
}

// Thresholds in tokens per second. A person reads prose at roughly 8 tok/s, so
// anything below that feels like waiting rather than watching; 15 is
// comfortable and 30 is faster than you can follow.
func verdictFor(decodeTPS float64) Verdict {
	switch {
	case decodeTPS >= 30:
		return Excellent
	case decodeTPS >= 15:
		return Good
	case decodeTPS >= 7:
		return Usable
	case decodeTPS >= 2.5:
		return Sluggish
	}
	return Unusable
}

// WeightBytes is the size of the model on disk and in memory.
//
// Split three ways because quantizers do: the body at the format's rate, and
// the embedding and output projection at whatever higher precision the format
// keeps them. Collapsing this to one average is accurate for a 70B and wrong by
// several percent for anything small with a large vocabulary.
func WeightBytes(m arch.Model, f quant.Format) int64 {
	embedOne := int64(m.Vocab) * int64(m.Hidden)
	var embed, output float64
	if m.TiedEmbeddings {
		embed = float64(embedOne) * f.EmbedBPW / 8
	} else {
		embed = float64(embedOne) * f.EmbedBPW / 8
		output = float64(embedOne) * f.OutputBPW / 8
	}
	body := float64(m.BodyParams()) * f.BodyBPW / 8
	return int64(body + embed + output)
}

// kvElementBytes is the per-element cost of the named cache precision.
//
// One place, because the fallback used to be written out at each of the three
// call sites: an unknown -kv silently sized every cache as f16, and a plan for
// a quantized cache came back with an f16 context ceiling.
func kvElementBytes(kvType string) float64 {
	if elem, ok := quant.KVCacheBytesPerElement[kvType]; ok {
		return elem
	}
	return 2.0 // f16
}

// KVBytes is the cache for the whole context window at the given precision.
func KVBytes(m arch.Model, ctx, batch int, kvType string) int64 {
	elem := kvElementBytes(kvType)
	if batch < 1 {
		batch = 1
	}
	return int64(m.KVBytesPerToken(elem) * float64(ctx) * float64(batch))
}

// Overhead is everything that is neither weights nor cache: the CUDA context,
// the runtime's own allocations, the activation working set, and the logits.
//
// The logits term is the one that surprises people. A 256k-vocabulary model
// materialises a float per vocabulary entry per token in the batch, so Gemma at
// batch 512 spends half a gigabyte on logits alone.
func Overhead(m arch.Model, e Engine, ctx, batch int, onGPU bool) int64 {
	var total float64

	if onGPU {
		total += e.RuntimeOverheadBytes
	}

	// Activation working set: a handful of hidden-sized tensors per token in
	// flight, times the layers being processed.
	ubatch := batch
	if ubatch < 1 {
		ubatch = 1
	}
	if ubatch > maxUbatch {
		ubatch = maxUbatch
	}
	total += float64(ubatch) * float64(m.Hidden) * 2 * 14

	// Attention scores are quadratic in context when the engine materialises
	// them. Flash attention does not, and every engine here uses it by default,
	// so this is a modest term rather than the dominant one.
	total += float64(ubatch) * float64(ctx) * float64(m.Layers) * 0.5

	// Logits: vocab floats per sampled position.
	total += float64(m.Vocab) * 4 * math.Min(float64(ubatch), 8)

	return int64(total)
}

// Engine is a runtime's constraints and its efficiency, both of which change
// the answer materially. vLLM cannot usefully offload to CPU; llama.cpp can and
// that is its whole point. Efficiency differences of 15% decide ties.
type Engine struct {
	Name string
	// MBU: fraction of theoretical memory bandwidth actually achieved during
	// decode. Nothing reaches 100%; well-tuned CUDA kernels reach 0.8.
	MBU float64
	// MFU: fraction of peak FLOPs achieved during prefill.
	MFU                  float64
	RuntimeOverheadBytes float64
	CanOffloadCPU        bool
	// MemoryFraction is how much of the card the engine will use. vLLM
	// preallocates 90% by default and hands the rest back to nobody.
	MemoryFraction float64
}

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

// EstimatePlan runs the whole calculation for one plan.
func EstimatePlan(p Plan, e Engine) Estimate {
	var est Estimate
	m := p.Model

	est.WeightBytes = WeightBytes(m, p.Format)
	est.KVBytes = KVBytes(m, p.Ctx, p.Batch, p.KVType)

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

	// --- decode: bandwidth bound -------------------------------------------
	//
	// One token reads every active weight once, plus the entire KV cache once.
	// The cache term is not a rounding error: at 128k context an 8B model
	// spends more time reading cache than weights, which is why long-context
	// generation slows down as the conversation grows.
	activeFraction := float64(m.Active()) / math.Max(float64(m.Params), 1)
	weightsRead := float64(est.WeightBytes) * activeFraction
	kvRead := float64(est.KVBytes)

	gpuShare := float64(layersFit) / math.Max(float64(m.Layers), 1)
	cpuShare := 1 - gpuShare

	gpuBW, cpuBW := bandwidths(p.Devices)
	var secondsPerToken float64
	if gpuShare > 0 && gpuBW > 0 {
		secondsPerToken += (weightsRead + kvRead) * gpuShare / (e.MBU * gpuBW * 1e9)
	}
	if cpuShare > 0 && cpuBW > 0 {
		secondsPerToken += (weightsRead + kvRead) * cpuShare / (cpuEfficiency(m) * cpuBW * 1e9)
	}
	if secondsPerToken > 0 {
		est.DecodeTPS = 1 / secondsPerToken
	}

	// --- prefill: compute bound ---------------------------------------------
	flopsPerToken := 2 * float64(m.Active())
	// Attention adds a term that grows with how much context is already there.
	flopsPerToken += 4 * float64(m.Layers) * float64(m.Hidden) * float64(p.Ctx) / 2
	tflops := effectiveTFLOPS(p.Devices, gpuShare, cpuShare)
	if flopsPerToken > 0 && tflops > 0 {
		est.PrefillTPS = e.MFU * tflops * 1e12 / flopsPerToken
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
	return out
}

func layerClause(n int) string {
	if n == 1 {
		return "1 layer is"
	}
	return strconv.Itoa(n) + " layers are"
}

// MaxContext is the largest context the remaining memory can hold once the
// weights are placed. Usually the number people actually want.
func maxContext(m arch.Model, p Plan, e Engine, capacity int64) int {
	perToken := m.KVBytesPerToken(kvElementBytes(p.KVType)) * float64(max(p.Batch, 1))
	if perToken <= 0 {
		return 0
	}
	free := float64(capacity) - float64(WeightBytes(m, p.Format)) - float64(Overhead(m, e, 4096, p.Batch, true))
	if free <= 0 {
		return 0
	}
	return clamp(int(free/perToken), 0, m.MaxCtx)
}

// Concurrency answers the question a serving engine is deployed to answer: how
// many simultaneous requests at this context fit in what is left after the
// weights.
func concurrency(m arch.Model, p Plan, e Engine, gpuBytes, weights, overhead int64) int {
	if !e.CanOffloadCPU && gpuBytes == 0 {
		return 0
	}
	perSeq := m.KVBytesPerToken(kvElementBytes(p.KVType)) * float64(p.Ctx)
	if perSeq <= 0 {
		return 0
	}
	free := float64(gpuBytes) - float64(weights) - float64(overhead)
	if free <= 0 {
		return 0
	}
	return int(free / perSeq)
}

func splitCapacity(devs []Device, fraction float64) (gpu, cpu int64) {
	if fraction <= 0 || fraction > 1 {
		fraction = 1
	}
	for _, d := range devs {
		if d.IsCPU {
			cpu += d.BytesFree
			continue
		}
		gpu += int64(float64(d.BytesFree) * fraction)
	}
	return gpu, cpu
}

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

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
