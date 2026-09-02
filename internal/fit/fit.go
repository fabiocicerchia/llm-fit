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

	"github.com/fabiocicerchia/llm-fit/internal/arch"
	"github.com/fabiocicerchia/llm-fit/internal/quant"
)

const (
	MiB = 1 << 20
	GiB = 1 << 30
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

// Parallelism is how a model is spread across more than one GPU. The two
// strategies have different arithmetic, and modelling only the first made vLLM
// look worse than it is.
type Parallelism int

const (
	// LayerSplit puts whole layers on each card. One card works at a time, so
	// the cards' bandwidths combine harmonically — for identical cards that is
	// one card's bandwidth, which is why a second 3090 buys capacity, not speed.
	// This is what llama.cpp does.
	LayerSplit Parallelism = iota
	// TensorParallel shards every layer's weight matrices across the cards, so
	// all of them read and compute at once and the bandwidths *sum*. The cost is
	// an all-reduce of the activations at each layer, which the interconnect has
	// to carry. This is what vLLM, TensorRT-LLM and ExLlamaV2 do.
	TensorParallel
)

func (p Parallelism) String() string {
	if p == TensorParallel {
		return "tensor-parallel"
	}
	return "layer-split"
}

// DefaultInterconnectGBs is one direction of PCIe 4.0 x16, the interconnect most
// multi-GPU desktops actually have. NVLink is an order of magnitude faster
// (~300 GB/s on a bridged 3090 pair, ~900 on an H100), so a plan that has it
// should say so rather than inherit this floor.
const DefaultInterconnectGBs = 32.0

// Speculative is a draft model paired with the target it speeds up.
//
// The draft proposes Lookahead tokens, the target verifies all of them in one
// forward pass, and every token up to the first rejection is kept. It is a
// throughput trade, not a free one: the draft's weights occupy VRAM the target
// could have used, and a draft that is rarely accepted costs more than it saves.
type Speculative struct {
	Draft  arch.Model
	Format quant.Format
	// Lookahead is how many tokens the draft proposes per verification step.
	Lookahead int
	// AcceptanceRate is the probability the target accepts a proposed token.
	// Zero means DefaultAcceptanceRate.
	AcceptanceRate float64
}

// DefaultAcceptanceRate is the middle of the range Leviathan et al. (2023),
// "Fast Inference from Transformers via Speculative Decoding", measure for a
// draft drawn from the same family as its target — roughly 0.6-0.8 depending on
// the pair and the sampling temperature. It is reported in the output rather
// than buried, because the speedup is far more sensitive to it than to anything
// else here.
const DefaultAcceptanceRate = 0.7

// DefaultLookahead is the draft depth vLLM and llama.cpp both default to.
const DefaultLookahead = 4

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

	// Parallelism across the GPUs. The zero value is LayerSplit; EstimatePlan
	// substitutes the engine's own default when ParallelismSet is false, so a
	// caller that says nothing gets the strategy the runtime would actually use.
	Parallelism    Parallelism
	ParallelismSet bool
	// InterconnectGBs is the per-GPU link bandwidth tensor parallelism has to
	// push activations over. Zero means DefaultInterconnectGBs.
	InterconnectGBs float64

	// Speculative, when set, pairs a draft model with this target.
	Speculative *Speculative
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

	// Parallelism actually used, and what the other strategy would have given.
	// The delta is the point: it is the difference between "a second card buys
	// capacity" and "a second card buys speed", and which one is true depends
	// entirely on the runtime.
	Parallelism     Parallelism
	AltDecodeTPS    float64
	InterconnectTPS float64 // decode ceiling the interconnect alone imposes
	// DraftBytes is the extra VRAM a speculative pair costs, and Speedup /
	// AcceptanceRate are what it buys. Zero when no draft is configured.
	DraftBytes      int64
	Speedup         float64
	AcceptanceRate  float64
	AcceptedPerStep float64

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

// KVBytes is the cache for the whole context window at the given precision.
func KVBytes(m arch.Model, ctx, batch int, kvType string) int64 {
	elem, ok := quant.KVCacheBytesPerElement[kvType]
	if !ok {
		elem = 2.0
	}
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
	if ubatch > 512 {
		ubatch = 512
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
	// DefaultParallelism is the strategy this runtime uses across several GPUs
	// when the caller does not say. llama.cpp splits layers; vLLM and
	// ExLlamaV2 shard tensors, and modelling them as layer-split made them look
	// slower than they are.
	DefaultParallelism Parallelism
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
// altDecodeTPS is the decode rate the *other* parallelism strategy would give,
// with everything else held equal. Reported so the choice is visible: under a
// layer split a second identical card changes nothing about speed, and under
// tensor parallelism it roughly doubles it, and no single number says that.
func altDecodeTPS(p Plan, e Engine, m arch.Model, used Parallelism, link, bytesPerToken, gpuShare, cpuShare float64) float64 {
	alt := TensorParallel
	if used == TensorParallel {
		alt = LayerSplit
	}
	gpuBW, cpuBW := bandwidths(p.Devices, alt)
	var spt float64
	if gpuShare > 0 && gpuBW > 0 {
		spt += bytesPerToken * gpuShare / (e.MBU * gpuBW * 1e9)
	}
	if cpuShare > 0 && cpuBW > 0 {
		spt += bytesPerToken * cpuShare / (cpuEfficiency(m) * cpuBW * 1e9)
	}
	if alt == TensorParallel {
		spt += allReduceSecondsPerToken(m, p.Batch, gpuCount(p.Devices), link)
	}
	if spt <= 0 {
		return 0
	}
	return 1 / spt
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
func speculativeSpeedup(target arch.Model, s Speculative, f quant.Format, baseTPS float64) (rate, accepted, speedup float64) {
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

	est.OverheadBytes = Overhead(m, e, p.Ctx, p.Batch, gpuTotal > 0)
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
	est.FullyOnGPU = cpuLayers == 0 && gpuTotal > 0

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

	gpuBW, cpuBW := bandwidths(p.Devices, par)
	var secondsPerToken float64
	if gpuShare > 0 && gpuBW > 0 {
		secondsPerToken += (weightsRead + kvRead) * gpuShare / (e.MBU * gpuBW * 1e9)
	}
	if cpuShare > 0 && cpuBW > 0 {
		secondsPerToken += (weightsRead + kvRead) * cpuShare / (cpuEfficiency(m) * cpuBW * 1e9)
	}
	// Tensor parallelism pays an all-reduce per layer per token. Added to the
	// bandwidth term rather than modelled as a separate ceiling, because the
	// two are serial: the card cannot start the next layer until the reduce for
	// this one has landed.
	link := p.InterconnectGBs
	if link <= 0 {
		link = DefaultInterconnectGBs
	}
	if par == TensorParallel {
		reduce := allReduceSecondsPerToken(m, p.Batch, gpuCount(p.Devices), link)
		secondsPerToken += reduce
		if reduce > 0 {
			est.InterconnectTPS = 1 / reduce
		}
	}
	if secondsPerToken > 0 {
		est.DecodeTPS = 1 / secondsPerToken
	}

	// What the other strategy would have given, so the delta is visible rather
	// than implied. This is the number that says whether a second card is worth
	// buying for *this* runtime.
	if gpuCount(p.Devices) > 1 {
		est.AltDecodeTPS = altDecodeTPS(p, e, m, par, link, weightsRead+kvRead, gpuShare, cpuShare)
	}

	// Speculative decoding multiplies the decode rate, so it is applied to the
	// bandwidth-bound figure above rather than replacing it.
	if p.Speculative != nil {
		est.AcceptanceRate, est.AcceptedPerStep, est.Speedup = speculativeSpeedup(
			m, *p.Speculative, p.Format, est.DecodeTPS)
		est.DecodeTPS *= est.Speedup
		est.AltDecodeTPS *= est.Speedup
	}

	// --- prefill: compute bound ---------------------------------------------
	flopsPerToken := 2 * float64(m.Active())
	// Attention adds a term that grows with how much context is already there.
	flopsPerToken += 4 * float64(m.Layers) * float64(m.Hidden) * float64(p.Ctx) / 2
	tflops := effectiveTFLOPS(p.Devices, gpuShare, cpuShare, par)
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
		out = append(out, plural(cpuLayers)+" running on the CPU: every token waits on system RAM, which is why this is slow")
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

func interconnect(p Plan) float64 {
	if p.InterconnectGBs > 0 {
		return p.InterconnectGBs
	}
	return DefaultInterconnectGBs
}

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

func plural(n int) string {
	if n == 1 {
		return "1 layer is"
	}
	return strconv.Itoa(n) + " layers are"
}

// MaxContext is the largest context the remaining memory can hold once the
// weights are placed. Usually the number people actually want.
func maxContext(m arch.Model, p Plan, e Engine, capacity int64) int {
	elem, ok := quant.KVCacheBytesPerElement[p.KVType]
	if !ok {
		elem = 2.0
	}
	perToken := m.KVBytesPerToken(elem) * float64(max(p.Batch, 1))
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
	elem, ok := quant.KVCacheBytesPerElement[p.KVType]
	if !ok {
		elem = 2.0
	}
	perSeq := m.KVBytesPerToken(elem) * float64(p.Ctx)
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
	// CPU prefill is roughly two orders of magnitude slower; treat the CPU
	// share as contributing almost nothing rather than pretending it helps.
	return gpuT*gpuShare + gpuT*cpuShare*0.02
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
