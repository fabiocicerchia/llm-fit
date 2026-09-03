package fit

// The shapes the arithmetic is expressed in: what a machine offers, what a plan
// asks for, and what comes back. No calculation lives here.

import (
	"github.com/fabiocicerchia/llm-fit/internal/arch"
	"github.com/fabiocicerchia/llm-fit/internal/quant"
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
