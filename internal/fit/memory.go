package fit

// How many bytes: weights, KV cache, runtime overhead, and the two questions
// memory alone answers — how much context fits, and how many requests.

import (
	"math"

	"github.com/fabiocicerchia/llm-fit/internal/arch"
	"github.com/fabiocicerchia/llm-fit/internal/quant"
)

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
