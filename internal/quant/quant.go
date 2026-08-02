// Package quant holds the bits-per-weight of every quantization format worth
// running, and which runtimes can load it.
//
// The naive assumption is that "4-bit" means four bits. None of them do. A
// quantized block stores its own scale, and often a minimum, alongside the
// weights: Q4_0 packs 32 weights into 18 bytes, which is 4.5 bits per weight,
// and the K-quants mix precisions across tensors so that Q4_K_M lands near
// 4.83. Estimating a 70B at a literal 4 bits predicts 35GB for a file that is
// really 42.5GB — the difference between "fits on two 24GB cards" and not.
//
// The numbers below are calibrated so that predicted file sizes land within a
// few percent of published ones; quant_test.go checks that against real
// releases rather than trusting the table.
package quant

import "sort"

type Family string

const (
	GGUF   Family = "gguf"   // llama.cpp, Ollama, llamafile
	AWQ    Family = "awq"    // vLLM, SGLang, TGI
	GPTQ   Family = "gptq"   // vLLM, SGLang, TGI, ExLlama
	EXL2   Family = "exl2"   // ExLlamaV2/V3 — NVIDIA only
	FP8    Family = "fp8"    // vLLM on Hopper/Ada; needs compute capability 8.9+
	BnB    Family = "bnb"    // bitsandbytes — transformers, vLLM (slow)
	MLX    Family = "mlx"    // Apple silicon only
	Native Family = "native" // unquantized fp16/bf16
)

type Format struct {
	Name   string
	Family Family

	// BodyBPW is bits per weight for the transformer layers.
	BodyBPW float64
	// EmbedBPW and OutputBPW cover the token embedding and the output
	// projection, which quantizers deliberately keep at higher precision than
	// the body. llama.cpp leaves output.weight at Q6_K inside a Q4_K_M model.
	EmbedBPW  float64
	OutputBPW float64

	// Quality is a coarse ranking used to break ties when several formats fit:
	// 100 is lossless, and anything under about 60 is visibly degraded.
	Quality int

	Notes string
}

// bpw returns a format where all three tensor groups share one precision.
func uniform(name string, family Family, bpw float64, quality int, notes string) Format {
	return Format{Name: name, Family: family, BodyBPW: bpw, EmbedBPW: bpw, OutputBPW: bpw, Quality: quality, Notes: notes}
}

// Formats is ordered roughly smallest to largest.
//
// The K-quant body figures are effective rates over a whole model, not the
// block arithmetic of a single tensor type: Q4_K_M applies Q6_K to attention
// value and feed-forward down projections, so its average sits above the 4.5
// that Q4_K alone would give.
var Formats = []Format{
	uniform("IQ1_S", GGUF, 1.56, 15, "barely coherent; only worth it to fit a very large model at all"),
	uniform("IQ1_M", GGUF, 1.75, 20, "as above, marginally better"),
	uniform("IQ2_XXS", GGUF, 2.06, 30, "heavy loss; usable only on 70B+ where there is redundancy to spare"),
	uniform("IQ2_M", GGUF, 2.70, 40, "the smallest quant most people find tolerable, and only on large models"),
	{Name: "Q2_K", Family: GGUF, BodyBPW: 3.35, EmbedBPW: 2.63, OutputBPW: 6.56, Quality: 42, Notes: "noticeable degradation"},
	uniform("IQ3_XXS", GGUF, 3.06, 48, "better than Q3_K at the same size, slower on CPU"),
	{Name: "Q3_K_S", Family: GGUF, BodyBPW: 3.50, EmbedBPW: 3.44, OutputBPW: 6.56, Quality: 52},
	{Name: "Q3_K_M", Family: GGUF, BodyBPW: 3.91, EmbedBPW: 3.44, OutputBPW: 6.56, Quality: 58},
	uniform("IQ3_M", GGUF, 3.66, 60, "i-quant; matches Q3_K_L quality at Q3_K_M size"),
	{Name: "Q3_K_L", Family: GGUF, BodyBPW: 4.27, EmbedBPW: 3.44, OutputBPW: 6.56, Quality: 62},
	uniform("IQ4_XS", GGUF, 4.25, 70, "smaller than Q4_K_S at similar quality; the best sub-4.5 option"),
	{Name: "Q4_0", Family: GGUF, BodyBPW: 4.55, EmbedBPW: 4.50, OutputBPW: 6.56, Quality: 63, Notes: "legacy; Q4_K_M is better at the same size"},
	{Name: "Q4_K_S", Family: GGUF, BodyBPW: 4.58, EmbedBPW: 4.50, OutputBPW: 6.56, Quality: 72},
	{Name: "Q4_K_M", Family: GGUF, BodyBPW: 4.83, EmbedBPW: 4.50, OutputBPW: 6.56, Quality: 78, Notes: "the default recommendation: the knee of the size/quality curve"},
	{Name: "Q5_K_S", Family: GGUF, BodyBPW: 5.52, EmbedBPW: 5.50, OutputBPW: 6.56, Quality: 84},
	{Name: "Q5_K_M", Family: GGUF, BodyBPW: 5.67, EmbedBPW: 5.50, OutputBPW: 6.56, Quality: 87},
	uniform("Q6_K", GGUF, 6.56, 94, "effectively indistinguishable from fp16 for most uses"),
	uniform("Q8_0", GGUF, 8.50, 98, "lossless in practice; twice the memory of Q4_K_M for no visible gain"),

	uniform("AWQ-4bit", AWQ, 4.25, 74, "activation-aware; the usual vLLM 4-bit choice"),
	uniform("GPTQ-4bit", GPTQ, 4.25, 72, "group size 128"),
	uniform("GPTQ-8bit", GPTQ, 8.25, 96, ""),
	uniform("EXL2-4.0", EXL2, 4.15, 73, "ExLlamaV2; fastest single-user option on NVIDIA"),
	uniform("EXL2-6.0", EXL2, 6.15, 92, ""),
	uniform("FP8", FP8, 8.00, 96, "near-lossless and hardware-accelerated, but needs Ada/Hopper or newer"),
	uniform("NF4", BnB, 4.50, 68, "bitsandbytes; convenient but slower than AWQ/GPTQ at the same size"),
	uniform("INT8", BnB, 8.50, 94, "bitsandbytes; notably slow"),
	uniform("MLX-4bit", MLX, 4.50, 74, "Apple silicon"),
	uniform("MLX-8bit", MLX, 8.50, 96, "Apple silicon"),
	uniform("FP16", Native, 16.0, 100, "unquantized"),
	uniform("BF16", Native, 16.0, 100, "unquantized"),
}

func ByName(name string) (Format, bool) {
	for _, f := range Formats {
		if f.Name == name {
			return f, true
		}
	}
	return Format{}, false
}

func ByFamily(fam Family) []Format {
	var out []Format
	for _, f := range Formats {
		if f.Family == fam {
			out = append(out, f)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].BodyBPW < out[j].BodyBPW })
	return out
}

// KVCacheBytesPerElement is the per-element cost of the KV cache at a given
// precision. Quantizing the cache is the cheapest way to buy context: at Q8_0
// it halves with no measurable quality cost, which on a long-context model
// saves more memory than dropping the weights a whole quantization level.
var KVCacheBytesPerElement = map[string]float64{
	"f32":  4.0,
	"f16":  2.0,
	"bf16": 2.0,
	"q8_0": 34.0 / 32.0, // 32 int8 values plus one fp16 scale
	"q5_1": 24.0 / 32.0,
	"q4_0": 18.0 / 32.0, // 16 packed nibbles plus one fp16 scale
}
