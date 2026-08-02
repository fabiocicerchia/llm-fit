package fit

import (
	"math"
	"testing"

	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/arch"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/quant"
)

// The whole tool rests on the weight calculation, so it is checked against file
// sizes that actually exist on Hugging Face rather than against itself.
//
// If these drift, every recommendation drifts with them: predicting 35GB for a
// 70B that is really 42.5GB is the difference between "fits on two 24GB cards"
// and a failed load.
func TestWeightBytesMatchesPublishedGGUFSizes(t *testing.T) {
	llama3_8b := arch.Model{Params: 8030261248, Vocab: 128256, Hidden: 4096, Layers: 32}
	llama3_70b := arch.Model{Params: 70553706496, Vocab: 128256, Hidden: 8192, Layers: 80}
	llama2_7b := arch.Model{Params: 6738415616, Vocab: 32000, Hidden: 4096, Layers: 32}
	mistral7b := arch.Model{Params: 7241732096, Vocab: 32000, Hidden: 4096, Layers: 32}

	cases := []struct {
		name     string
		model    arch.Model
		format   string
		wantGB   float64 // published file size, GB as 1e9 bytes
		tolerate float64 // fractional
	}{
		{"Llama-3-8B Q4_K_M", llama3_8b, "Q4_K_M", 4.92, 0.03},
		{"Llama-3-8B Q5_K_M", llama3_8b, "Q5_K_M", 5.73, 0.03},
		{"Llama-3-8B Q6_K", llama3_8b, "Q6_K", 6.60, 0.03},
		{"Llama-3-8B Q8_0", llama3_8b, "Q8_0", 8.54, 0.03},
		{"Llama-3-70B Q4_K_M", llama3_70b, "Q4_K_M", 42.5, 0.03},
		{"Llama-2-7B Q4_K_M", llama2_7b, "Q4_K_M", 4.08, 0.03},
		{"Llama-2-7B Q2_K", llama2_7b, "Q2_K", 2.83, 0.05},
		{"Llama-2-7B Q5_K_M", llama2_7b, "Q5_K_M", 4.78, 0.03},
		{"Llama-2-7B Q6_K", llama2_7b, "Q6_K", 5.53, 0.03},
		{"Llama-2-7B Q8_0", llama2_7b, "Q8_0", 7.16, 0.03},
		{"Mistral-7B Q4_K_M", mistral7b, "Q4_K_M", 4.37, 0.03},
	}

	for _, c := range cases {
		f, ok := quant.ByName(c.format)
		if !ok {
			t.Fatalf("%s: unknown format", c.name)
		}
		got := float64(WeightBytes(c.model, f)) / 1e9
		if rel := math.Abs(got-c.wantGB) / c.wantGB; rel > c.tolerate {
			t.Errorf("%s: got %.2f GB, published %.2f GB (%.1f%% off, tolerance %.0f%%)",
				c.name, got, c.wantGB, rel*100, c.tolerate*100)
		}
	}
}

// A quantized model must be smaller than the same model unquantized, and the
// ordering must follow bits-per-weight. Trivially true if the table is sane,
// and it catches a mistyped constant.
func TestWeightOrderingFollowsBitsPerWeight(t *testing.T) {
	m := arch.Model{Params: 8030261248, Vocab: 128256, Hidden: 4096, Layers: 32}
	order := []string{"IQ2_M", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"}
	var prev int64
	for _, name := range order {
		f, _ := quant.ByName(name)
		got := WeightBytes(m, f)
		if got <= prev {
			t.Errorf("%s (%d) is not larger than the previous format (%d)", name, got, prev)
		}
		prev = got
	}
}

// Grouped-query attention is the single biggest correction in the KV maths. A
// tool that ignores it overestimates a 70B's cache eightfold and tells you a
// context fits when it does not.
func TestKVCacheAccountsForGroupedQueryAttention(t *testing.T) {
	gqa := arch.Model{Layers: 80, Heads: 64, KVHeads: 8, HeadDim: 128}
	mha := arch.Model{Layers: 80, Heads: 64, KVHeads: 64, HeadDim: 128}

	gqaBytes := KVBytes(gqa, 8192, 1, "f16")
	mhaBytes := KVBytes(mha, 8192, 1, "f16")

	if ratio := float64(mhaBytes) / float64(gqaBytes); math.Abs(ratio-8) > 0.01 {
		t.Errorf("MHA should cost 8x the GQA cache here, got %.2fx", ratio)
	}
	// 2 * 80 layers * 8 heads * 128 dim * 2 bytes * 8192 tokens = 2.68 GB
	if want := int64(2 * 80 * 8 * 128 * 2 * 8192); gqaBytes != want {
		t.Errorf("got %d bytes, want %d", gqaBytes, want)
	}
}

// MLA is not a scaling factor on the GQA formula, it is a different cache. It
// is why DeepSeek-V3 serves 160k context on hardware that could not hold a
// GQA model's cache at that length.
func TestMLACacheIsDramaticallySmaller(t *testing.T) {
	mla := arch.Model{Layers: 61, Heads: 128, KVHeads: 128, HeadDim: 128,
		Attention: arch.MLA, KVLoraRank: 512, QKRopeHeadDim: 64}
	asGQA := arch.Model{Layers: 61, Heads: 128, KVHeads: 128, HeadDim: 128}

	small := KVBytes(mla, 8192, 1, "f16")
	big := KVBytes(asGQA, 8192, 1, "f16")
	if ratio := float64(big) / float64(small); ratio < 10 {
		t.Errorf("MLA should be an order of magnitude smaller, got %.1fx", ratio)
	}
}

// Quantizing the cache is often a better trade than quantizing the weights, and
// the tool can only say so if the ratios are right.
func TestKVQuantizationRatios(t *testing.T) {
	m := arch.Model{Layers: 32, Heads: 32, KVHeads: 8, HeadDim: 128}
	f16 := KVBytes(m, 32768, 1, "f16")
	q8 := KVBytes(m, 32768, 1, "q8_0")
	q4 := KVBytes(m, 32768, 1, "q4_0")

	if r := float64(f16) / float64(q8); r < 1.8 || r > 1.95 {
		t.Errorf("q8_0 cache should be a shade under half of f16, got %.2fx", r)
	}
	if r := float64(f16) / float64(q4); r < 3.4 || r > 3.6 {
		t.Errorf("q4_0 cache should be about 3.5x smaller than f16, got %.2fx", r)
	}
}

// An unknown KV type must not silently produce a tiny cache — falling back to
// f16 overestimates memory, which is the safe direction.
func TestUnknownKVTypeFallsBackToF16(t *testing.T) {
	m := arch.Model{Layers: 32, Heads: 32, KVHeads: 8, HeadDim: 128}
	if KVBytes(m, 4096, 1, "nonsense") != KVBytes(m, 4096, 1, "f16") {
		t.Error("unknown KV type should fall back to f16")
	}
}

func llamaCpp() Engine {
	return Engine{Name: "llama.cpp", MBU: 0.80, MFU: 0.32, RuntimeOverheadBytes: 250 * MiB, CanOffloadCPU: true, MemoryFraction: 0.95}
}

func vllm() Engine {
	return Engine{Name: "vLLM", MBU: 0.72, MFU: 0.55, RuntimeOverheadBytes: 2200 * MiB, CanOffloadCPU: false, MemoryFraction: 0.90}
}

func rtx3090() Device {
	return Device{Name: "RTX 3090", BytesFree: 24 * GiB, BandwidthGBs: 936, TFLOPS: 71}
}

func hostRAM(gib int64) Device {
	return Device{Name: "system RAM", BytesFree: gib * GiB, BandwidthGBs: 80, IsCPU: true}
}

// The headline claim of the whole tool: decode speed is bandwidth divided by
// bytes read per token. A 3090 running an 8B at Q4_K_M reads about 4.9GB per
// token against 936 GB/s, so it lands near 150 tok/s — which is what people
// actually measure.
func TestDecodeSpeedIsBandwidthBound(t *testing.T) {
	m := arch.Model{Params: 8030261248, Vocab: 128256, Hidden: 4096, Layers: 32,
		Heads: 32, KVHeads: 8, HeadDim: 128, MaxCtx: 131072}
	f, _ := quant.ByName("Q4_K_M")

	est := EstimatePlan(Plan{
		Model: m, Format: f, Ctx: 4096, KVType: "f16", Batch: 1,
		Devices: []Device{rtx3090(), hostRAM(64)},
	}, llamaCpp())

	if !est.Fits || !est.FullyOnGPU {
		t.Fatalf("an 8B at Q4_K_M must fit entirely on a 24GB card: %+v", est.Reasons)
	}
	if est.DecodeTPS < 100 || est.DecodeTPS > 200 {
		t.Errorf("expected roughly 100-200 tok/s, got %.0f", est.DecodeTPS)
	}
	if est.Verdict != Excellent {
		t.Errorf("verdict = %v, want excellent", est.Verdict)
	}
}

// The case the tool exists to warn about. A 70B at Q4_K_M is 42GB and a 24GB
// card holds about half of it; llama.cpp will run it, and it will be unusable.
// "Fits" must not be reported as "works".
func TestOffloadingToCPUIsReportedAsSlow(t *testing.T) {
	m := arch.Model{Params: 70553706496, Vocab: 128256, Hidden: 8192, Layers: 80,
		Heads: 64, KVHeads: 8, HeadDim: 128, MaxCtx: 131072}
	f, _ := quant.ByName("Q4_K_M")

	est := EstimatePlan(Plan{
		Model: m, Format: f, Ctx: 4096, KVType: "f16", Batch: 1,
		Devices: []Device{rtx3090(), hostRAM(64)},
	}, llamaCpp())

	if !est.Fits {
		t.Fatal("with 24GB VRAM and 64GB RAM this does fit, just slowly")
	}
	if est.FullyOnGPU {
		t.Fatal("a 42GB model cannot be fully on a 24GB card")
	}
	if est.DecodeTPS > 8 {
		t.Errorf("partial offload should be slow, got %.1f tok/s", est.DecodeTPS)
	}
	if est.Verdict > Usable {
		t.Errorf("verdict = %v, should not be recommended", est.Verdict)
	}
	if !hasReasonContaining(est.Reasons, "CPU") {
		t.Errorf("the reason must name CPU offload, got %v", est.Reasons)
	}
}

// vLLM does not offload. The same plan that llama.cpp runs slowly, vLLM cannot
// run at all — and saying so is more useful than a speed estimate.
func TestVLLMRefusesWhatDoesNotFitInVRAM(t *testing.T) {
	m := arch.Model{Params: 70553706496, Vocab: 128256, Hidden: 8192, Layers: 80,
		Heads: 64, KVHeads: 8, HeadDim: 128, MaxCtx: 131072}
	f, _ := quant.ByName("AWQ-4bit")

	est := EstimatePlan(Plan{
		Model: m, Format: f, Ctx: 4096, KVType: "f16", Batch: 1,
		Devices: []Device{rtx3090(), hostRAM(256)},
	}, vllm())

	if est.Fits {
		t.Error("a 40GB model must not be reported as fitting on a 24GB card under vLLM")
	}
	if !hasReasonContaining(est.Reasons, "VRAM") {
		t.Errorf("the reason should explain the VRAM requirement, got %v", est.Reasons)
	}
}

// An MoE reads only its active experts, so it decodes like a small model while
// occupying memory like a large one. Both halves must show up.
func TestMoEDecodesFasterThanItsSize(t *testing.T) {
	moe := arch.Model{Params: 46702792704, ActiveParams: 12879069184, Vocab: 32000, Hidden: 4096,
		Layers: 32, Heads: 32, KVHeads: 8, HeadDim: 128, MaxCtx: 32768, Experts: 8, ExpertsActive: 2}
	dense := arch.Model{Params: 46702792704, Vocab: 32000, Hidden: 4096,
		Layers: 32, Heads: 32, KVHeads: 8, HeadDim: 128, MaxCtx: 32768}
	f, _ := quant.ByName("Q4_K_M")

	devs := []Device{{Name: "A100", BytesFree: 80 * GiB, BandwidthGBs: 2039, TFLOPS: 312}, hostRAM(256)}
	plan := func(m arch.Model) Plan {
		return Plan{Model: m, Format: f, Ctx: 4096, KVType: "f16", Batch: 1, Devices: devs}
	}

	moeEst := EstimatePlan(plan(moe), llamaCpp())
	denseEst := EstimatePlan(plan(dense), llamaCpp())

	if moeEst.WeightBytes != denseEst.WeightBytes {
		t.Error("the MoE occupies the same memory as the dense model of equal parameter count")
	}
	if ratio := moeEst.DecodeTPS / denseEst.DecodeTPS; ratio < 3 {
		t.Errorf("the MoE should decode at least 3x faster (12.9B of 46.7B active), got %.1fx", ratio)
	}
}

// Long context is where the KV cache stops being a rounding error. At 128k an
// 8B model's cache exceeds its weights, and generation slows accordingly.
func TestLongContextSlowsGenerationViaKVReads(t *testing.T) {
	m := arch.Model{Params: 8030261248, Vocab: 128256, Hidden: 4096, Layers: 32,
		Heads: 32, KVHeads: 8, HeadDim: 128, MaxCtx: 131072}
	f, _ := quant.ByName("Q4_K_M")
	devs := []Device{{Name: "A100-80", BytesFree: 80 * GiB, BandwidthGBs: 2039, TFLOPS: 312}, hostRAM(128)}

	short := EstimatePlan(Plan{Model: m, Format: f, Ctx: 2048, KVType: "f16", Batch: 1, Devices: devs}, llamaCpp())
	long := EstimatePlan(Plan{Model: m, Format: f, Ctx: 131072, KVType: "f16", Batch: 1, Devices: devs}, llamaCpp())

	if long.DecodeTPS >= short.DecodeTPS {
		t.Errorf("128k context must decode slower than 2k: %.0f vs %.0f", long.DecodeTPS, short.DecodeTPS)
	}
	if long.KVBytes <= long.WeightBytes {
		t.Error("at 128k the cache should exceed the weights for this model")
	}
	if !hasReasonContaining(long.Reasons, "KV cache is larger") {
		t.Errorf("the tool should point at cache quantization here, got %v", long.Reasons)
	}
}

// MaxCtxAtMem is the number people are usually really asking for.
func TestMaxContextShrinksAsQuantizationGrows(t *testing.T) {
	m := arch.Model{Params: 8030261248, Vocab: 128256, Hidden: 4096, Layers: 32,
		Heads: 32, KVHeads: 8, HeadDim: 128, MaxCtx: 131072}
	devs := []Device{rtx3090(), hostRAM(0)}

	small, _ := quant.ByName("Q4_K_M")
	big, _ := quant.ByName("Q8_0")
	lo := EstimatePlan(Plan{Model: m, Format: small, Ctx: 4096, KVType: "f16", Batch: 1, Devices: devs}, llamaCpp())
	hi := EstimatePlan(Plan{Model: m, Format: big, Ctx: 4096, KVType: "f16", Batch: 1, Devices: devs}, llamaCpp())

	if lo.MaxCtxAtMem <= hi.MaxCtxAtMem {
		t.Errorf("a smaller quant must leave room for more context: Q4_K_M %d vs Q8_0 %d",
			lo.MaxCtxAtMem, hi.MaxCtxAtMem)
	}
	if lo.MaxCtxAtMem > m.MaxCtx {
		t.Error("max context must never exceed what the model was trained for")
	}
}

// Two cards hold twice the model but each layer still runs on one card, so
// capacity doubles and speed does not. Claiming otherwise would recommend a
// second GPU for the wrong reason.
func TestSecondGPUAddsCapacityNotSpeed(t *testing.T) {
	m := arch.Model{Params: 32763876352, Vocab: 152064, Hidden: 5120, Layers: 64,
		Heads: 40, KVHeads: 8, HeadDim: 128, MaxCtx: 32768}
	f, _ := quant.ByName("Q4_K_M")

	// 18.6 GiB of weights fits one card on its own; it is the 8 GiB of cache at
	// 32k that pushes it over, which is the realistic version of this problem.
	one := EstimatePlan(Plan{Model: m, Format: f, Ctx: 32768, KVType: "f16", Batch: 1,
		Devices: []Device{rtx3090(), hostRAM(64)}}, llamaCpp())
	two := EstimatePlan(Plan{Model: m, Format: f, Ctx: 32768, KVType: "f16", Batch: 1,
		Devices: []Device{rtx3090(), rtx3090(), hostRAM(64)}}, llamaCpp())

	if one.FullyOnGPU {
		t.Fatal("weights plus a 32k cache should not fit one 24GB card")
	}
	if !two.FullyOnGPU {
		t.Fatal("two cards should hold it")
	}
	// Speed improves because the CPU leaves the critical path, not because the
	// cards are summed — the fully-resident figure must still be one card's.
	solo := EstimatePlan(Plan{Model: m, Format: f, Ctx: 32768, KVType: "f16", Batch: 1,
		Devices: []Device{{Name: "big", BytesFree: 48 * GiB, BandwidthGBs: 936, TFLOPS: 71}, hostRAM(64)}}, llamaCpp())
	if rel := math.Abs(two.DecodeTPS-solo.DecodeTPS) / solo.DecodeTPS; rel > 0.02 {
		t.Errorf("two 936 GB/s cards should decode like one, got %.0f vs %.0f", two.DecodeTPS, solo.DecodeTPS)
	}
}

// Concurrency is the question a serving deployment is actually asking.
func TestConcurrencyFallsAsContextGrows(t *testing.T) {
	m := arch.Model{Params: 8030261248, Vocab: 128256, Hidden: 4096, Layers: 32,
		Heads: 32, KVHeads: 8, HeadDim: 128, MaxCtx: 131072}
	f, _ := quant.ByName("AWQ-4bit")
	devs := []Device{{Name: "A100-80", BytesFree: 80 * GiB, BandwidthGBs: 2039, TFLOPS: 312}}

	short := EstimatePlan(Plan{Model: m, Format: f, Ctx: 4096, KVType: "f16", Batch: 1, Devices: devs}, vllm())
	long := EstimatePlan(Plan{Model: m, Format: f, Ctx: 32768, KVType: "f16", Batch: 1, Devices: devs}, vllm())

	if short.Concurrency <= long.Concurrency {
		t.Errorf("shorter contexts must allow more concurrent sequences: %d vs %d",
			short.Concurrency, long.Concurrency)
	}
	if long.Concurrency < 1 {
		t.Error("an 8B at 4-bit on an 80GB card should serve many 32k sequences")
	}
}

// A model that cannot fit anywhere must say so rather than return a fast
// estimate for a plan that will never load.
func TestModelTooLargeForAnyMemoryDoesNotFit(t *testing.T) {
	m := arch.Model{Params: 671000000000, ActiveParams: 37000000000, Vocab: 129280, Hidden: 7168,
		Layers: 61, Heads: 128, KVHeads: 128, HeadDim: 128, MaxCtx: 163840,
		Attention: arch.MLA, KVLoraRank: 512, QKRopeHeadDim: 64}
	f, _ := quant.ByName("Q4_K_M")

	est := EstimatePlan(Plan{Model: m, Format: f, Ctx: 4096, KVType: "f16", Batch: 1,
		Devices: []Device{rtx3090(), hostRAM(64)}}, llamaCpp())

	if est.Fits {
		t.Error("a 400GB model must not fit in 24GB of VRAM and 64GB of RAM")
	}
	if est.Verdict != Unusable {
		t.Errorf("verdict = %v, want unusable", est.Verdict)
	}
}

// Verdict boundaries are the tool's definition of "smoothly", so they are
// pinned rather than left to drift.
func TestVerdictThresholds(t *testing.T) {
	cases := []struct {
		tps  float64
		want Verdict
	}{
		{45, Excellent}, {30, Excellent}, {29.9, Good}, {15, Good},
		{14.9, Usable}, {7, Usable}, {6.9, Sluggish}, {2.5, Sluggish}, {2.4, Unusable}, {0, Unusable},
	}
	for _, c := range cases {
		if got := verdictFor(c.tps); got != c.want {
			t.Errorf("%.1f tok/s: got %v, want %v", c.tps, got, c.want)
		}
	}
}

func hasReasonContaining(reasons []string, sub string) bool {
	for _, r := range reasons {
		if len(sub) == 0 || containsFold(r, sub) {
			return true
		}
	}
	return false
}

func containsFold(s, sub string) bool {
	if len(sub) > len(s) {
		return false
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if equalFoldASCII(s[i:i+len(sub)], sub) {
			return true
		}
	}
	return false
}

func equalFoldASCII(a, b string) bool {
	for i := range a {
		x, y := a[i], b[i]
		if 'A' <= x && x <= 'Z' {
			x += 32
		}
		if 'A' <= y && y <= 'Z' {
			y += 32
		}
		if x != y {
			return false
		}
	}
	return true
}
