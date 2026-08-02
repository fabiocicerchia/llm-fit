package advisor

import (
	"testing"

	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/arch"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/catalog"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/engine"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/fit"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/hw"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/quant"
)

func machine(vramGiB, ramGiB float64, name string, bw, tflops, cc float64) hw.Machine {
	m := hw.Machine{
		OS: "linux", Arch: "amd64", CPUCores: 16,
		RAMTotal: int64(ramGiB * (1 << 30)), RAMFree: int64(ramGiB * (1 << 30)),
		RAMBandwidth: 45,
	}
	if vramGiB > 0 {
		b := int64(vramGiB * (1 << 30))
		m.GPUs = []hw.GPU{{
			Name: name, Vendor: hw.NVIDIA, TotalBytes: b, FreeBytes: b,
			BandwidthGBs: bw, TFLOPS: tflops, ComputeCapability: cc,
		}}
	}
	return m
}

func rtx3090() hw.Machine { return machine(24, 64, "NVIDIA GeForce RTX 3090", 936, 71, 8.6) }
func rtx3060() hw.Machine { return machine(12, 32, "NVIDIA GeForce RTX 3060", 360, 26, 8.6) }

func req() Request {
	return Request{Ctx: 8192, KVType: "f16", Batch: 1, MinVerdict: fit.Usable}
}

func opt(params int64, format string, tps float64) Option {
	f, ok := quant.ByName(format)
	if !ok {
		panic("unknown format " + format)
	}
	return Option{
		Model:    arch.Model{Params: params},
		Format:   f,
		Estimate: fit.Estimate{DecodeTPS: tps},
	}
}

// The ranking rule is the tool's opinion, so the comparisons people actually
// face are pinned here rather than left to emerge.
func TestScoreMakesTheRightTrades(t *testing.T) {
	cases := []struct {
		name          string
		winner, loser Option
	}{
		{
			// The whole reason quantization is worth doing.
			"a 32B at Q4_K_M beats an 8B at Q8_0",
			opt(32_000_000_000, "Q4_K_M", 30), opt(8_000_000_000, "Q8_0", 30),
		},
		{
			// The failure the naive ranking produced: biggest file wins.
			"a 14B at Q4_K_M beats a 109B at IQ1_M",
			opt(14_000_000_000, "Q4_K_M", 30), opt(109_000_000_000, "IQ1_M", 7),
		},
		{
			"a 14B at Q4_K_M beats a 32B at Q2_K",
			opt(14_000_000_000, "Q4_K_M", 30), opt(32_000_000_000, "Q2_K", 25),
		},
		{
			// Same model: a marginal quality gain must not buy CPU offload.
			"the same model fast at Q4_K_M beats slow at Q6_K",
			opt(30_000_000_000, "Q4_K_M", 25), opt(30_000_000_000, "Q6_K", 10),
		},
		{
			// But when both are fast, quality wins — speed saturates.
			"when both are fast, the higher quant wins",
			opt(8_000_000_000, "Q6_K", 45), opt(8_000_000_000, "Q4_K_M", 60),
		},
		{
			"a bigger model is not worth dropping to unusable speed",
			opt(8_000_000_000, "Q5_K_M", 40), opt(70_000_000_000, "Q4_K_M", 2),
		},
	}
	for _, c := range cases {
		if Score(c.winner) <= Score(c.loser) {
			t.Errorf("%s: got %.3g vs %.3g", c.name, Score(c.winner), Score(c.loser))
		}
	}
}

// Sub-3-bit quants are available but never volunteered: they are the formats
// people regret, and a default that recommends them is a default that misleads.
func TestDestructiveQuantsAreNotRecommendedByDefault(t *testing.T) {
	opts := Suggest(rtx3060(), req())
	if len(opts) == 0 {
		t.Fatal("a 12GB card should be able to run something")
	}
	for _, o := range opts {
		if o.Format.Quality < DefaultMinQuality {
			t.Errorf("%s was recommended at %s (quality %d), below the default floor",
				o.Model.Name, o.Format.Name, o.Format.Quality)
		}
	}

	// ...but asking for them explicitly works.
	r := req()
	r.MinQuality = 1
	low := Suggest(rtx3060(), r)
	var sawLow bool
	for _, o := range low {
		if o.Format.Quality < DefaultMinQuality {
			sawLow = true
		}
	}
	if !sawLow {
		t.Error("-min-quality should unlock the aggressive quants")
	}
}

// Nothing that needs the CPU to hold half the model should be described as
// running well.
func TestNothingRecommendedIsSecretlySlow(t *testing.T) {
	for _, m := range []hw.Machine{rtx3060(), rtx3090()} {
		for _, o := range Suggest(m, req()) {
			if o.Estimate.DecodeTPS < 7 {
				t.Errorf("%s on %s: %.1f tok/s should not clear the 'usable' bar",
					o.Model.Name, o.Engine.Name, o.Estimate.DecodeTPS)
			}
			if !o.Estimate.Fits {
				t.Errorf("%s was suggested but does not fit", o.Model.Name)
			}
		}
	}
}

// A machine with no GPU must still answer, and must not offer engines that
// require one.
func TestCPUOnlyMachineGetsCPUOnlyAdvice(t *testing.T) {
	m := machine(0, 32, "", 0, 0, 0)
	r := req()
	r.MinVerdict = fit.Sluggish
	for _, o := range Suggest(m, r) {
		if !o.Engine.CanOffloadCPU {
			t.Errorf("%s cannot run without a GPU but was suggested", o.Engine.Name)
		}
	}
	// vLLM must be reported as unavailable rather than silently missing.
	for _, e := range enginesFor(m) {
		if e.name == "vLLM" && e.ok {
			t.Error("vLLM should not be considered runnable with no GPU")
		}
	}
}

// The MoE result is the non-obvious one worth protecting: on a small card a
// 30B that reads 3.3B per token beats dense models half its size, because
// offloading it costs far less per token.
func TestMoEBeatsDenseModelsOfSimilarSizeOnASmallCard(t *testing.T) {
	opts := Suggest(rtx3060(), req())
	var moeRank, denseRank = -1, -1
	for i, o := range opts {
		if o.Model.IsMoE() && moeRank == -1 {
			moeRank = i
		}
		if !o.Model.IsMoE() && o.Model.Params > 20_000_000_000 && denseRank == -1 {
			denseRank = i
		}
	}
	if moeRank == -1 {
		t.Skip("no MoE cleared the bar on this configuration")
	}
	if denseRank != -1 && moeRank > denseRank {
		t.Errorf("the MoE ranked %d, behind a similarly sized dense model at %d", moeRank, denseRank)
	}
}

// Serving changes the question from "how fast for me" to "how many at once",
// and the engine set changes with it.
func TestServingOnlyOffersServingEngines(t *testing.T) {
	r := req()
	r.Serving = true
	opts := Suggest(machine(80, 256, "NVIDIA A100-SXM4-80GB", 2039, 312, 8.0), r)
	if len(opts) == 0 {
		t.Fatal("an 80GB A100 should serve something")
	}
	for _, o := range opts {
		if !o.Engine.Serving {
			t.Errorf("%s is not a serving engine but was suggested in serving mode", o.Engine.Name)
		}
		if o.Estimate.Concurrency < 1 {
			t.Errorf("%s: serving mode must report concurrency", o.Model.Name)
		}
	}
}

// A bigger card should never produce worse advice.
func TestMoreVRAMNeverLowersTheTopRecommendation(t *testing.T) {
	small := Suggest(rtx3060(), req())
	large := Suggest(rtx3090(), req())
	if len(small) == 0 || len(large) == 0 {
		t.Fatal("both machines should have options")
	}
	if Score(large[0]) < Score(small[0]) {
		t.Errorf("24GB top pick (%s %s, %.3g) scores below the 12GB one (%s %s, %.3g)",
			large[0].Model.Name, large[0].Format.Name, Score(large[0]),
			small[0].Model.Name, small[0].Format.Name, Score(small[0]))
	}
}

// Quantizing the KV cache should unlock context, not break the estimate.
func TestQuantizedKVCacheUnlocksLongerContext(t *testing.T) {
	// A model whose trained context exceeds what a 12GB card can cache, so the
	// answer is set by memory rather than clamped to the model's ceiling —
	// Llama 3.1 8B is small enough that both cache types reach 131k and the
	// comparison says nothing.
	model, ok := catalog.Find("Mistral-Nemo-Instruct-2407")
	if !ok {
		t.Fatal("catalogue is missing Mistral Nemo 12B")
	}
	r := req()
	r.Ctx = 32768
	r.Models = []arch.Model{model}

	f16 := Suggest(rtx3060(), r)
	r.KVType = "q8_0"
	q8 := Suggest(rtx3060(), r)

	if len(f16) == 0 || len(q8) == 0 {
		t.Fatal("a 12B should run on a 12GB card at 32k either way")
	}
	if q8[0].Estimate.MaxCtxAtMem <= f16[0].Estimate.MaxCtxAtMem {
		t.Errorf("q8_0 cache should allow more context: %d vs %d",
			q8[0].Estimate.MaxCtxAtMem, f16[0].Estimate.MaxCtxAtMem)
	}
}

// Every catalogue entry must have the fields the maths divides by. A zero here
// silently produces an infinite context or a zero-byte cache.
func TestCatalogueEntriesAreComplete(t *testing.T) {
	for _, m := range catalog.All() {
		if m.Params <= 0 || m.Layers <= 0 || m.Hidden <= 0 || m.Heads <= 0 || m.Vocab <= 0 || m.MaxCtx <= 0 {
			t.Errorf("%s: incomplete architecture %+v", m.ID, m)
		}
		if m.KVHeads <= 0 && m.Attention != arch.MLA {
			t.Errorf("%s: missing kv_heads", m.ID)
		}
		if m.HeadDimension() <= 0 {
			t.Errorf("%s: head dimension resolves to zero", m.ID)
		}
		if m.IsMoE() && m.ActiveParams >= m.Params {
			t.Errorf("%s: an MoE must read fewer parameters than it stores", m.ID)
		}
		if m.EmbeddingParams() >= m.Params {
			t.Errorf("%s: embeddings cannot exceed the whole model", m.ID)
		}
		if m.Attention == arch.MLA && (m.KVLoraRank <= 0 || m.QKRopeHeadDim <= 0) {
			t.Errorf("%s: MLA needs kv_lora_rank and qk_rope_head_dim", m.ID)
		}
	}
}

type engStatus struct {
	name string
	ok   bool
}

func enginesFor(m hw.Machine) []engStatus {
	var out []engStatus
	for _, e := range engine.All {
		ok, _ := e.RunsOn(m)
		out = append(out, engStatus{e.Name, ok})
	}
	return out
}
