// Package advisor turns the arithmetic into a recommendation.
//
// The search is small enough to be exhaustive: every model against every engine
// the machine can run against every quantization that engine can load. A few
// thousand estimates, none of which touch the disk or the network, so there is
// no reason to be clever about pruning.
//
// The ranking rule is the opinion in the tool, and it is a trade rather than a
// maximum. A 32B at Q4_K_M beats an 8B at Q8_0, because the quantization loss
// between those two is far smaller than the capability gap — but the same logic
// taken to its end recommends a 109B at 1.75 bits, which is a broken model that
// happens to be large. So capability is scored *after* discounting for
// quantization damage, and formats below roughly three bits are excluded
// entirely unless asked for.
package advisor

import (
	"sort"
	"strings"

	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/arch"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/catalog"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/engine"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/fit"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/hw"
	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/quant"
)

type Request struct {
	Ctx    int
	KVType string
	Batch  int
	// MinVerdict filters out plans that fit but are not worth running.
	MinVerdict fit.Verdict
	// MinQuality excludes quantizations that damage the model more than the
	// extra parameters are worth. Zero means the default floor.
	MinQuality int
	// Serving switches the goal from single-stream latency to concurrent
	// throughput, which changes both the engine choice and the ranking.
	Serving bool
	Engines []string // restrict to these; empty means all the machine can run
	Models  []arch.Model
}

type Option struct {
	Model    arch.Model
	Engine   engine.Engine
	Format   quant.Format
	Estimate fit.Estimate
	Ctx      int
	KVType   string
}

// Devices converts detected hardware into the memory/bandwidth view the
// estimator works in.
//
// Apple silicon is deliberately not given a separate CPU device: the memory is
// unified, so counting it twice would double the machine's capacity.
func Devices(m hw.Machine) []fit.Device {
	var devs []fit.Device
	for _, g := range m.GPUs {
		devs = append(devs, fit.Device{
			Name: g.Name, BytesFree: g.FreeBytes,
			BandwidthGBs: g.BandwidthGBs, TFLOPS: g.TFLOPS,
		})
	}
	if !m.UnifiedMem {
		free := m.RAMFree
		// Leave headroom: filling system RAM to the brim invites the OOM killer
		// mid-generation, and on Linux it will pick the inference process.
		free = int64(float64(free) * 0.85)
		devs = append(devs, fit.Device{
			Name: "system RAM", BytesFree: free,
			BandwidthGBs: m.RAMBandwidth, IsCPU: true,
		})
	}
	return devs
}

// Suggest returns the best plan per model, ranked, keeping only what clears the
// requested verdict.
func Suggest(m hw.Machine, req Request) []Option {
	models := req.Models
	if len(models) == 0 {
		models = catalog.All()
	}
	devs := Devices(m)

	var out []Option
	for _, model := range models {
		if best, ok := bestFor(model, m, devs, req); ok {
			out = append(out, best)
		}
	}

	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if sa, sb := rank(a, req), rank(b, req); sa != sb {
			return sa > sb
		}
		return a.Estimate.DecodeTPS > b.Estimate.DecodeTPS
	})
	return out
}

// DefaultMinQuality sits just below Q3_K_M and IQ3_M. Below this line the
// model is measurably damaged, and in practice a smaller model at Q4_K_M is
// the better tool — so those formats are available on request but never
// recommended by default.
const DefaultMinQuality = 55

// Score ranks plans, and it is the one place all three axes meet: how capable
// the model is, how much the quantization damaged it, and whether it is fast
// enough to use.
//
// All three are needed. Capability alone recommends a 109B at 1.75 bits.
// Capability and quality alone recommend Q6_K weights spilling half onto the
// CPU at 11 tokens a second, when the same model at Q4_K_M sits entirely in
// VRAM at triple the speed and gives up almost nothing.
func Score(o Option) float64 {
	return float64(o.Model.Params) * qualityFactor(o.Format.Quality) * speedUtility(o.Estimate.DecodeTPS)
}

// rank is Score, plus the concurrency term when the goal is serving.
//
// Concurrency saturates for the same reason speed does. Ranking on it directly
// puts a 0.5B model first, because 742 copies of a tiny model fit — which
// answers a question nobody asked. Past a few dozen simultaneous requests the
// next one is worth far less than a more capable model.
func rank(o Option, req Request) float64 {
	if !req.Serving {
		return Score(o)
	}
	return Score(o) * concurrencyUtility(o.Estimate.Concurrency)
}

func concurrencyUtility(c int) float64 {
	switch {
	case c >= 64:
		return 1.00
	case c >= 32:
		return 0.98
	case c >= 16:
		return 0.95
	case c >= 8:
		return 0.88
	case c >= 4:
		return 0.70
	case c >= 2:
		return 0.45
	case c >= 1:
		return 0.25
	}
	// Nothing left for a second request is not a serving deployment.
	return 0
}

// EffectiveParams is Score without the speed term: what the model is worth
// after quantization, ignoring how fast it runs.
func EffectiveParams(o Option) float64 {
	return float64(o.Model.Params) * qualityFactor(o.Format.Quality)
}

// Speed stops mattering once it exceeds reading pace, and matters enormously
// below it. A flat multiplier would let a marginal quality gain buy an
// unusable model; a saturating one says what a person would: past about 30
// tokens a second, spend the headroom on quality instead.
func speedUtility(tps float64) float64 {
	switch {
	case tps >= 30:
		return 1.00
	case tps >= 20:
		return 0.95
	case tps >= 15:
		return 0.88
	case tps >= 10:
		return 0.72
	case tps >= 7:
		return 0.55
	case tps >= 4:
		return 0.15
	}
	// Below about four tokens a second the plan is not a slower way to use the
	// model, it is a different activity. The penalty is steep on purpose: no
	// amount of extra parameters should let a 2 tok/s plan outrank one that
	// answers at reading speed.
	return 0.03
}

func qualityFactor(q int) float64 {
	switch {
	case q >= 94: // Q6_K and up: indistinguishable from fp16
		return 1.00
	case q >= 84: // Q5_K
		return 0.99
	case q >= 72: // Q4_K_M, AWQ, GPTQ — the knee of the curve
		return 0.96
	case q >= 62: // Q4_0, Q3_K_L
		return 0.88
	case q >= 55: // Q3_K_M, IQ3_M
		return 0.78
	case q >= 45: // Q3_K_S, Q2_K
		return 0.55
	case q >= 35: // IQ2
		return 0.30
	}
	return 0.10 // IQ1: large, and broken
}

// bestFor finds the highest-quality plan for one model that still clears the
// verdict bar.
func bestFor(model arch.Model, m hw.Machine, devs []fit.Device, req Request) (Option, bool) {
	var best Option
	var found bool

	minQ := req.MinQuality
	if minQ == 0 {
		minQ = DefaultMinQuality
	}
	for _, eng := range engine.All {
		if len(req.Engines) > 0 && !named(req.Engines, eng.Name) {
			continue
		}
		if ok, _ := eng.RunsOn(m); !ok {
			continue
		}
		if req.Serving && !eng.Serving {
			continue
		}
		for _, f := range quant.Formats {
			if !eng.Supports(f) || f.Quality < minQ {
				continue
			}
			ctx := req.Ctx
			if ctx > model.MaxCtx {
				ctx = model.MaxCtx
			}
			est := fit.EstimatePlan(fit.Plan{
				Model: model, Format: f, Ctx: ctx, KVType: req.KVType,
				Batch: req.Batch, Engine: eng.Name, Devices: devs,
			}, eng.Engine)

			if !est.Fits || est.Verdict < req.MinVerdict {
				continue
			}
			cand := Option{Model: model, Engine: eng, Format: f, Estimate: est, Ctx: ctx, KVType: req.KVType}
			if !found || better(cand, best, req) {
				best, found = cand, true
			}
		}
	}
	return best, found
}

// better compares two plans for the same model, so the parameter count cancels
// and what is left is the quantization-versus-speed trade.
func better(a, b Option, req Request) bool {
	if sa, sb := rank(a, req), rank(b, req); sa != sb {
		return sa > sb
	}
	// Equal score: take the one that leaves more memory free.
	return a.Estimate.WeightBytes < b.Estimate.WeightBytes
}

// Inspect returns every viable plan for one model, for the drill-down view.
func Inspect(model arch.Model, m hw.Machine, req Request) []Option {
	devs := Devices(m)
	var out []Option
	for _, eng := range engine.All {
		if ok, _ := eng.RunsOn(m); !ok {
			continue
		}
		for _, f := range quant.Formats {
			if !eng.Supports(f) {
				continue
			}
			ctx := req.Ctx
			if ctx > model.MaxCtx {
				ctx = model.MaxCtx
			}
			est := fit.EstimatePlan(fit.Plan{
				Model: model, Format: f, Ctx: ctx, KVType: req.KVType,
				Batch: req.Batch, Engine: eng.Name, Devices: devs,
			}, eng.Engine)
			out = append(out, Option{Model: model, Engine: eng, Format: f, Estimate: est, Ctx: ctx, KVType: req.KVType})
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Estimate.Fits != out[j].Estimate.Fits {
			return out[i].Estimate.Fits
		}
		if out[i].Format.Quality != out[j].Format.Quality {
			return out[i].Format.Quality > out[j].Format.Quality
		}
		return out[i].Estimate.DecodeTPS > out[j].Estimate.DecodeTPS
	})
	return out
}

func named(list []string, name string) bool {
	for _, s := range list {
		if strings.EqualFold(s, name) {
			return true
		}
	}
	return false
}
