// Package validate compares predicted tokens/sec against measured ones.
//
// WHY: the arithmetic is documented and the constants come from published
// figures, but only ONE end-to-end number has ever been checked against a
// stopwatch — the Qwen3-30B-A3B run that produced cpuGatherEfficiency, which
// exists precisely because the constant before it had no measurement behind it
// at all and was out by 7x. Every other estimate the tool prints rests on
// arithmetic nobody has held up against a clock.
//
// This does not produce measurements. It CANNOT: no model runs here. What it
// does is turn a file of measurements into the table the question actually
// needs — predicted, measured, ratio, and whether the error is systematic or
// scattered — so that collecting three runs is an afternoon rather than a
// project, and so that the answer lands somewhere a reader can find it.
//
// THE FILE IS THE ARTEFACT. bench/observations.tsv holds one row per measured
// run, with the machine and the settings written down beside the number,
// because a tokens/sec figure without its context is not evidence of anything.
// It ships with the one row that exists. Two more and the accuracy band stops
// being a guess.
package validate

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/fabiocicerchia/llm-fit/internal/catalog"
	"github.com/fabiocicerchia/llm-fit/internal/engine"
	"github.com/fabiocicerchia/llm-fit/internal/fit"
	"github.com/fabiocicerchia/llm-fit/internal/quant"
)

// Observation is one measured run, as recorded in bench/observations.tsv.
//
// Every field that changes the arithmetic is here. A row missing one of them is
// refused rather than defaulted: a default that happens to be wrong turns a
// measurement into a data point arguing for the wrong correction, which is
// worse than having no row.
type Observation struct {
	Model    string  // catalog name or HF id
	Quant    string  // q4_k_m, q8_0, ...
	Ctx      int     // context length the run used
	KVType   string  // f16, q8_0, ...
	Batch    int     //
	Engine   string  // llama.cpp, vllm, ...
	GPU      string  // free-text, for the report only
	VRAMGB   float64 // FREE VRAM, not installed
	VRAMBWGB float64 // GB/s
	VRAMTF   float64 // dense fp16 TFLOPS
	RAMGB    float64 // free system RAM
	RAMBWGB  float64 // MEASURED memcpy bandwidth, not the spec sheet
	Measured float64 // tokens/sec, decode
	Note     string
}

// Result is one comparison.
type Result struct {
	Obs       Observation
	Predicted float64
	// Ratio is predicted/measured. 1.0 is perfect; above 1 means the tool is
	// optimistic, which is the direction that matters — an overestimate puts a
	// model at the top of a list on a machine where it is unusable.
	Ratio float64
	Err   string
}

// Summary is what the docs should say.
type Summary struct {
	Results []Result
	// Ratios of the rows that could be predicted at all.
	Ratios []float64
	// GeoMean of the ratios. Geometric, because these are ratios: a 2x
	// overestimate and a 2x underestimate should cancel, and with an arithmetic
	// mean they read as a 25% optimistic bias that is not there.
	GeoMean float64
	// Spread is the ratio of the worst overestimate to the worst
	// underestimate — the width of the band, in the only units that mean
	// anything for a multiplicative model.
	Spread  float64
	Skipped int
}

// Parse reads observations.tsv. Tab-separated with a header, because the
// numbers are typed in by hand from a terminal and a comma is the one
// separator a model name might contain.
func Parse(r io.Reader) ([]Observation, error) {
	var out []Observation
	sc := bufio.NewScanner(r)
	line := 0
	var header []string
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		fields := strings.Split(text, "\t")
		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
		}
		if header == nil {
			header = fields
			continue
		}
		if len(fields) != len(header) {
			return nil, fmt.Errorf("line %d: %d column(s), header has %d",
				line, len(fields), len(header))
		}
		var o Observation
		for i, name := range header {
			if err := assign(&o, name, fields[i]); err != nil {
				return nil, fmt.Errorf("line %d, column %q: %w", line, name, err)
			}
		}
		if o.Measured <= 0 {
			return nil, fmt.Errorf("line %d: measured_tps must be positive", line)
		}
		out = append(out, o)
	}
	return out, sc.Err()
}

func assign(o *Observation, name, v string) error {
	num := func() (float64, error) { return strconv.ParseFloat(v, 64) }
	i := func() (int, error) { return strconv.Atoi(v) }
	var err error
	switch name {
	case "model":
		o.Model = v
	case "quant":
		o.Quant = v
	case "kv":
		o.KVType = v
	case "engine":
		o.Engine = v
	case "gpu":
		o.GPU = v
	case "note":
		o.Note = v
	case "ctx":
		o.Ctx, err = i()
	case "batch":
		o.Batch, err = i()
	case "vram_gb":
		o.VRAMGB, err = num()
	case "vram_bw_gbs":
		o.VRAMBWGB, err = num()
	case "vram_tflops":
		o.VRAMTF, err = num()
	case "ram_gb":
		o.RAMGB, err = num()
	case "ram_bw_gbs":
		o.RAMBWGB, err = num()
	case "measured_tps":
		o.Measured, err = num()
	default:
		return fmt.Errorf("unknown column")
	}
	return err
}

const gib = 1 << 30

// Predict rebuilds the plan the observation describes and estimates it. Any
// piece it cannot resolve is an error rather than a guess, for the same reason
// a missing column is: a substituted default produces a number that looks like
// a prediction and is not one.
func Predict(o Observation) (float64, error) {
	m, ok := catalog.Find(o.Model)
	if !ok {
		return 0, fmt.Errorf("no model named %q in the catalog", o.Model)
	}
	// Upper-cased first: the table's names are Q4_K_M and these rows are typed
	// in by hand from a terminal where the file was q4_k_m.gguf.
	f, ok := quant.ByName(strings.ToUpper(o.Quant))
	if !ok {
		return 0, fmt.Errorf("no quantisation named %q", o.Quant)
	}
	eng, ok := engine.ByName(o.Engine)
	if !ok {
		return 0, fmt.Errorf("no engine named %q — see `llm-fit engines`", o.Engine)
	}
	var devices []fit.Device
	if o.VRAMGB > 0 {
		devices = append(devices, fit.Device{
			Name:         o.GPU,
			BytesFree:    int64(o.VRAMGB * gib),
			BandwidthGBs: o.VRAMBWGB,
			TFLOPS:       o.VRAMTF,
		})
	}
	if o.RAMGB > 0 {
		devices = append(devices, fit.Device{
			Name: "system RAM", IsCPU: true,
			BytesFree:    int64(o.RAMGB * gib),
			BandwidthGBs: o.RAMBWGB,
		})
	}
	if len(devices) == 0 {
		return 0, fmt.Errorf("no devices: give vram_gb or ram_gb")
	}
	batch := o.Batch
	if batch < 1 {
		batch = 1
	}
	est := fit.EstimatePlan(fit.Plan{
		Model: m, Format: f, Ctx: o.Ctx, KVType: o.KVType,
		Batch: batch, Engine: o.Engine, Devices: devices,
	}, eng.Engine)
	if est.DecodeTPS <= 0 {
		return 0, fmt.Errorf("the plan does not fit, so there is no rate to compare")
	}
	return est.DecodeTPS, nil
}

// Compare runs every observation and summarises.
func Compare(obs []Observation) Summary {
	var s Summary
	for _, o := range obs {
		r := Result{Obs: o}
		p, err := Predict(o)
		if err != nil {
			r.Err = err.Error()
			s.Skipped++
		} else {
			r.Predicted = p
			r.Ratio = p / o.Measured
			s.Ratios = append(s.Ratios, r.Ratio)
		}
		s.Results = append(s.Results, r)
	}
	s.GeoMean = geoMean(s.Ratios)
	s.Spread = spread(s.Ratios)
	return s
}

// geoMean of the ratios. Geometric because these ARE ratios: a 2x overestimate
// and a 2x underestimate must cancel, and an arithmetic mean reports them as a
// 25% optimistic bias that does not exist.
func geoMean(rs []float64) float64 {
	if len(rs) == 0 {
		return 0
	}
	sum := 0.0
	for _, r := range rs {
		if r <= 0 {
			return 0
		}
		sum += math.Log(r)
	}
	return math.Exp(sum / float64(len(rs)))
}

func spread(rs []float64) float64 {
	if len(rs) < 2 {
		return 0
	}
	sorted := append([]float64(nil), rs...)
	sort.Float64s(sorted)
	if sorted[0] <= 0 {
		return 0
	}
	return sorted[len(sorted)-1] / sorted[0]
}

// minObservations is how many rows it takes before the band means anything.
// Below this the report says so instead of quoting a number: one point cannot
// distinguish a systematic bias from a machine that was swapping, and quoting
// an accuracy band from it would be exactly the false precision the issue asks
// to remove.
const minObservations = 3

// Text renders the table the issue asks for.
func (s Summary) Text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-22s %-8s %-10s %9s %9s %8s\n",
		"model", "quant", "engine", "predicted", "measured", "ratio")
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 72))
	for _, r := range s.Results {
		if r.Err != "" {
			fmt.Fprintf(&b, "%-22s %-8s %-10s %s\n",
				trunc(r.Obs.Model, 22), r.Obs.Quant, r.Obs.Engine, "SKIPPED: "+r.Err)
			continue
		}
		fmt.Fprintf(&b, "%-22s %-8s %-10s %9.1f %9.1f %7.2fx\n",
			trunc(r.Obs.Model, 22), r.Obs.Quant, r.Obs.Engine,
			r.Predicted, r.Obs.Measured, r.Ratio)
	}

	n := len(s.Ratios)
	fmt.Fprintf(&b, "\n%d observation(s) compared", n)
	if s.Skipped > 0 {
		fmt.Fprintf(&b, ", %d skipped", s.Skipped)
	}
	fmt.Fprintln(&b)

	if n == 0 {
		fmt.Fprintf(&b, "\nNothing to say: no row could be predicted.\n")
		return b.String()
	}

	fmt.Fprintf(&b, "geometric mean ratio  %.2fx  (>1 means the tool is optimistic)\n", s.GeoMean)
	if s.Spread > 0 {
		fmt.Fprintf(&b, "spread                %.1fx  (worst over to worst under)\n", s.Spread)
	}

	if n < minObservations {
		fmt.Fprintf(&b, "\nNOT AN ACCURACY BAND. %d observation(s) cannot separate a systematic\n"+
			"bias from one machine that was swapping, and quoting a band from it would\n"+
			"be exactly the false precision this is meant to remove. %d more run(s) on\n"+
			"different hardware and it becomes one.\n", n, minObservations-n)
		return b.String()
	}

	switch {
	case s.GeoMean > 1.3:
		fmt.Fprintf(&b, "\nSYSTEMATIC OVERESTIMATE. The tool is optimistic by %.0f%% on average,\n"+
			"which is the direction that matters: an overestimate ranks a model first on\n"+
			"a machine where it is unusable. The efficiency constants in internal/fit\n"+
			"are what to correct.\n", (s.GeoMean-1)*100)
	case s.GeoMean < 0.77:
		fmt.Fprintf(&b, "\nSYSTEMATIC UNDERESTIMATE, by %.0f%% on average. Less harmful than the\n"+
			"other direction — it hides models that would have worked — but still a\n"+
			"constant to correct.\n", (1-s.GeoMean)*100)
	default:
		fmt.Fprintf(&b, "\nNo systematic bias at this sample size: the mean ratio is within 30%%\n"+
			"of 1. The SPREAD is the number to quote in the docs, not the mean.\n")
	}
	return b.String()
}

// Band is the accuracy band for the docs, or the honest refusal to state one.
func (s Summary) Band() string {
	n := len(s.Ratios)
	if n < minObservations {
		return fmt.Sprintf("unknown — %d measured run(s), and it takes %d before a band "+
			"means anything", n, minObservations)
	}
	lo, hi := 1/s.GeoMean, 1/s.GeoMean
	for _, r := range s.Ratios {
		if v := 1 / r; v < lo {
			lo = v
		} else if v > hi {
			hi = v
		}
	}
	return fmt.Sprintf("measured/predicted between %.2fx and %.2fx across %d run(s)", lo, hi, n)
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
