package validate

import (
	"math"
	"os"
	"strings"
	"testing"
)

// The tests that matter here are not "does the arithmetic work" — internal/fit
// has those. They are "does this report refuse to overclaim", because a
// validation harness that reports a confident accuracy band from one data point
// is worse than having no harness at all: it converts an unknown into a
// published number.

func TestOneObservationIsNotAnAccuracyBand(t *testing.T) {
	s := Compare([]Observation{qwen()})
	if len(s.Ratios) != 1 {
		t.Fatalf("the shipped observation did not predict: %+v", s.Results)
	}
	text := s.Text()
	if !strings.Contains(text, "NOT AN ACCURACY BAND") {
		t.Errorf("one point was reported as a band:\n%s", text)
	}
	if !strings.Contains(s.Band(), "unknown") {
		t.Errorf("Band() quoted a figure from one run: %s", s.Band())
	}
}

func TestThreeObservationsProduceABand(t *testing.T) {
	o := qwen()
	s := Compare([]Observation{o, o, o})
	if strings.Contains(s.Text(), "NOT AN ACCURACY BAND") {
		t.Error("three observations still refused to say anything")
	}
	if strings.Contains(s.Band(), "unknown") {
		t.Errorf("Band() still refuses at three runs: %s", s.Band())
	}
}

// ---- the shipped measurement ------------------------------------------------

func TestTheOneRealMeasurementIsReproducedByTheModel(t *testing.T) {
	// The Qwen3-30B-A3B run that produced cpuGatherEfficiency. If a change to
	// the arithmetic moves this far, it has moved away from the only number
	// anybody actually held a stopwatch to.
	f, err := os.Open("../../bench/observations.tsv")
	if err != nil {
		t.Skipf("no observations file: %v", err)
	}
	defer f.Close()
	obs, err := Parse(f)
	if err != nil {
		t.Fatalf("the shipped observations file does not parse: %v", err)
	}
	if len(obs) == 0 {
		t.Fatal("the observations file has no rows; the one real measurement is gone")
	}
	s := Compare(obs)
	for _, r := range s.Results {
		if r.Err != "" {
			t.Fatalf("%s could not be predicted: %s", r.Obs.Model, r.Err)
		}
		// Wide on purpose. This is a regression guard on the constants, not a
		// claim about accuracy: the measured figure is explicitly a floor (the
		// machine was still swapping), so predicting somewhat above it is the
		// expected shape.
		if r.Ratio < 0.5 || r.Ratio > 2.0 {
			t.Errorf("%s: predicted %.1f vs measured %.1f (%.2fx) — the arithmetic has "+
				"moved away from the one measurement behind it",
				r.Obs.Model, r.Predicted, r.Obs.Measured, r.Ratio)
		}
	}
}

// ---- refusing to guess ------------------------------------------------------

func TestAnUnknownModelQuantOrEngineIsSkippedNotDefaulted(t *testing.T) {
	// A substituted default produces a number that looks like a prediction and
	// is not one, and it would then argue for a correction to the constants.
	for _, tc := range []struct {
		name string
		mut  func(*Observation)
	}{
		{"model", func(o *Observation) { o.Model = "not-a-model" }},
		{"quant", func(o *Observation) { o.Quant = "q9_k_xxl" }},
		{"engine", func(o *Observation) { o.Engine = "not-an-engine" }},
	} {
		o := qwen()
		tc.mut(&o)
		if _, err := Predict(o); err == nil {
			t.Errorf("an unknown %s was silently defaulted", tc.name)
		}
	}
}

func TestAnObservationWithNoDevicesIsRefused(t *testing.T) {
	o := qwen()
	o.VRAMGB, o.RAMGB = 0, 0
	if _, err := Predict(o); err == nil {
		t.Fatal("a run on no hardware produced a prediction")
	}
}

// ---- the statistics ---------------------------------------------------------

func TestTheMeanIsGeometricSoOverAndUnderCancel(t *testing.T) {
	// The reason this matters: with an ARITHMETIC mean, a 2x overestimate and a
	// 2x underestimate average to 1.25 and the report claims a 25% optimistic
	// bias that is not there.
	got := geoMean([]float64{2, 0.5})
	if math.Abs(got-1) > 1e-9 {
		t.Fatalf("2x over and 2x under averaged to %.3f, not 1", got)
	}
	if arithmetic := (2 + 0.5) / 2; math.Abs(arithmetic-1) < 1e-9 {
		t.Fatal("the arithmetic mean would have been right, so this test proves nothing")
	}
}

func TestSpreadIsWorstOverToWorstUnder(t *testing.T) {
	if got := spread([]float64{0.5, 1.0, 2.0}); math.Abs(got-4) > 1e-9 {
		t.Fatalf("spread of 0.5..2.0 is %.2f, want 4", got)
	}
	if spread([]float64{1.1}) != 0 {
		t.Error("one ratio has no spread")
	}
}

func TestASystematicOverestimateIsCalledOutAsTheDangerousDirection(t *testing.T) {
	// An overestimate ranks a model first on a machine where it is unusable.
	// An underestimate hides one that would have worked. They are not the same
	// mistake and the report must not describe them as if they were.
	over := Summary{Ratios: []float64{2, 2, 2}, GeoMean: 2, Spread: 1}
	under := Summary{Ratios: []float64{0.5, 0.5, 0.5}, GeoMean: 0.5, Spread: 1}
	if !strings.Contains(over.Text(), "SYSTEMATIC OVERESTIMATE") {
		t.Error("a 2x optimistic model was not called out")
	}
	if !strings.Contains(over.Text(), "unusable") {
		t.Error("the report does not say why optimism is the worse direction")
	}
	if !strings.Contains(under.Text(), "SYSTEMATIC UNDERESTIMATE") {
		t.Error("a 2x pessimistic model was not called out")
	}
}

// ---- the file format --------------------------------------------------------

func TestParseRefusesARowThatIsMissingAColumn(t *testing.T) {
	// A short row silently zeroing a field is how a measurement becomes a data
	// point arguing for the wrong correction.
	_, err := Parse(strings.NewReader(
		"model\tquant\tmeasured_tps\n" +
			"Qwen/Qwen3-30B-A3B\tq4_k_m\n"))
	if err == nil {
		t.Fatal("a short row was accepted")
	}
}

func TestParseRefusesAnUnknownColumn(t *testing.T) {
	_, err := Parse(strings.NewReader("model\ttps\nx\t1\n"))
	if err == nil {
		t.Fatal("an unknown column was accepted, so a typo'd header would be ignored")
	}
}

func TestParseRefusesANonPositiveRate(t *testing.T) {
	_, err := Parse(strings.NewReader("model\tmeasured_tps\nx\t0\n"))
	if err == nil {
		t.Fatal("a zero tokens/sec row was accepted")
	}
}

func TestCommentsAndBlankLinesAreSkipped(t *testing.T) {
	obs, err := Parse(strings.NewReader(
		"# how to add a row\n\nmodel\tmeasured_tps\n# a note\nQwen/Qwen3-30B-A3B\t2.4\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(obs) != 1 {
		t.Fatalf("got %d observation(s), want 1", len(obs))
	}
}

func TestQuantNamesAreCaseInsensitive(t *testing.T) {
	// The table says Q4_K_M; the file on disk was q4_k_m.gguf, and these rows
	// are typed in by hand from a terminal.
	lower, upper := qwen(), qwen()
	lower.Quant, upper.Quant = "q4_k_m", "Q4_K_M"
	a, err := Predict(lower)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Predict(upper)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatalf("case changed the prediction: %.3f vs %.3f", a, b)
	}
}

func qwen() Observation {
	return Observation{
		Model: "Qwen/Qwen3-30B-A3B", Quant: "q4_k_m", Ctx: 16384, KVType: "q8_0",
		Batch: 1, Engine: "llama.cpp", GPU: "RTX 3060 12GB",
		VRAMGB: 11.5, VRAMBWGB: 360, VRAMTF: 12.7,
		RAMGB: 28, RAMBWGB: 43, Measured: 2.4,
	}
}
