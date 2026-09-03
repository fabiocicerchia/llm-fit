package hw

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
)

// Memory bandwidth is the number that decides decode speed, and no driver
// reports it: nvidia-smi gives you the name and the capacity and nothing about
// how fast the memory is. So it has to be looked up.
//
// TFLOPS figures are dense fp16 through the tensor cores — not the "with
// sparsity" number vendors print on slides, which is double and which no
// inference engine achieves.
type Spec struct {
	Match             string  `json:"match"` // matched as a substring against the reported name
	VRAMGiB           float64 `json:"vram_gib"`
	BandwidthGBs      float64 `json:"bandwidth_gbs"`
	TFLOPS            float64 `json:"tflops"`
	ComputeCapability float64 `json:"compute_capability,omitempty"`
	Vendor            Vendor  `json:"vendor"`
}

// Apple silicon is unified memory: the bandwidth figure is the whole system's,
// and the GPU can address most of the RAM. The spread is enormous — an M4 Max
// has eight times the bandwidth of a base M1, which is eight times the decode
// speed on the same model.
type appleSpec struct {
	Match        string  `json:"match"`
	BandwidthGBs float64 `json:"bandwidth_gbs"`
	TFLOPS       float64 `json:"tflops"`
}

// specDB is the shape of gpus.json: discrete cards matched by substring, Apple
// chips matched on a word boundary.
type specDB struct {
	Discrete []Spec      `json:"discrete"`
	Apple    []appleSpec `json:"apple"`
}

// The table is a catalogue of hardware, not logic, and it changes on the
// vendors' release schedule rather than this program's — so it lives beside
// the code as data, the way the model catalogue already does. Embedded rather
// than read from disk: a lookup path outside the binary would let a stray file
// change what the tool believes about a card.
//
// Entries stay in the original order — vendor, then generation, newest first —
// and Lookup takes the longest match rather than the first, so "RTX 4090" wins
// over "4090" and "A100 80GB" over "A100".
//
//go:embed gpus.json
var rawSpecs []byte

var (
	specs      []Spec
	appleSpecs []appleSpec
)

func init() {
	var db specDB
	if err := json.Unmarshal(rawSpecs, &db); err != nil {
		// Embedded at build time, so a parse failure is a build error that
		// escaped rather than a runtime condition worth handling.
		panic("hw: embedded gpus.json is invalid: " + err.Error())
	}
	// Every decode estimate divides by bandwidth and sizes against VRAM. A
	// missing key unmarshals to zero in silence, which would report a card as
	// fitting nothing at zero tokens per second instead of failing here.
	for i, s := range db.Discrete {
		if s.Match == "" || s.VRAMGiB <= 0 || s.BandwidthGBs <= 0 || s.Vendor == "" {
			panic(fmt.Sprintf("hw: gpus.json discrete[%d] %q: match, vram_gib, bandwidth_gbs and vendor are all required", i, s.Match))
		}
	}
	for i, s := range db.Apple {
		if s.Match == "" || s.BandwidthGBs <= 0 {
			panic(fmt.Sprintf("hw: gpus.json apple[%d] %q: match and bandwidth_gbs are required", i, s.Match))
		}
	}
	specs, appleSpecs = db.Discrete, db.Apple
}

// Lookup matches a reported device name against the table. Longest match wins,
// so a name containing both "RTX 4090" and "4090" resolves to the specific row.
func Lookup(name string) (Spec, bool) {
	up := strings.ToUpper(name)
	best := -1
	for i, s := range specs {
		if strings.Contains(up, s.Match) {
			if best == -1 || len(s.Match) > len(specs[best].Match) {
				best = i
			}
		}
	}
	if best == -1 {
		return Spec{}, false
	}
	return specs[best], true
}

// Fallbacks for a discrete card the table has never heard of — which is the
// normal state of affairs for the first months of any new generation.
//
// Leaving bandwidth at zero is worse than guessing: every decode estimate
// divides by it, so the whole catalogue comes back at 0 tok/s and `suggest`
// prints nothing at all on a machine that would in fact run most of it. A
// deliberately conservative pair — a little below the slowest current-gen
// discrete card in the table (RTX 4060, 272 GB/s) — under-promises instead,
// and GPU.Estimated marks every number downstream as a guess.
//
// These are calibration knobs, not constants of nature: raise them as the
// floor of the market moves.
const (
	FallbackBandwidthGBs = 300
	FallbackTFLOPS       = 30
)

// EstimateUnknown supplies those fallbacks for a card that missed the table.
// Capacity is deliberately not guessed — memory is the hard constraint, and
// inventing it would report models as fitting when they do not.
func EstimateUnknown(g *GPU) {
	if g.BandwidthGBs == 0 {
		g.BandwidthGBs = FallbackBandwidthGBs
	}
	if g.TFLOPS == 0 {
		g.TFLOPS = FallbackTFLOPS
	}
	g.Estimated = true
}

func lookupApple(chip string) (bandwidth, tflops float64, ok bool) {
	up := strings.ToUpper(chip)
	bestLen := 0
	for _, s := range appleSpecs {
		// "M2 MAX" must beat "M2"; require a word boundary so "M2" does not
		// match inside "M24".
		if containsWord(up, s.Match) && len(s.Match) > bestLen {
			bandwidth, tflops, bestLen = s.BandwidthGBs, s.TFLOPS, len(s.Match)
		}
	}
	return bandwidth, tflops, bestLen > 0
}

func containsWord(hay, needle string) bool {
	i := strings.Index(hay, needle)
	for i >= 0 {
		before := i == 0 || !isAlnum(hay[i-1])
		end := i + len(needle)
		after := end == len(hay) || !isAlnum(hay[end])
		if before && after {
			return true
		}
		next := strings.Index(hay[i+1:], needle)
		if next < 0 {
			return false
		}
		i += 1 + next
	}
	return false
}

func isAlnum(b byte) bool {
	return (b >= '0' && b <= '9') || (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')
}
