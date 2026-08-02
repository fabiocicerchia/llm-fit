package hw

import "strings"

// Memory bandwidth is the number that decides decode speed, and no driver
// reports it: nvidia-smi gives you the name and the capacity and nothing about
// how fast the memory is. So it has to be looked up.
//
// TFLOPS figures are dense fp16 through the tensor cores — not the "with
// sparsity" number vendors print on slides, which is double and which no
// inference engine achieves.
type Spec struct {
	Match             string // matched as a substring against the reported name
	VRAMGiB           float64
	BandwidthGBs      float64
	TFLOPS            float64
	ComputeCapability float64
	Vendor            Vendor
}

// Ordered longest-match-first at lookup time, so "RTX 4090" wins over "4090"
// and "A100 80GB" over "A100".
var specs = []Spec{
	// NVIDIA datacentre
	{"H200", 141, 4800, 989, 9.0, NVIDIA},
	{"H100 NVL", 94, 3900, 989, 9.0, NVIDIA},
	{"H100 PCIE", 80, 2000, 756, 9.0, NVIDIA},
	{"H100", 80, 3350, 989, 9.0, NVIDIA},
	{"A100-SXM4-80GB", 80, 2039, 312, 8.0, NVIDIA},
	{"A100 80GB", 80, 2039, 312, 8.0, NVIDIA},
	{"A100", 40, 1555, 312, 8.0, NVIDIA},
	{"L40S", 48, 864, 181, 8.9, NVIDIA},
	{"L40", 48, 864, 181, 8.9, NVIDIA},
	{"L4", 24, 300, 121, 8.9, NVIDIA},
	{"A40", 48, 696, 149, 8.6, NVIDIA},
	{"A30", 24, 933, 165, 8.0, NVIDIA},
	{"A10G", 24, 600, 125, 8.6, NVIDIA},
	{"A10", 24, 600, 125, 8.6, NVIDIA},
	{"V100", 32, 900, 125, 7.0, NVIDIA},
	{"T4", 16, 320, 65, 7.5, NVIDIA},

	// NVIDIA workstation
	{"RTX 6000 ADA", 48, 960, 182, 8.9, NVIDIA},
	{"RTX A6000", 48, 768, 155, 8.6, NVIDIA},
	{"RTX A5000", 24, 768, 111, 8.6, NVIDIA},
	{"RTX A4000", 16, 448, 77, 8.6, NVIDIA},

	// NVIDIA consumer — Blackwell
	{"RTX 5090", 32, 1792, 210, 12.0, NVIDIA},
	{"RTX 5080", 16, 960, 113, 12.0, NVIDIA},
	{"RTX 5070 TI", 16, 896, 88, 12.0, NVIDIA},
	{"RTX 5070", 12, 672, 62, 12.0, NVIDIA},
	// Ada
	{"RTX 4090", 24, 1008, 165, 8.9, NVIDIA},
	{"RTX 4080 SUPER", 16, 736, 104, 8.9, NVIDIA},
	{"RTX 4080", 16, 717, 98, 8.9, NVIDIA},
	{"RTX 4070 TI SUPER", 16, 672, 88, 8.9, NVIDIA},
	{"RTX 4070 TI", 12, 504, 80, 8.9, NVIDIA},
	{"RTX 4070", 12, 504, 59, 8.9, NVIDIA},
	{"RTX 4060 TI", 16, 288, 44, 8.9, NVIDIA},
	{"RTX 4060", 8, 272, 30, 8.9, NVIDIA},
	// Ampere
	{"RTX 3090 TI", 24, 1008, 80, 8.6, NVIDIA},
	{"RTX 3090", 24, 936, 71, 8.6, NVIDIA},
	{"RTX 3080 TI", 12, 912, 68, 8.6, NVIDIA},
	{"RTX 3080", 10, 760, 60, 8.6, NVIDIA},
	{"RTX 3070", 8, 448, 40, 8.6, NVIDIA},
	{"RTX 3060 TI", 8, 448, 32, 8.6, NVIDIA},
	{"RTX 3060", 12, 360, 26, 8.6, NVIDIA},
	// Turing
	{"RTX 2080 TI", 11, 616, 27, 7.5, NVIDIA},
	{"GTX 1080 TI", 11, 484, 0, 6.1, NVIDIA},

	// AMD
	{"MI300X", 192, 5300, 1307, 0, AMD},
	{"MI250X", 128, 3277, 383, 0, AMD},
	{"MI210", 64, 1638, 181, 0, AMD},
	{"RX 7900 XTX", 24, 960, 123, 0, AMD},
	{"RX 7900 XT", 20, 800, 103, 0, AMD},
	{"RX 6800", 16, 512, 32, 0, AMD},

	// Intel
	{"ARC A770", 16, 560, 39, 0, Intel},
	{"ARC B580", 12, 456, 46, 0, Intel},
}

// Apple silicon is unified memory: the bandwidth figure is the whole system's,
// and the GPU can address most of the RAM. The spread is enormous — an M4 Max
// has eight times the bandwidth of a base M1, which is eight times the decode
// speed on the same model.
var appleSpecs = []struct {
	Match        string
	BandwidthGBs float64
	TFLOPS       float64
}{
	{"M4 MAX", 546, 34},
	{"M4 PRO", 273, 17},
	{"M4", 120, 9},
	{"M3 ULTRA", 819, 57},
	{"M3 MAX", 400, 28},
	{"M3 PRO", 150, 14},
	{"M3", 100, 7},
	{"M2 ULTRA", 800, 54},
	{"M2 MAX", 400, 27},
	{"M2 PRO", 200, 13.6},
	{"M2", 100, 6.8},
	{"M1 ULTRA", 800, 42},
	{"M1 MAX", 400, 21},
	{"M1 PRO", 200, 10.4},
	{"M1", 68, 5.2},
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
