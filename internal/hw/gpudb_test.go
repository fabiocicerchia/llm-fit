package hw

import "testing"

// nvidia-smi reports names like "NVIDIA GeForce RTX 4090", and the table is
// matched by substring. Without longest-match the "RTX 4090" row loses to
// whatever shorter row also appears in the string, and the bandwidth — which
// every speed estimate divides by — comes out wrong for the wrong card.
func TestLookupPrefersTheLongestMatch(t *testing.T) {
	cases := []struct {
		reported string
		want     string
		bw       float64
	}{
		{"NVIDIA GeForce RTX 4090", "RTX 4090", 1008},
		{"NVIDIA RTX 4080 SUPER", "RTX 4080 SUPER", 736},
		{"NVIDIA GeForce RTX 4070 Ti SUPER", "RTX 4070 TI SUPER", 672},
		{"NVIDIA GeForce RTX 4070 Ti", "RTX 4070 TI", 504},
		{"NVIDIA GeForce RTX 3090 Ti", "RTX 3090 TI", 1008},
		{"NVIDIA GeForce RTX 3090", "RTX 3090", 936},
		// The 40GB and 80GB A100s have very different bandwidth, and the same
		// substring. Getting this backwards misestimates decode by a third.
		{"NVIDIA A100-SXM4-80GB", "A100-SXM4-80GB", 2039},
		{"NVIDIA A100-PCIE-40GB", "A100", 1555},
		{"NVIDIA H100 PCIe", "H100 PCIE", 2000},
		{"NVIDIA H100 80GB HBM3", "H100", 3350},
	}
	for _, c := range cases {
		spec, ok := Lookup(c.reported)
		if !ok {
			t.Errorf("%s: not found", c.reported)
			continue
		}
		if spec.Match != c.want || spec.BandwidthGBs != c.bw {
			t.Errorf("%s: matched %q at %.0f GB/s, want %q at %.0f",
				c.reported, spec.Match, spec.BandwidthGBs, c.want, c.bw)
		}
	}
}

func TestLookupMissesAreReported(t *testing.T) {
	if _, ok := Lookup("Some Future GPU 9000"); ok {
		t.Error("an unknown card must not silently match something")
	}
}

// Apple bandwidth spans 68 to 819 GB/s across chips whose names are prefixes of
// each other, so "M2" must not match inside "M2 Max" — a twelvefold error.
func TestAppleChipMatchingIsWordBounded(t *testing.T) {
	cases := []struct {
		brand string
		bw    float64
	}{
		{"Apple M1", 68},
		{"Apple M1 Pro", 200},
		{"Apple M1 Max", 400},
		{"Apple M1 Ultra", 800},
		{"Apple M2", 100},
		{"Apple M2 Max", 400},
		{"Apple M3 Max", 400},
		{"Apple M4", 120},
		{"Apple M4 Pro", 273},
		{"Apple M4 Max", 546},
	}
	for _, c := range cases {
		bw, _, ok := lookupApple(c.brand)
		if !ok {
			t.Errorf("%s: not recognised", c.brand)
			continue
		}
		if bw != c.bw {
			t.Errorf("%s: got %.0f GB/s, want %.0f", c.brand, bw, c.bw)
		}
	}
}

// The measurement is the whole reason CPU-offload estimates are trustworthy, so
// it must at least return something physically plausible.
func TestMeasuredBandwidthIsPlausible(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates and copies several hundred MB")
	}
	bw := MeasureRAMBandwidth()
	// Slower than single-channel DDR3 or faster than HBM means the measurement
	// is broken, not that the machine is unusual.
	if bw < 2 || bw > 2000 {
		t.Errorf("measured %.1f GB/s, which is not a real machine", bw)
	}
}
