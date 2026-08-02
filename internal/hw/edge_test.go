package hw

import "testing"

// A card newer than the spec table is the normal case for the first months of
// any generation. Leaving its bandwidth at zero makes every decode estimate
// downstream come out at 0 tok/s, so `suggest` reports that a brand-new 5090
// can run nothing at all.
func TestUnknownCardGetsAConservativeEstimateNotZero(t *testing.T) {
	g := GPU{Name: "NVIDIA GeForce RTX 9090", Vendor: NVIDIA, TotalBytes: 32 << 30, FreeBytes: 30 << 30}
	EstimateUnknown(&g)

	if g.BandwidthGBs <= 0 || g.TFLOPS <= 0 {
		t.Fatalf("unknown card left unusable: bandwidth=%v tflops=%v", g.BandwidthGBs, g.TFLOPS)
	}
	if !g.Estimated {
		t.Error("estimated figures must be marked, or the CLI presents a guess as a measurement")
	}
	// Under-promise: guessing high would recommend models that then crawl.
	if g.BandwidthGBs > 400 {
		t.Errorf("fallback bandwidth %v GB/s is optimistic; it should sit near the slowest current card", g.BandwidthGBs)
	}
	// Capacity is the hard constraint and must never be invented.
	if g.TotalBytes != 32<<30 || g.FreeBytes != 30<<30 {
		t.Error("EstimateUnknown must not touch memory capacity")
	}
}

// A card the table *does* know keeps its real figures.
func TestKnownCardIsNotOverwrittenByTheFallback(t *testing.T) {
	spec, ok := Lookup("NVIDIA GeForce RTX 4090")
	if !ok {
		t.Fatal("RTX 4090 missing from the spec table")
	}
	g := GPU{Name: "NVIDIA GeForce RTX 4090", BandwidthGBs: spec.BandwidthGBs, TFLOPS: spec.TFLOPS}
	EstimateUnknown(&g)
	if g.BandwidthGBs != 1008 {
		t.Errorf("known bandwidth overwritten: got %v, want 1008", g.BandwidthGBs)
	}
}

// The degenerate machine: something reported a device with nothing on it.
// Nothing here should panic or divide by zero.
func TestZeroCapacityMachineIsInert(t *testing.T) {
	m := Machine{
		GPUs:    []GPU{{Name: "Mystery", TotalBytes: 0, FreeBytes: 0}},
		RAMFree: 0, RAMTotal: 0,
	}
	if got := m.VRAMBytes(); got != 0 {
		t.Errorf("VRAMBytes on an empty machine = %d, want 0", got)
	}
}

// Lookup is substring-matched, so the empty string would otherwise match the
// first row in the table and hand back an H200.
func TestLookupDoesNotMatchEmptyOrJunkNames(t *testing.T) {
	for _, name := range []string{"", "   ", "definitely not a gpu"} {
		if spec, ok := Lookup(name); ok {
			t.Errorf("Lookup(%q) matched %q", name, spec.Match)
		}
	}
}
