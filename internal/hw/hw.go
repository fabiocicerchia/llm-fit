// Package hw works out what the machine actually has.
//
// Free memory is used rather than total wherever it can be read, because total
// is a lie on a machine that is already doing something: a desktop with a
// browser open has a gigabyte of its 24 already gone, and a plan built on the
// nameplate figure will not load.
package hw

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// Vendor is who made a GPU, which decides how it is detected and which
// engines will talk to it.
type Vendor string

// The vendors llm-fit can detect. Unknown is a real device whose vendor
// string did not match any of the others, not the absence of a device.
const (
	NVIDIA  Vendor = "NVIDIA"
	AMD     Vendor = "AMD"
	Intel   Vendor = "Intel"
	Apple   Vendor = "Apple"
	Unknown Vendor = "unknown"
)

const (
	// assumedRAMBandwidthGBs is the last resort when the measurement fails: a
	// mid-range DDR4 figure, deliberately unremarkable. It appears in the
	// warning that reports it, so the number and the sentence cannot drift.
	assumedRAMBandwidthGBs = 50

	// Fallbacks for an Apple chip the table does not know — roughly a base M1,
	// the slowest silicon still worth planning for.
	unknownAppleBandwidthGBs = 100
	unknownAppleTFLOPS       = 5

	// Metal will not hand the GPU all of system memory. The wired limit is
	// about 75% of RAM, a little more on machines with plenty of it.
	metalWiredFraction      = 0.75
	metalWiredFractionLarge = 0.80
	largeUnifiedRAMBytes    = 64 << 30

	// probeTimeout bounds each detection subprocess. A wedged nvidia-smi on a
	// broken driver hangs for ever, and detection must not be what hangs.
	probeTimeout = 5 * time.Second
)

// GPU is one detected accelerator. Bandwidth and TFLOPS come from the table
// in gpudb.go when the driver does not report them -- see Estimated.
type GPU struct {
	Name              string
	Vendor            Vendor
	TotalBytes        int64
	FreeBytes         int64
	BandwidthGBs      float64
	TFLOPS            float64
	ComputeCapability float64
	// Estimated marks a device whose bandwidth came from neither the driver nor
	// the table, so the speed numbers derived from it are guesses.
	Estimated bool
}

// Machine is the host llm-fit is deciding for: its CPU, its RAM, and every
// GPU it could find.
type Machine struct {
	OS           string
	Arch         string
	CPUModel     string
	CPUCores     int
	RAMTotal     int64
	RAMFree      int64
	RAMBandwidth float64
	RAMEstimated bool
	RAMMeasured  bool
	GPUs         []GPU
	UnifiedMem   bool // Apple silicon: the GPU addresses system RAM
	Warnings     []string
}

// Detect inspects the machine. It never fails: an undetectable field comes back
// zero with a warning attached, because a partial answer plus a note beats an
// error on a box where nvidia-smi happens not to be on PATH.
func Detect() Machine {
	m := Machine{OS: runtime.GOOS, Arch: runtime.GOARCH, CPUCores: runtime.NumCPU()}

	switch runtime.GOOS {
	case "linux":
		detectLinuxCPU(&m)
	case "darwin":
		detectDarwin(&m)
	}

	if gpus := detectNvidia(); len(gpus) > 0 {
		m.GPUs = append(m.GPUs, gpus...)
	}
	if gpus := detectAMD(); len(gpus) > 0 {
		m.GPUs = append(m.GPUs, gpus...)
	}

	if m.RAMBandwidth == 0 && !m.UnifiedMem {
		// Measured, not assumed. Costs a few hundred milliseconds and removes
		// the largest error term in every CPU-offload estimate.
		if bw := MeasureRAMBandwidth(); bw > 0 {
			m.RAMBandwidth = bw
			m.RAMMeasured = true
		} else {
			m.RAMBandwidth = assumedRAMBandwidthGBs
			m.RAMEstimated = true
			m.Warnings = append(m.Warnings, fmt.Sprintf(
				"system RAM bandwidth could not be measured; assuming %d GB/s. Pass -ram-bandwidth to correct it — every "+
					"CPU-offload speed figure divides by this",
				assumedRAMBandwidthGBs))
		}
	}
	return m
}

func detectLinuxCPU(m *Machine) {
	if f, err := os.Open("/proc/cpuinfo"); err == nil {
		//nolint:errcheck // read-only probe of /proc; nothing to report a failed close to
		defer func() { _ = f.Close() }()
		s := bufio.NewScanner(f)
		for s.Scan() {
			if name, val, ok := strings.Cut(s.Text(), ":"); ok && strings.TrimSpace(name) == "model name" {
				m.CPUModel = strings.TrimSpace(val)
				break
			}
		}
	}
	if f, err := os.Open("/proc/meminfo"); err == nil {
		//nolint:errcheck // read-only probe of /proc; nothing to report a failed close to
		defer func() { _ = f.Close() }()
		s := bufio.NewScanner(f)
		for s.Scan() {
			key, val, ok := strings.Cut(s.Text(), ":")
			if !ok {
				continue
			}
			kb, err := strconv.ParseInt(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(val), "kB")), 10, 64)
			if err != nil {
				continue
			}
			switch key {
			case "MemTotal":
				m.RAMTotal = kb * 1024
			case "MemAvailable":
				// MemAvailable, not MemFree: cache is reclaimable and counting
				// it as used would understate capacity by tens of gigabytes.
				m.RAMFree = kb * 1024
			}
		}
	}
}

func detectDarwin(m *Machine) {
	m.CPUModel = strings.TrimSpace(run("sysctl", "-n", "machdep.cpu.brand_string"))
	if v := strings.TrimSpace(run("sysctl", "-n", "hw.memsize")); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			m.RAMTotal = n
			m.RAMFree = n
		}
	}
	if runtime.GOARCH != "arm64" {
		return
	}
	m.UnifiedMem = true
	bw, tf, ok := lookupApple(m.CPUModel)
	if !ok {
		m.Warnings = append(m.Warnings, "unrecognised Apple chip "+m.CPUModel+"; memory bandwidth unknown")
		bw, tf = unknownAppleBandwidthGBs, unknownAppleTFLOPS
		m.RAMEstimated = true
	}
	m.RAMBandwidth = bw

	// Exceeding the wired limit does not fail — it silently swaps, which looks
	// like the model working and running at one token per second.
	usable := int64(float64(m.RAMTotal) * metalWiredFraction)
	if m.RAMTotal >= largeUnifiedRAMBytes {
		usable = int64(float64(m.RAMTotal) * metalWiredFractionLarge)
	}
	m.GPUs = append(m.GPUs, GPU{
		Name: m.CPUModel, Vendor: Apple,
		TotalBytes: usable, FreeBytes: usable,
		BandwidthGBs: bw, TFLOPS: tf, Estimated: !ok,
	})
}

func detectNvidia() []GPU {
	out := run("nvidia-smi",
		"--query-gpu=name,memory.total,memory.free,compute_cap",
		"--format=csv,noheader,nounits")
	if strings.TrimSpace(out) == "" {
		return nil
	}
	var gpus []GPU
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			continue
		}
		name := strings.TrimSpace(parts[0])
		g := GPU{Name: name, Vendor: NVIDIA}
		// nvidia-smi reports MiB.
		g.TotalBytes = int64(parseFloat(parts[1]) * 1024 * 1024)
		g.FreeBytes = int64(parseFloat(parts[2]) * 1024 * 1024)
		if len(parts) > 3 {
			g.ComputeCapability = parseFloat(parts[3])
		}
		if spec, ok := Lookup(name); ok {
			g.BandwidthGBs, g.TFLOPS = spec.BandwidthGBs, spec.TFLOPS
			if g.ComputeCapability == 0 {
				g.ComputeCapability = spec.ComputeCapability
			}
		} else {
			EstimateUnknown(&g)
		}
		gpus = append(gpus, g)
	}
	return gpus
}

func detectAMD() []GPU {
	out := run("rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv")
	if strings.TrimSpace(out) == "" {
		return nil
	}
	var gpus []GPU
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if !strings.HasPrefix(line, "card") {
			continue
		}
		f := strings.Split(line, ",")
		if len(f) < 3 {
			continue
		}
		name := strings.TrimSpace(f[1])
		g := GPU{Name: name, Vendor: AMD}
		if spec, ok := Lookup(name); ok {
			g.BandwidthGBs, g.TFLOPS = spec.BandwidthGBs, spec.TFLOPS
			g.TotalBytes = int64(spec.VRAMGiB * (1 << 30))
			g.FreeBytes = g.TotalBytes
		} else {
			// Capacity is still unknown here — rocm-smi's meminfo columns are
			// not parsed — so this card contributes bandwidth but no memory,
			// and nothing will be recommended onto it.
			EstimateUnknown(&g)
		}
		gpus = append(gpus, g)
	}
	return gpus
}

// VRAMBytes is the total free GPU memory across all cards.
func (m Machine) VRAMBytes() int64 {
	var n int64
	for _, g := range m.GPUs {
		n += g.FreeBytes
	}
	return n
}

// HasGPU reports whether there is a real accelerator, as opposed to the CPU
// standing in for one.
func (m Machine) HasGPU() bool { return len(m.GPUs) > 0 }

func run(name string, args ...string) string {
	if _, err := exec.LookPath(name); err != nil {
		return ""
	}
	done := make(chan string, 1)
	// The timeout below is what bounds this probe; a context would duplicate
	// it, and the goroutine already owns the process.
	ctx, cancel := context.WithTimeout(context.Background(), probeTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	go func() {
		out, err := cmd.Output()
		if err != nil {
			done <- ""
			return
		}
		done <- string(out)
	}()
	select {
	case s := <-done:
		return s
	case <-time.After(probeTimeout):
		// The probe already timed out and its output is discarded; a process
		// that will not die is not something this detection can act on.
		_ = cmd.Process.Kill() //nolint:errcheck // see above
		return ""
	}
}

func parseFloat(s string) float64 {
	// A field that is not a number reads as zero, which every caller already
	// treats as "the driver did not tell us".
	f, _ := strconv.ParseFloat(strings.TrimSpace(s), 64) //nolint:errcheck // see above
	return f
}
