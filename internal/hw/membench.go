package hw

import (
	"runtime"
	"time"
)

// System RAM bandwidth cannot be read without root — dmidecode knows the DIMM
// speed, nothing in /proc or /sys does — and it is the number every CPU-offload
// estimate divides by. A table keyed on CPU model would be a guess about the
// memory fitted, not the CPU: the same Ryzen runs with one DDR4 stick or four
// DDR5 ones, a 4x spread.
//
// So measure it. A few hundred milliseconds of memcpy over buffers far larger
// than L3 gives the real figure for this machine as configured, which beats any
// lookup table.
//
// This measures achievable copy bandwidth rather than the theoretical peak,
// which is the right quantity: it is what an inference engine also gets.
func MeasureRAMBandwidth() float64 {
	const (
		// Comfortably past any consumer L3 (96MB on the largest X3D parts), so
		// the measurement is of DRAM rather than cache.
		perThread = 192 << 20
		rounds    = 3
	)
	threads := runtime.NumCPU()
	if threads > 8 {
		// Past about eight threads the controller saturates and extra threads
		// only add scheduling noise.
		threads = 8
	}
	if threads < 1 {
		threads = 1
	}

	src := make([][]byte, threads)
	dst := make([][]byte, threads)
	for i := range src {
		src[i] = make([]byte, perThread)
		dst[i] = make([]byte, perThread)
		// Touch every page so the measurement is not dominated by first-touch
		// page faults.
		for j := 0; j < perThread; j += 4096 {
			src[i][j] = byte(j)
		}
	}

	best := 0.0
	for r := 0; r < rounds; r++ {
		done := make(chan struct{}, threads)
		start := time.Now()
		for i := 0; i < threads; i++ {
			go func(i int) {
				copy(dst[i], src[i])
				done <- struct{}{}
			}(i)
		}
		for i := 0; i < threads; i++ {
			<-done
		}
		elapsed := time.Since(start).Seconds()
		if elapsed <= 0 {
			continue
		}
		// A copy both reads and writes, so it moves twice the buffer size.
		gbps := 2 * float64(threads) * float64(perThread) / elapsed / 1e9
		if gbps > best {
			best = gbps
		}
	}
	return best
}
