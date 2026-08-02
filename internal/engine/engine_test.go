package engine

import (
	"strings"
	"testing"

	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/hw"
)

// The refusal message quotes the compute capability back at the user, so it has
// to be the number they will read off nvidia-smi. Truncating to one decimal by
// hand turned 8.6 into "8.5" (8.6-8 is 0.5999999999999996 in binary floating
// point), which sends someone hunting for a driver problem they do not have.
func TestComputeCapabilityIsPrintedExactly(t *testing.T) {
	cases := map[float64]string{
		8.6: "8.6", 8.9: "8.9", 7.5: "7.5", 12.0: "12", 9.0: "9", 6.1: "6.1",
	}
	for in, want := range cases {
		if got := trim(in); got != want {
			t.Errorf("trim(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestRunsOnRejectsTooOldACardAndSaysWhy(t *testing.T) {
	tensorrt, ok := ByName("TensorRT-LLM") // needs cc 8.0
	if !ok {
		t.Fatal("TensorRT-LLM missing from the engine table")
	}
	// A V100 is cc 7.0: new enough for vLLM, too old for TensorRT-LLM.
	v100 := hw.Machine{GPUs: []hw.GPU{{Name: "Tesla V100", Vendor: hw.NVIDIA, ComputeCapability: 7.0}}}

	runs, why := tensorrt.RunsOn(v100)
	if runs {
		t.Fatal("TensorRT-LLM accepted a compute capability 7.0 card")
	}
	for _, want := range []string{"8", "7"} {
		if !strings.Contains(why, want) {
			t.Errorf("refusal %q does not name capability %s", why, want)
		}
	}
}

func TestAppleOnlyAndNvidiaOnlyEnginesRefuseTheWrongVendor(t *testing.T) {
	mlx, _ := ByName("MLX")
	exl2, _ := ByName("ExLlamaV2")
	llamacpp, _ := ByName("llama.cpp")

	nvidia := hw.Machine{GPUs: []hw.GPU{{Name: "RTX 4090", Vendor: hw.NVIDIA, ComputeCapability: 8.9}}}
	apple := hw.Machine{GPUs: []hw.GPU{{Name: "Apple M3 Max", Vendor: hw.Apple}}, UnifiedMem: true}

	if runs, _ := mlx.RunsOn(nvidia); runs {
		t.Error("MLX claimed to run on NVIDIA")
	}
	if runs, _ := exl2.RunsOn(apple); runs {
		t.Error("ExLlamaV2 claimed to run on Apple silicon")
	}
	// No vendor constraint means anywhere, including a machine with no GPU.
	if runs, _ := llamacpp.RunsOn(hw.Machine{}); !runs {
		t.Error("llama.cpp refused a CPU-only machine, which is the case it exists for")
	}
}
