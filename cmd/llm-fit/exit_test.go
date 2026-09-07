package main

import (
	"os/exec"
	"path/filepath"
	"testing"
)

// The exit code is the only part of a failure a script can read without
// parsing prose, so it is checked the way a script sees it: build the binary,
// run it, look at the status. Asserting on the constants instead would assert
// that 64 is 64.
func TestExitCodesDistinguishTheKindsOfFailure(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the binary and runs it several times")
	}
	bin := filepath.Join(t.TempDir(), "llm-fit")
	if out, err := exec.CommandContext(t.Context(), "go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}

	// Hardware is overridden wherever the verb reaches detection, so the codes
	// are the same on a machine with no GPU as on one with four.
	fixed := []string{"-gpu", "RTX 4090", "-ram", "64", "-ram-bandwidth", "50"}
	cases := []struct {
		what string
		args []string
		want int
	}{
		{"no arguments", nil, exitUsage},
		{"unknown verb", []string{"nonsense"}, exitUsage},
		{"check with no operand", []string{"check"}, exitUsage},
		{"a -kv value outside the table", []string{"suggest", "-kv", "fp16"}, exitUsage},
		{"a card the spec table lacks", []string{"suggest", "-gpu", "No Such Card"}, exitUsage},
		{"a model the catalogue lacks", append([]string{"check", "no-such-model"}, fixed...), exitUsage},
		{"a query matching several models", append([]string{"check", "llama"}, fixed...), exitUsage},
		{"a file that is not GGUF (this package's own source)", append([]string{"check", "main.go"}, fixed...), exitDataErr},
		{"success", []string{"models"}, 0},
	}
	for _, c := range cases {
		run := exec.CommandContext(t.Context(), bin, c.args...)
		_ = run.Run() // a non-zero exit is the subject here, not a test failure
		if got := run.ProcessState.ExitCode(); got != c.want {
			t.Errorf("%s: exit %d, want %d", c.what, got, c.want)
		}
	}
}
