package main

// Value formatting: how one number becomes one column. Nothing here knows what
// the number means.

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/fabiocicerchia/llm-fit/internal/engine"
	"github.com/fabiocicerchia/llm-fit/internal/fit"
)

func verdictMark(v fit.Verdict) string {
	switch v {
	case fit.Excellent:
		return "excellent"
	case fit.Good:
		return "good"
	case fit.Usable:
		return "usable"
	case fit.Sluggish:
		return "sluggish"
	case fit.Unusable:
	}
	return "unusable"
}

func servingNote(s bool) string {
	if s {
		return ", serving"
	}
	return ""
}

func families(e engine.Engine) string {
	parts := make([]string, 0, len(e.Formats))
	for _, f := range e.Formats {
		parts = append(parts, string(f))
	}
	return strings.Join(parts, ", ")
}

func gib(b int64) string {
	if b <= 0 {
		return "—"
	}
	g := float64(b) / (1 << 30)
	if g < 1 {
		return fmt.Sprintf("%.0f MiB", float64(b)/(1<<20))
	}
	return fmt.Sprintf("%.1f GiB", g)
}

func params(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.0fM", float64(n)/1e6)
	}
	return fmt.Sprintf("%d", n)
}

func thousands(n int) string {
	switch {
	case n >= 1000000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%dk", n/1000)
	}
	return fmt.Sprintf("%d", n)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func emitJSON(v any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	// Encoding to stdout: a write error here has already lost the output, and
	// the only place left to report it is the same broken stream.
	_ = enc.Encode(v) //nolint:errcheck // see above
}
