// Package catalog is the built-in list of models worth considering, with the
// architecture fields the math needs.
//
// Embedded rather than fetched so the tool works on an air-gapped box, which is
// exactly the kind of box people run local models on. `--hf <id>` fetches any
// other model's config.json when the network is there.
package catalog

import (
	_ "embed"
	"encoding/json"
	"sort"
	"strings"

	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/arch"
)

//go:embed models.json
var raw []byte

var models []arch.Model

func init() {
	if err := json.Unmarshal(raw, &models); err != nil {
		// The file is embedded at build time, so a parse failure is a build
		// error that escaped, not a runtime condition worth handling.
		panic("catalog: embedded models.json is invalid: " + err.Error())
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Params < models[j].Params })
}

func All() []arch.Model { return append([]arch.Model(nil), models...) }

// Find matches an id, a name, or any unambiguous fragment of either, so
// "qwen3-30" and "Mixtral" both resolve.
func Find(q string) (arch.Model, bool) {
	lq := strings.ToLower(q)
	for _, m := range models {
		if strings.EqualFold(m.ID, q) || strings.EqualFold(m.Name, q) {
			return m, true
		}
	}
	var hits []arch.Model
	for _, m := range models {
		if strings.Contains(strings.ToLower(m.ID), lq) || strings.Contains(strings.ToLower(m.Name), lq) {
			hits = append(hits, m)
		}
	}
	if len(hits) == 1 {
		return hits[0], true
	}
	return arch.Model{}, false
}

// Matches returns every candidate for an ambiguous query, so the CLI can list
// them instead of guessing.
func Matches(q string) []arch.Model {
	lq := strings.ToLower(q)
	var hits []arch.Model
	for _, m := range models {
		if strings.Contains(strings.ToLower(m.ID), lq) || strings.Contains(strings.ToLower(m.Name), lq) {
			hits = append(hits, m)
		}
	}
	return hits
}
