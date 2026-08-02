// Package hfapi reads a model's shape from Hugging Face, for anything not in
// the built-in catalogue.
//
// config.json carries exactly the fields the maths needs and nothing else has
// to be downloaded — a few kilobytes rather than the tens of gigabytes of
// weights. The parameter count comes from the safetensors index, which is also
// metadata rather than tensors.
package hfapi

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/fabiocicerchia/local-ai-lab/llm-fit/internal/arch"
)

const base = "https://huggingface.co"

type config struct {
	Architectures     []string `json:"architectures"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	HiddenSize        int      `json:"hidden_size"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	HeadDim           int      `json:"head_dim"`
	VocabSize         int      `json:"vocab_size"`
	MaxPositionEmbed  int      `json:"max_position_embeddings"`
	TieWordEmbeddings bool     `json:"tie_word_embeddings"`
	NumLocalExperts   int      `json:"num_local_experts"`
	NumExperts        int      `json:"num_experts"`
	NumExpertsPerTok  int      `json:"num_experts_per_tok"`
	KVLoraRank        int      `json:"kv_lora_rank"`
	QKRopeHeadDim     int      `json:"qk_rope_head_dim"`
	ModelType         string   `json:"model_type"`
	// Multimodal models nest the language model, so the fields above are absent
	// at the top level.
	TextConfig *config `json:"text_config,omitempty"`
}

type safetensorsIndex struct {
	Metadata struct {
		TotalSize int64 `json:"total_size"`
	} `json:"metadata"`
}

// ValidateID checks a Hugging Face repo id before it is pasted into a URL.
//
// The id arrives from the command line and is interpolated into the request
// path, so "../.." or a "%2e%2e" escape would aim the request somewhere other
// than a model repo. Hugging Face ids are "owner/name" over a small charset, so
// the check is an allowlist rather than an attempt to spot bad input.
func ValidateID(id string) error {
	owner, name, ok := strings.Cut(id, "/")
	if !ok || strings.Contains(name, "/") {
		return fmt.Errorf("%q is not a Hugging Face repo id (expected owner/name, e.g. Qwen/Qwen3-8B)", id)
	}
	for _, part := range []string{owner, name} {
		if part == "" || part == "." || part == ".." {
			return fmt.Errorf("%q is not a Hugging Face repo id (empty or relative path segment)", id)
		}
		for _, r := range part {
			switch {
			case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			case r == '-', r == '_', r == '.':
			default:
				return fmt.Errorf("%q is not a Hugging Face repo id (illegal character %q)", id, r)
			}
		}
	}
	return nil
}

// Fetch builds a Model from a Hugging Face repo id such as "Qwen/Qwen3-8B".
func Fetch(id string) (arch.Model, error) {
	if err := ValidateID(id); err != nil {
		return arch.Model{}, err
	}
	client := &http.Client{Timeout: 20 * time.Second}

	var cfg config
	if err := getJSON(client, id, "config.json", &cfg); err != nil {
		return arch.Model{}, err
	}
	// Multimodal repos put the language model one level down; the vision tower
	// is not what generates tokens, so its shape is not what we plan against.
	if cfg.TextConfig != nil && cfg.TextConfig.NumHiddenLayers > 0 {
		inner := *cfg.TextConfig
		inner.TieWordEmbeddings = cfg.TieWordEmbeddings || inner.TieWordEmbeddings
		cfg = inner
	}
	if cfg.NumHiddenLayers == 0 || cfg.HiddenSize == 0 {
		return arch.Model{}, fmt.Errorf("%s: config.json has no usable architecture (is it a GGUF-only or adapter repo?)", id)
	}

	m := arch.Model{
		ID: id, Name: id, Family: cfg.ModelType,
		Layers: cfg.NumHiddenLayers, Hidden: cfg.HiddenSize,
		Heads: cfg.NumAttentionHeads, KVHeads: cfg.NumKeyValueHeads,
		HeadDim: cfg.HeadDim, Vocab: cfg.VocabSize,
		MaxCtx: cfg.MaxPositionEmbed, TiedEmbeddings: cfg.TieWordEmbeddings,
		ExpertsActive: cfg.NumExpertsPerTok,
	}
	if m.KVHeads == 0 {
		m.KVHeads = m.Heads // no GQA stated means one KV head per query head
	}
	if m.MaxCtx == 0 {
		m.MaxCtx = 4096
	}
	m.Experts = cfg.NumLocalExperts
	if m.Experts == 0 {
		m.Experts = cfg.NumExperts
	}
	if cfg.KVLoraRank > 0 {
		m.Attention = arch.MLA
		m.KVLoraRank = cfg.KVLoraRank
		m.QKRopeHeadDim = cfg.QKRopeHeadDim
	}

	// The parameter count is not in config.json. The safetensors index records
	// the total byte size, which divided by the dtype width gives it.
	var idx safetensorsIndex
	if err := getJSON(client, id, "model.safetensors.index.json", &idx); err == nil && idx.Metadata.TotalSize > 0 {
		m.Params = idx.Metadata.TotalSize / 2 // bf16/fp16 checkpoints
	} else {
		m.Params = estimateParams(m, cfg)
	}

	if m.IsMoE() && m.ExpertsActive > 0 && m.Experts > 0 {
		// Only the routed experts scale down; attention and embeddings are read
		// every token regardless.
		m.ActiveParams = activeParams(m)
	}
	return m, nil
}

// estimateParams reconstructs the count from the shape when the index is
// missing, which happens on single-file and older repos.
func estimateParams(m arch.Model, cfg config) int64 {
	h := int64(m.Hidden)
	l := int64(m.Layers)
	kvRatio := float64(m.KVHeads) / float64(max(m.Heads, 1))

	// Attention: Q is hidden², K and V scale with the KV head ratio, O is hidden².
	attn := int64(float64(h*h) * (2 + 2*kvRatio))
	// Feed-forward is typically 8/3·hidden per layer across three projections
	// in a gated architecture.
	ffn := int64(8 * h * h)
	perLayer := attn + ffn
	if m.Experts > 1 {
		perLayer = attn + ffn*int64(m.Experts)
	}
	embed := int64(m.Vocab) * h
	if !m.TiedEmbeddings {
		embed *= 2
	}
	return perLayer*l + embed
}

func activeParams(m arch.Model) int64 {
	h := int64(m.Hidden)
	l := int64(m.Layers)
	kvRatio := float64(m.KVHeads) / float64(max(m.Heads, 1))
	attn := int64(float64(h*h) * (2 + 2*kvRatio))
	ffnOne := int64(8 * h * h)
	embed := int64(m.Vocab) * h
	if !m.TiedEmbeddings {
		embed *= 2
	}
	return (attn+ffnOne*int64(m.ExpertsActive))*l + embed
}

func getJSON(c *http.Client, id, file string, v any) error {
	url := fmt.Sprintf("%s/%s/resolve/main/%s", base, id, file)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	// Gated repos (Llama, Gemma) need a token; without one they 401 and the
	// message should say why rather than "unexpected status".
	if tok := token(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("fetching %s: %w", id, err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("%s is gated; accept its licence on Hugging Face and set HF_TOKEN", id)
	case http.StatusNotFound:
		return fmt.Errorf("%s has no %s", id, file)
	default:
		return fmt.Errorf("%s: %s", id, resp.Status)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(body, v); err != nil {
		return fmt.Errorf("%s/%s: %w", id, file, errors.New("not valid JSON"))
	}
	return nil
}

func token() string {
	for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} {
		if v := strings.TrimSpace(os.Getenv(k)); v != "" {
			return v
		}
	}
	return ""
}
