package hfapi

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/fabiocicerchia/llm-fit/internal/arch"
)

// toServer sends a request built for huggingface.co to the test server instead,
// leaving the path — the part the id allowlist guards — exactly as fetch built it.
type toServer struct{ target *url.URL }

func (t toServer) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.URL.Scheme, req.URL.Host = t.target.Scheme, t.target.Host
	return http.DefaultTransport.RoundTrip(req)
}

// serving returns a client whose requests are answered from files: a path
// suffix such as "config.json" maps to the body, and a missing one 404s the way
// Hugging Face does for a repo without a safetensors index.
func serving(t *testing.T, files map[string]string) *http.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for name, body := range files {
			if strings.HasSuffix(r.URL.Path, "/"+name) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(body))
				return
			}
		}
		http.Error(w, "not found", http.StatusNotFound)
	}))
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("test server URL %q: %v", srv.URL, err)
	}
	return &http.Client{Transport: toServer{target: u}}
}

const denseConfig = `{
  "model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096,
  "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 151936,
  "max_position_embeddings": 40960
}`

// A dense repo with a safetensors index: every field lands where the maths
// expects it and the parameter count comes off the index, not the estimator.
func TestFetchReadsADenseRepo(t *testing.T) {
	c := serving(t, map[string]string{
		"config.json":                  denseConfig,
		"model.safetensors.index.json": `{"metadata":{"total_size":16000000000}}`,
	})
	m, err := fetch(t.Context(), c, "Qwen/Qwen3-8B")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	want := arch.Model{
		ID: "Qwen/Qwen3-8B", Name: "Qwen/Qwen3-8B", Family: "qwen3",
		Params: 8000000000, Layers: 36, Hidden: 4096, Heads: 32, KVHeads: 8,
		Vocab: 151936, MaxCtx: 40960,
	}
	if m != want {
		t.Errorf("fetch returned\n %+v\nwant\n %+v", m, want)
	}
}

// No index file is the common case for a GGUF-first or older repo; the shape
// then has to carry the parameter count on its own.
func TestFetchEstimatesParamsWithoutAnIndex(t *testing.T) {
	c := serving(t, map[string]string{"config.json": denseConfig})
	m, err := fetch(t.Context(), c, "Qwen/Qwen3-8B")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if m.Params != estimateParams(m) {
		t.Errorf("Params = %d, want the estimate %d", m.Params, estimateParams(m))
	}
	if m.Params <= 0 {
		t.Errorf("Params = %d, want a positive estimate", m.Params)
	}
}

// An MoE routes a token to a few of its experts, so ActiveParams must come back
// well below Params — that ratio is what the speed estimate divides by.
func TestFetchSetsActiveParamsForAnMoE(t *testing.T) {
	c := serving(t, map[string]string{"config.json": `{
	  "model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048,
	  "num_attention_heads": 32, "num_key_value_heads": 4, "vocab_size": 151936,
	  "max_position_embeddings": 40960, "num_experts": 128, "num_experts_per_tok": 8
	}`})
	m, err := fetch(t.Context(), c, "Qwen/Qwen3-30B-A3B")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if !m.IsMoE() {
		t.Fatalf("Experts = %d, want the MoE branch taken", m.Experts)
	}
	if m.ActiveParams <= 0 || m.ActiveParams >= m.Params {
		t.Errorf("ActiveParams = %d, want 0 < it < Params (%d)", m.ActiveParams, m.Params)
	}
}

// A multimodal repo nests the language model under text_config. The vision
// tower is not what generates tokens, so the inner shape is what comes back —
// and a tie flag set only at the top level still has to reach it.
func TestFetchUnwrapsAMultimodalConfig(t *testing.T) {
	c := serving(t, map[string]string{"config.json": `{
	  "model_type": "gemma3", "tie_word_embeddings": true,
	  "text_config": {"model_type": "gemma3_text", "num_hidden_layers": 34,
	    "hidden_size": 2560, "num_attention_heads": 8, "num_key_value_heads": 4,
	    "vocab_size": 262144, "max_position_embeddings": 131072}
	}`})
	m, err := fetch(t.Context(), c, "google/gemma-3-4b-it")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if m.Layers != 34 || m.Hidden != 2560 {
		t.Errorf("got %d layers of %d, want the inner 34 of 2560", m.Layers, m.Hidden)
	}
	if !m.TiedEmbeddings {
		t.Error("TiedEmbeddings = false, want the outer flag carried into text_config")
	}
}

// An older repo may omit max_position_embeddings entirely; a zero there would
// be divided by downstream, so the loader has to substitute a floor.
func TestFetchFillsInAMissingContextLength(t *testing.T) {
	c := serving(t, map[string]string{"config.json": `{
	  "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
	  "num_attention_heads": 32, "vocab_size": 32000
	}`})
	m, err := fetch(t.Context(), c, "meta-llama/Llama-2-7b-hf")
	if err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if m.MaxCtx != defaultMaxCtx {
		t.Errorf("MaxCtx = %d, want the %d floor", m.MaxCtx, defaultMaxCtx)
	}
	if m.KVHeads != m.Heads {
		t.Errorf("KVHeads = %d, want %d — no GQA stated means one per query head", m.KVHeads, m.Heads)
	}
}

// The failures a user actually hits, each of which must be an error rather than
// a Model with zeroes in it that the arithmetic would go on to divide by.
func TestFetchRefusesWhatItCannotPlanAgainst(t *testing.T) {
	cases := []struct {
		name  string
		id    string
		files map[string]string
		want  string
	}{
		{"bad id", "../../etc/passwd", map[string]string{"config.json": denseConfig}, "not a Hugging Face repo id"},
		{"no config", "Qwen/Qwen3-8B", map[string]string{}, "has no config.json"},
		{"adapter repo", "some/lora", map[string]string{"config.json": `{"model_type":"lora"}`}, "no usable architecture"},
		{"not json", "Qwen/Qwen3-8B", map[string]string{"config.json": `<html>login</html>`}, "not valid JSON"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := fetch(t.Context(), serving(t, tc.files), tc.id)
			if err == nil {
				t.Fatalf("fetch(t.Context(), %q) returned no error, want one mentioning %q", tc.id, tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("fetch(t.Context(), %q) said %q, want it to mention %q", tc.id, err, tc.want)
			}
		})
	}
}

// A gated repo answers 401 until the licence is accepted. The message is the
// whole value of the branch: "unexpected status" would send a user nowhere.
func TestFetchExplainsAGatedRepo(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("test server URL %q: %v", srv.URL, err)
	}
	_, err = fetch(t.Context(), &http.Client{Transport: toServer{target: u}}, "meta-llama/Llama-3.1-8B")
	if err == nil || !strings.Contains(err.Error(), "gated") {
		t.Errorf("fetch on a 401 said %v, want it to mention the repo is gated", err)
	}
}
