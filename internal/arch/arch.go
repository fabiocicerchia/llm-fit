// Package arch describes a model's shape, which is all the memory and speed
// math actually depends on.
//
// Parameter count alone is not enough, and the places it misleads are the
// places this tool has to get right:
//
//   - Grouped-query attention. Llama-3-70B has 64 query heads and 8 KV heads,
//     so its KV cache is an eighth of what multi-head math predicts. Sizing a
//     70B's context off head count is wrong by 8x.
//   - Mixture of experts. Mixtral-8x7B holds 46.7B parameters and reads 12.9B
//     per token, so it needs the memory of a 47B and decodes at the speed of a
//     13B. Both numbers are needed, and they are not derivable from each other.
//   - Vocabulary. Gemma's 256k-token embedding is a fifth of a 2B model's
//     weights and is quantized differently from the body, so folding it into an
//     average bits-per-weight misestimates every small model.
package arch

// Attention is how the model stores keys and values, which decides the KV cache
// formula rather than merely scaling it.
type Attention string

const (
	// MHA/GQA/MQA are the same formula; the head count is what differs. MQA is
	// GQA with one KV head, MHA is GQA with as many KV heads as query heads.
	GQA Attention = "gqa"
	// MLA compresses KV into a shared latent vector (DeepSeek-V2 onward). The
	// cache is a different shape entirely and roughly an order of magnitude
	// smaller, which is the whole reason those models serve long contexts.
	MLA Attention = "mla"
)

type Model struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Family string `json:"family,omitempty"`

	// Params is every weight in the file. ActiveParams is what a single token
	// reads: equal to Params for a dense model, much smaller for an MoE.
	Params       int64 `json:"params"`
	ActiveParams int64 `json:"active_params,omitempty"`

	Layers  int `json:"layers"`
	Hidden  int `json:"hidden"`
	Heads   int `json:"heads"`
	KVHeads int `json:"kv_heads"`
	HeadDim int `json:"head_dim,omitempty"`
	Vocab   int `json:"vocab"`
	MaxCtx  int `json:"ctx"`

	// TiedEmbeddings means the output projection reuses the input embedding
	// matrix. Common in small models (Gemma, Qwen 0.5B), and it halves the
	// parameters that sit at a different quantization from the body.
	TiedEmbeddings bool `json:"tied_embeddings,omitempty"`

	Experts       int `json:"experts,omitempty"`
	ExpertsActive int `json:"experts_active,omitempty"`

	Attention     Attention `json:"attention,omitempty"`
	KVLoraRank    int       `json:"kv_lora_rank,omitempty"`
	QKRopeHeadDim int       `json:"qk_rope_head_dim,omitempty"`

	// Vision, when set, notes a multimodal tower whose weights ship in the same
	// file. Recorded so the number is not silently missing from Params.
	Notes string `json:"notes,omitempty"`
}

// HeadDimension falls back to the usual hidden/heads when a config does not
// state it. Llama-3 and Command-R state it explicitly and it does not always
// equal the quotient, so the explicit value wins where present.
func (m Model) HeadDimension() int {
	if m.HeadDim > 0 {
		return m.HeadDim
	}
	if m.Heads == 0 {
		return 0
	}
	return m.Hidden / m.Heads
}

// Active is the parameter count a single token actually reads. For an MoE only
// the router plus the selected experts are touched, which is why an MoE decodes
// far faster than its file size suggests.
func (m Model) Active() int64 {
	if m.ActiveParams > 0 {
		return m.ActiveParams
	}
	return m.Params
}

func (m Model) IsMoE() bool { return m.Experts > 1 }

// EmbeddingParams counts the token embedding matrix, and the output projection
// when it is a separate tensor.
//
// Kept apart from the body because quantizers treat it differently: llama.cpp
// leaves output.weight at Q6_K inside a Q4_K_M model. On a 70B that is noise;
// on Gemma-2-2B the embedding is 23% of the model and ignoring it puts the
// estimate out by more than the quantization choice does.
func (m Model) EmbeddingParams() int64 {
	one := int64(m.Vocab) * int64(m.Hidden)
	if m.TiedEmbeddings {
		return one
	}
	return one * 2
}

// BodyParams is everything that is not an embedding — the layers, which is what
// the quantization format's bits-per-weight applies to.
func (m Model) BodyParams() int64 {
	body := m.Params - m.EmbeddingParams()
	if body < 0 {
		// A stated parameter count that cannot cover its own embeddings means
		// the metadata is wrong. Degrade to the whole model rather than return
		// a negative that would silently underestimate memory.
		return m.Params
	}
	return body
}

// KVBytesPerToken is the cache cost of one token of context, across all layers,
// before any KV quantization is applied.
//
// The factor of two is K and V. It does not apply to MLA, which stores one
// compressed latent vector instead of a separate key and value.
func (m Model) KVBytesPerToken(bytesPerElem float64) float64 {
	if m.Attention == MLA {
		// DeepSeek-V3: 512 latent + 64 rope dims per layer per token — about
		// 1/14th of what the equivalent GQA cache would cost.
		perLayer := m.KVLoraRank + m.QKRopeHeadDim
		return float64(m.Layers) * float64(perLayer) * bytesPerElem
	}
	kvHeads := m.KVHeads
	if kvHeads == 0 {
		kvHeads = m.Heads // pre-GQA models store one KV head per query head
	}
	return 2 * float64(m.Layers) * float64(kvHeads) * float64(m.HeadDimension()) * bytesPerElem
}
