package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The fixtures below are written byte for byte to the GGUF spec rather than
// checked in as a binary: a real file is gigabytes, and a trimmed one would be
// a fixture nobody could re-derive. The key names and layout are those a
// llama.cpp conversion emits — verified against the published
// Qwen2.5-7B-Instruct-Q4_K_M header shape.
type builder struct{ b bytes.Buffer }

func (w *builder) u32(v uint32) { _ = binary.Write(&w.b, binary.LittleEndian, v) }
func (w *builder) u64(v uint64) { _ = binary.Write(&w.b, binary.LittleEndian, v) }

func (w *builder) str(s string) {
	w.u64(uint64(len(s)))
	w.b.WriteString(s)
}

func (w *builder) kvU32(key string, v uint32) {
	w.str(key)
	w.u32(typeUint32)
	w.u32(v)
}

func (w *builder) kvStr(key, v string) {
	w.str(key)
	w.u32(typeString)
	w.str(v)
}

// kvStrArray is the tokenizer vocabulary: the largest thing in a real header
// and the one the parser must skip rather than hold.
func (w *builder) kvStrArray(key string, n int) {
	w.str(key)
	w.u32(typeArray)
	w.u32(typeString)
	w.u64(uint64(n))
	for i := 0; i < n; i++ {
		w.str("tok")
	}
}

func (w *builder) kvF32Array(key string, n int) {
	w.str(key)
	w.u32(typeArray)
	w.u32(typeFloat32)
	w.u64(uint64(n))
	for i := 0; i < n; i++ {
		_ = binary.Write(&w.b, binary.LittleEndian, math.Float32bits(1.5))
	}
}

func (w *builder) tensor(name string, dims ...uint64) {
	w.str(name)
	w.u32(uint32(len(dims)))
	for _, d := range dims {
		w.u64(d)
	}
	w.u32(12) // ggml type Q4_K
	w.u64(0)  // data offset
}

type tensorSpec struct {
	name string
	dims []uint64
}

// dense builds a 4-layer dense model: 2 tensors per layer plus embedding and
// output, so the parameter count is checkable by hand.
func denseFile(t *testing.T, ftype uint32, withOutput bool) []byte {
	t.Helper()
	tensors := []tensorSpec{{"token_embd.weight", []uint64{64, 1000}}}
	for i := 0; i < 4; i++ {
		tensors = append(tensors,
			tensorSpec{"blk." + string(rune('0'+i)) + ".attn_q.weight", []uint64{64, 64}},
			tensorSpec{"blk." + string(rune('0'+i)) + ".ffn_up.weight", []uint64{64, 128}},
		)
	}
	if withOutput {
		tensors = append(tensors, tensorSpec{"output.weight", []uint64{64, 1000}})
	}

	var w builder
	w.u32(magic)
	w.u32(3)
	w.u64(uint64(len(tensors)))
	w.u64(9) // metadata count
	w.kvStr("general.architecture", "llama")
	w.kvStr("general.name", "Test 7B")
	w.kvU32("general.file_type", ftype)
	w.kvU32("llama.block_count", 4)
	w.kvU32("llama.embedding_length", 64)
	w.kvU32("llama.attention.head_count", 8)
	w.kvU32("llama.attention.head_count_kv", 2)
	w.kvU32("llama.context_length", 32768)
	w.kvStrArray("tokenizer.ggml.tokens", 1000)
	for _, ts := range tensors {
		w.tensor(ts.name, ts.dims...)
	}
	return w.b.Bytes()
}

func TestReadDense(t *testing.T) {
	info, err := read(bytes.NewReader(denseFile(t, 15, true)))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if info.Arch != "llama" || info.Name != "Test 7B" {
		t.Errorf("arch/name = %q/%q", info.Arch, info.Name)
	}
	if info.Layers != 4 || info.Hidden != 64 || info.Heads != 8 || info.KVHeads != 2 {
		t.Errorf("shape = %d/%d/%d/%d", info.Layers, info.Hidden, info.Heads, info.KVHeads)
	}
	if info.MaxCtx != 32768 {
		t.Errorf("ctx = %d", info.MaxCtx)
	}
	if info.Vocab != 1000 {
		t.Errorf("vocab = %d, want 1000 (from the embedding shape)", info.Vocab)
	}
	// 64*1000 embedding + 4*(64*64 + 64*128) + 64*1000 output
	want := int64(64*1000 + 4*(64*64+64*128) + 64*1000)
	if info.Params != want {
		t.Errorf("params = %d, want %d", info.Params, want)
	}
	if info.Tied {
		t.Error("output.weight present, so embeddings are not tied")
	}
	f, ok := info.Format()
	if !ok || f.Name != "Q4_K_M" {
		t.Errorf("format = %v/%v, want Q4_K_M", f.Name, ok)
	}
}

func TestTiedEmbeddingsWhenOutputAbsent(t *testing.T) {
	info, err := read(bytes.NewReader(denseFile(t, 15, false)))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !info.Tied {
		t.Error("no output.weight, so the embedding is reused and Tied should be set")
	}
}

func TestMoEActiveParams(t *testing.T) {
	tensors := []tensorSpec{
		{"token_embd.weight", []uint64{64, 1000}},
		{"blk.0.attn_q.weight", []uint64{64, 64}},
		{"blk.0.ffn_gate_exps.weight", []uint64{64, 128, 8}}, // 8 experts, stacked
		{"output.weight", []uint64{64, 1000}},
	}
	var w builder
	w.u32(magic)
	w.u32(3)
	w.u64(uint64(len(tensors)))
	w.u64(8)
	w.kvStr("general.architecture", "qwen3moe")
	w.kvU32("general.file_type", 15)
	w.kvU32("qwen3moe.block_count", 1)
	w.kvU32("qwen3moe.embedding_length", 64)
	w.kvU32("qwen3moe.attention.head_count", 8)
	w.kvU32("qwen3moe.attention.head_count_kv", 2)
	w.kvU32("qwen3moe.expert_count", 8)
	w.kvU32("qwen3moe.expert_used_count", 2)
	for _, ts := range tensors {
		w.tensor(ts.name, ts.dims...)
	}

	info, err := read(bytes.NewReader(w.b.Bytes()))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	experts := int64(64 * 128 * 8)
	total := int64(64*1000) + 64*64 + experts + 64*1000
	if info.Params != total {
		t.Errorf("params = %d, want %d", info.Params, total)
	}
	// Two of eight experts are read per token.
	want := total - experts + experts*2/8
	if info.ActiveParams != want {
		t.Errorf("active = %d, want %d", info.ActiveParams, want)
	}
	if m := info.Model(); !m.IsMoE() || m.Active() != want {
		t.Errorf("model active = %d, moe = %v", m.Active(), m.IsMoE())
	}
}

func TestSkipsLargeArraysWithoutReadingThemIn(t *testing.T) {
	// A float array between the keys we want: the parser must step over it and
	// still find what follows.
	var w builder
	w.u32(magic)
	w.u32(3)
	w.u64(1)
	w.u64(5)
	w.kvStr("general.architecture", "llama")
	w.kvF32Array("llama.rope.freqs", 4096)
	w.kvU32("llama.block_count", 2)
	w.kvU32("llama.embedding_length", 32)
	w.kvStrArray("tokenizer.ggml.tokens", 500)
	w.tensor("token_embd.weight", 32, 500)

	info, err := read(bytes.NewReader(w.b.Bytes()))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if info.Layers != 2 || info.Hidden != 32 || info.Vocab != 500 {
		t.Errorf("got %d/%d/%d after the arrays", info.Layers, info.Hidden, info.Vocab)
	}
}

func TestRejectsBadInput(t *testing.T) {
	cases := []struct {
		name string
		in   []byte
		want string
	}{
		{"not gguf", []byte("this is a safetensors file"), "bad magic"},
		{"truncated", denseFile(t, 15, true)[:40], "unexpected EOF"},
		{"empty", nil, "bad magic"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := read(bytes.NewReader(tc.in))
			if err == nil {
				t.Fatal("want an error, got a reading")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}

func TestRejectsUnsupportedVersion(t *testing.T) {
	var w builder
	w.u32(magic)
	w.u32(1)
	w.u64(0)
	w.u64(0)
	if _, err := read(bytes.NewReader(w.b.Bytes())); err == nil || !strings.Contains(err.Error(), "version 1") {
		t.Errorf("err = %v, want it to name the version", err)
	}
}

func TestUnknownFileTypeIsNotGuessed(t *testing.T) {
	info, err := read(bytes.NewReader(denseFile(t, 250, true)))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if _, ok := info.Format(); ok {
		t.Error("file_type 250 is unknown; reporting a quant for it would be a guess")
	}
}

func TestReadFromDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, denseFile(t, 18, true), 0o600); err != nil {
		t.Fatal(err)
	}
	info, err := Read(path)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if f, ok := info.Format(); !ok || f.Name != "Q6_K" {
		t.Errorf("format = %q", f.Name)
	}
	m := info.Model()
	if m.Name != "Test 7B" || m.Layers != 4 || m.KVHeads != 2 {
		t.Errorf("model = %+v", m)
	}
	if _, err := Read(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Error("want an error for a missing file")
	}
}
