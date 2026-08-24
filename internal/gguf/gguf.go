// Package gguf reads a model's shape out of a GGUF file's own header.
//
// Everywhere else the shape is inferred: the catalogue keys off a model name,
// hfapi reads the upstream repo's config.json. Both describe a model, not the
// file on disk — and the file is what someone is about to load. A repo's
// config.json says nothing about which quant was baked into the copy in
// ~/.cache, and a filename says only what whoever uploaded it chose to type.
//
// The header carries both: the architecture keys give layers, embedding width
// and head counts, and the tensor directory that follows gives every tensor's
// shape, which sums to an exact parameter count rather than an estimate.
//
// Only the header is read. Tensor *data* is never touched, so this costs a few
// hundred kilobytes on a file of any size, and headerLimit caps what a
// malformed or hostile file can make us allocate.
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/fabiocicerchia/llm-fit/internal/arch"
	"github.com/fabiocicerchia/llm-fit/internal/quant"
)

// magic is "GGUF" little-endian.
const magic = 0x46554747

// headerLimit bounds the metadata read. Real headers run to a few hundred KB —
// the tokenizer vocabulary is the bulk of it — so 64 MB is far above anything
// legitimate and still refuses a file whose declared counts would have us read
// for ever.
const headerLimit = 64 << 20

// maxTensors is the same guard for the tensor directory: a 405B model has a few
// thousand tensors, and each entry costs a name plus five numbers.
const maxTensors = 1 << 20

// GGUF metadata value types, in the order the spec assigns them.
const (
	typeUint8 uint32 = iota
	typeInt8
	typeUint16
	typeInt16
	typeUint32
	typeInt32
	typeFloat32
	typeBool
	typeString
	typeArray
	typeUint64
	typeInt64
	typeFloat64
)

// fileTypes maps general.file_type — llama.cpp's ggml_ftype enum — onto the
// names in the quant table. Only the values a published GGUF actually carries
// are listed; anything else is reported as unknown rather than guessed at,
// because a wrong bits-per-weight is worse than none.
var fileTypes = map[uint32]string{
	0:  "FP16", // all F32, but the table has no F32 row and 16 is the closer lie
	1:  "FP16",
	2:  "Q4_0",
	3:  "Q4_0", // Q4_1
	7:  "Q8_0",
	8:  "Q5_K_S", // Q5_0
	9:  "Q5_K_S", // Q5_1
	10: "Q2_K",
	11: "Q3_K_S",
	12: "Q3_K_M",
	13: "Q3_K_L",
	14: "Q4_K_S",
	15: "Q4_K_M",
	16: "Q5_K_S",
	17: "Q5_K_M",
	18: "Q6_K",
	19: "IQ2_XXS",
	22: "IQ3_XXS",
	23: "IQ3_XXS",
	24: "IQ1_S",
	26: "IQ3_M",
	27: "IQ3_M",
	29: "IQ2_M",
	30: "IQ4_XS",
	31: "IQ1_M",
	32: "BF16",
}

// Info is what the header says. Params and Vocab come from the tensor
// directory rather than the metadata, because the metadata does not carry them
// and a header that did would still be the uploader's word for it.
type Info struct {
	Arch     string
	Name     string
	FileType uint32

	Layers  int
	Hidden  int
	Heads   int
	KVHeads int
	HeadDim int
	MaxCtx  int

	Experts       int
	ExpertsActive int

	Params       int64
	ActiveParams int64
	Vocab        int
	Tied         bool

	TensorCount uint64
}

type reader struct {
	r    *bufio.Reader
	read int64
}

func (d *reader) take(n int64) error {
	d.read += n
	if d.read > headerLimit {
		return fmt.Errorf("header exceeds %d bytes: not a GGUF file, or a corrupt one", int64(headerLimit))
	}
	return nil
}

func (d *reader) u32() (uint32, error) {
	if err := d.take(4); err != nil {
		return 0, err
	}
	var b [4]byte
	if _, err := io.ReadFull(d.r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (d *reader) u64() (uint64, error) {
	if err := d.take(8); err != nil {
		return 0, err
	}
	var b [8]byte
	if _, err := io.ReadFull(d.r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (d *reader) str() (string, error) {
	n, err := d.u64()
	if err != nil {
		return "", err
	}
	if n > headerLimit {
		return "", errors.New("string length in header is implausible")
	}
	if err := d.take(int64(n)); err != nil {
		return "", err
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(d.r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

// skip discards n bytes through the buffered reader. Used for the tokenizer
// arrays, which are the biggest thing in the header and of no interest here:
// the vocabulary size comes from the embedding tensor's own shape.
func (d *reader) skip(n int64) error {
	if err := d.take(n); err != nil {
		return err
	}
	_, err := io.CopyN(io.Discard, d.r, n)
	return err
}

// scalarSize is the width of a fixed-size metadata value, or 0 for the
// variable-length ones.
func scalarSize(t uint32) int64 {
	switch t {
	case typeUint8, typeInt8, typeBool:
		return 1
	case typeUint16, typeInt16:
		return 2
	case typeUint32, typeInt32, typeFloat32:
		return 4
	case typeUint64, typeInt64, typeFloat64:
		return 8
	}
	return 0
}

// value reads one metadata value, returning it only when it is a scalar this
// package might want. Strings come back as themselves; arrays are skipped and
// reported as their element count, which is all any caller here needs.
func (d *reader) value(t uint32) (any, error) {
	if n := scalarSize(t); n > 0 {
		if err := d.take(n); err != nil {
			return nil, err
		}
		b := make([]byte, n)
		if _, err := io.ReadFull(d.r, b); err != nil {
			return nil, err
		}
		switch t {
		case typeUint8, typeBool:
			return uint64(b[0]), nil
		case typeInt8:
			return int64(int8(b[0])), nil
		case typeUint16:
			return uint64(binary.LittleEndian.Uint16(b)), nil
		case typeInt16:
			return int64(int16(binary.LittleEndian.Uint16(b))), nil
		case typeUint32:
			return uint64(binary.LittleEndian.Uint32(b)), nil
		case typeInt32:
			return int64(int32(binary.LittleEndian.Uint32(b))), nil
		case typeFloat32:
			return float64(math.Float32frombits(binary.LittleEndian.Uint32(b))), nil
		case typeUint64:
			return binary.LittleEndian.Uint64(b), nil
		case typeInt64:
			return int64(binary.LittleEndian.Uint64(b)), nil
		case typeFloat64:
			return math.Float64frombits(binary.LittleEndian.Uint64(b)), nil
		}
	}
	switch t {
	case typeString:
		return d.str()
	case typeArray:
		elem, err := d.u32()
		if err != nil {
			return nil, err
		}
		count, err := d.u64()
		if err != nil {
			return nil, err
		}
		if n := scalarSize(elem); n > 0 {
			if count > uint64(headerLimit) {
				return nil, errors.New("array length in header is implausible")
			}
			if err := d.skip(int64(count) * n); err != nil {
				return nil, err
			}
			return count, nil
		}
		if elem != typeString {
			return nil, fmt.Errorf("unsupported array element type %d", elem)
		}
		for i := uint64(0); i < count; i++ {
			if _, err := d.str(); err != nil {
				return nil, err
			}
		}
		return count, nil
	}
	return nil, fmt.Errorf("unknown metadata type %d", t)
}

func asInt(v any) int {
	switch n := v.(type) {
	case uint64:
		return int(n)
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}

// Read parses the header of the GGUF file at path.
func Read(path string) (Info, error) {
	f, err := os.Open(path)
	if err != nil {
		return Info{}, err
	}
	defer func() { _ = f.Close() }() // read-only: a failed close has nothing to report
	return read(f)
}

func read(f io.Reader) (Info, error) {
	d := &reader{r: bufio.NewReaderSize(f, 1<<16)}

	m, err := d.u32()
	if err != nil || m != magic {
		return Info{}, errors.New("not a GGUF file: bad magic")
	}
	version, err := d.u32()
	if err != nil {
		return Info{}, err
	}
	// v1 put counts in 32 bits and is long gone from published models; refusing
	// it is honest, since parsing it with the v2 layout silently misreads every
	// subsequent offset.
	if version < 2 || version > 3 {
		return Info{}, fmt.Errorf("unsupported GGUF version %d", version)
	}
	tensorCount, err := d.u64()
	if err != nil {
		return Info{}, err
	}
	if tensorCount > maxTensors {
		return Info{}, fmt.Errorf("implausible tensor count %d", tensorCount)
	}
	kvCount, err := d.u64()
	if err != nil {
		return Info{}, err
	}
	if kvCount > headerLimit {
		return Info{}, fmt.Errorf("implausible metadata count %d", kvCount)
	}

	kv := make(map[string]any, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, err := d.str()
		if err != nil {
			return Info{}, fmt.Errorf("metadata key %d: %w", i, err)
		}
		t, err := d.u32()
		if err != nil {
			return Info{}, fmt.Errorf("metadata %q: %w", key, err)
		}
		v, err := d.value(t)
		if err != nil {
			return Info{}, fmt.Errorf("metadata %q: %w", key, err)
		}
		kv[key] = v
	}

	info := Info{TensorCount: tensorCount}
	if s, ok := kv["general.architecture"].(string); ok {
		info.Arch = s
	}
	if info.Arch == "" {
		return Info{}, errors.New("GGUF header has no general.architecture")
	}
	if s, ok := kv["general.name"].(string); ok {
		info.Name = s
	}
	info.FileType = uint32(asInt(kv["general.file_type"]))

	// Architecture keys are namespaced by the architecture itself, which is why
	// they cannot be read until general.architecture has been.
	p := info.Arch + "."
	info.Layers = asInt(kv[p+"block_count"])
	info.Hidden = asInt(kv[p+"embedding_length"])
	info.Heads = asInt(kv[p+"attention.head_count"])
	info.KVHeads = asInt(kv[p+"attention.head_count_kv"])
	info.MaxCtx = asInt(kv[p+"context_length"])
	info.HeadDim = asInt(kv[p+"attention.key_length"])
	info.Experts = asInt(kv[p+"expert_count"])
	info.ExpertsActive = asInt(kv[p+"expert_used_count"])
	if info.KVHeads == 0 {
		info.KVHeads = info.Heads // pre-GQA models omit the key entirely
	}

	if info.Layers == 0 || info.Hidden == 0 {
		return Info{}, fmt.Errorf("GGUF header for %q has no block_count/embedding_length", info.Arch)
	}

	// Tensor directory. Every entry is name, rank, dims, type, offset — the
	// shapes are the exact parameter count, which no metadata key carries.
	var params, expertParams int64
	var hasOutput bool
	for i := uint64(0); i < tensorCount; i++ {
		name, err := d.str()
		if err != nil {
			return Info{}, fmt.Errorf("tensor %d: %w", i, err)
		}
		rank, err := d.u32()
		if err != nil {
			return Info{}, fmt.Errorf("tensor %q: %w", name, err)
		}
		if rank > 4 {
			return Info{}, fmt.Errorf("tensor %q has rank %d", name, rank)
		}
		n := int64(1)
		dims := make([]int64, 0, rank)
		for j := uint32(0); j < rank; j++ {
			dv, err := d.u64()
			if err != nil {
				return Info{}, fmt.Errorf("tensor %q: %w", name, err)
			}
			dims = append(dims, int64(dv))
			n *= int64(dv)
		}
		if _, err := d.u32(); err != nil { // ggml type
			return Info{}, fmt.Errorf("tensor %q: %w", name, err)
		}
		if _, err := d.u64(); err != nil { // data offset
			return Info{}, fmt.Errorf("tensor %q: %w", name, err)
		}

		params += n
		// llama.cpp stacks an MoE layer's experts into one "_exps" tensor, so
		// this is the whole expert bank in a layer rather than one expert.
		if strings.Contains(name, "_exps") {
			expertParams += n
		}
		switch {
		case name == "token_embd.weight" && len(dims) == 2:
			// Stored [embedding_length, vocab]: the row count is the vocabulary.
			info.Vocab = int(dims[1])
		case name == "output.weight":
			hasOutput = true
		}
	}
	info.Params = params
	// No separate output.weight means the output projection reuses the input
	// embedding, which is what TiedEmbeddings records.
	info.Tied = !hasOutput
	if info.Vocab == 0 {
		// Fall back to the tokenizer array's length, recorded while skipping it.
		info.Vocab = asInt(kv["tokenizer.ggml.tokens"])
	}

	// An MoE reads the router plus the experts it selects, so its active
	// parameters are the dense remainder plus that fraction of the expert bank.
	if info.Experts > 1 && info.ExpertsActive > 0 && expertParams > 0 {
		info.ActiveParams = params - expertParams + expertParams*int64(info.ExpertsActive)/int64(info.Experts)
	}

	return info, nil
}

// Model converts the header into the shape the rest of the tool works in.
func (i Info) Model() arch.Model {
	name := i.Name
	if name == "" {
		name = i.Arch
	}
	m := arch.Model{
		ID:             name,
		Name:           name,
		Family:         i.Arch,
		Params:         i.Params,
		ActiveParams:   i.ActiveParams,
		Layers:         i.Layers,
		Hidden:         i.Hidden,
		Heads:          i.Heads,
		KVHeads:        i.KVHeads,
		HeadDim:        i.HeadDim,
		Vocab:          i.Vocab,
		MaxCtx:         i.MaxCtx,
		TiedEmbeddings: i.Tied,
		Experts:        i.Experts,
		ExpertsActive:  i.ExpertsActive,
		Attention:      arch.GQA,
		Notes:          "read from the GGUF header",
	}
	return m
}

// Format is the quantization the file is actually in. Reported as not-ok
// rather than defaulted when general.file_type is a value this table does not
// know: the fit maths would otherwise run on a bits-per-weight nobody checked.
func (i Info) Format() (quant.Format, bool) {
	name, ok := fileTypes[i.FileType]
	if !ok {
		return quant.Format{}, false
	}
	return quant.ByName(name)
}
