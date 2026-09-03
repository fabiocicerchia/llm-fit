package gguf

// The byte-level decoder: every read goes through here, every read is bounded,
// and nothing above this file touches the buffer directly.

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

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
		return d.scalar(t, n)
	}
	switch t {
	case typeString:
		return d.str()
	case typeArray:
		return d.array()
	}
	return nil, fmt.Errorf("unknown metadata type %d", t)
}

// scalar reads one fixed-width value of n bytes and widens it to the largest
// type of its signedness, so a caller only has to know three shapes.
func (d *reader) scalar(t uint32, n int64) (any, error) {
	if err := d.take(n); err != nil {
		return nil, err
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(d.r, b); err != nil {
		return nil, err
	}
	return widen(t, b)
}

// widen turns the raw little-endian bytes of a scalar into the widest Go type
// of its signedness, so a caller only has to handle uint64, int64 and float64.
func widen(t uint32, b []byte) (any, error) {
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
	return nil, fmt.Errorf("unknown metadata type %d", t)
}

// array skips a metadata array and reports its element count. The tokenizer
// vocabulary is the biggest thing in the header and its contents are of no
// interest, so nothing is retained.
func (d *reader) array() (any, error) {
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
	// The scalar branch above bounds its count; this one did not, so a declared
	// length of 2^64-1 was walked one string at a time until the read hit EOF.
	// Same limit, same message.
	if count > uint64(headerLimit) {
		return nil, errors.New("array length in header is implausible")
	}
	for i := uint64(0); i < count; i++ {
		if _, err := d.str(); err != nil {
			return nil, err
		}
	}
	return count, nil
}
