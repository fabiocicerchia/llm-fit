package gguf

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
