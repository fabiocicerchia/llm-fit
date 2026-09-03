package main

// Every way this program can fail, and the code it leaves behind. One table so
// a caller can tell "you typed something I don't have" from "Hugging Face is
// down" without parsing the message — previously every failure was 1 or 2 and
// the two were used interchangeably.
//
// The numbers are sysexits.h (FreeBSD /usr/include/sysexits.h), which is what
// BSD and most Unix CLIs use above the 0/1/2 range.
const (
	// exitUsage is a usage error this program detects itself: an unknown verb,
	// a missing operand, a flag value outside the set it accepts, a model name
	// that matches nothing or matches several things.
	//
	// Errors the flag package finds are its own to report; flag.ExitOnError
	// exits 2 before returning, and that stays 2.
	exitUsage = 64 // EX_USAGE

	// exitDataErr is input this program could read but not parse — a file that
	// does not hold the GGUF structure it claims.
	exitDataErr = 65 // EX_DATAERR

	// exitUnavailable is a service that could not be reached: the Hugging Face
	// API behind -hf. Nothing is wrong with what the caller typed, so it is not
	// a usage error, and retrying later may work.
	exitUnavailable = 69 // EX_UNAVAILABLE
)
