# TODO

Open items only. Completed work is dropped from here — the CHANGELOG
is the record of what shipped.

- [ ] **Validate the speed estimates against real runs.** Nothing here has been
      checked against a stopwatch; the arithmetic is sound and the constants are
      from published figures, but the end-to-end numbers are unverified.
- [ ] Read an actual GGUF header, so a specific file is measured rather than a
      format assumed
- [ ] Speculative decoding and draft-model pairs
- [ ] Multi-GPU tensor parallelism (currently modelled as layer split, which is
      right for llama.cpp and pessimistic for vLLM)
