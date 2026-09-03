# TODO

Open items only. Completed work is dropped from here — the CHANGELOG
is the record of what shipped.

- [ ] **Validate the speed estimates against real runs.** Nothing here has been
      checked against a stopwatch; the arithmetic is sound and the constants are
      from published figures, but the end-to-end numbers are unverified. The
      tensor-parallel and speculative paths raise the stakes: both now claim a
      multiplier, and neither has been measured.
- [ ] Read an actual GGUF header, so a specific file is measured rather than a
      format assumed
- [ ] A measured acceptance rate per draft/target pair, instead of one assumed
      default for all of them
