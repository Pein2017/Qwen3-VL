- [x] Add `toon_mode` flag to config schema, CLI, and dataset construction wiring.
- [x] Implement TOON serializer/decoder utilities and integrate with dense caption builder.
- [x] Mirror TOON spec delimiter & quoting rules (comma default, optional tab, escapes) and add regression fixtures for edge captions.
- [x] Introduce TOON-specific prompts and switch based on `toon_mode`.
- [x] Update docs/specs to describe JSON vs TOON outputs and geometry handling.
- [x] Extend visualization/eval tooling to parse TOON outputs via shared decoder.
- [x] Add tests plus smoke training/visual validation for both modes.

