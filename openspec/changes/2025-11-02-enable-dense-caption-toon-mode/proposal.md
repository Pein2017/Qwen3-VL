---
title: Add TOON serialization mode for dense caption dataset
author: Qwen3-VL
created: 2025-11-02
change-id: 2025-11-02-enable-dense-caption-toon-mode
status: draft
---

## Why

- **Token pressure**: Dense caption training still emits verbose JSON with repeated keys. Early measurements show assistant replies spend ~40–60% of tokens on structure rather than captions or coordinates, limiting sequence budget.
- **Ablation flexibility**: We want to compare learning dynamics between the current JSON format and a compact Token-Oriented Object Notation (TOON) variant without rewriting configs each time.
- **Prompt clarity**: TOON requires explicit guidance. Introducing a dedicated flag and prompt variant keeps patterns separated and avoids confusing the model with mixed instructions.

## What Changes

- **Global toggle**
  - Add a boolean `toon_mode` argument in the training config schema and CLI wiring. Default `false` to preserve existing output.
  - Thread the flag through dataset creation so dense caption builders know which serialization path to use.
- **Serializer infrastructure**
  - Implement a TOON builder (or branch) that emits `objs[N]{type,desc,xs}` rows where `type ∈ {0:bbox, 1:quad, 2:line}` and `xs` are normalized coordinates.
  - Follow the upstream TOON spec conventions for headers, delimiter suffixes (comma default, tab via `[N\t]`), and string quoting to minimize token count while keeping captions readable.
  - Ensure lines omit `line_points`; the row decoder infers point counts from the remaining coordinate pairs and re-populates `line_points` only when converting back to JSON.
  - Ship a parser that round-trips TOON text back into the canonical object dictionary so evaluation and visualization keep their existing inputs, reusing the same quoting and delimiter handling as the serializer.
- **Prompt + docs**
  - Add TOON-specific system/user prompt instructions and swap them in when `toon_mode` is true.
  - Update dense caption docs and references to describe both JSON (with `line_points`) and TOON payloads.
  - Extend specs to note the optional TOON format and its geometry conventions.
- **Validation & tooling**
  - Expand unit tests covering serializer selection and round-trip conversion, including fixtures lifted from the upstream TOON implementation for quoting/delimiter edge cases.
  - Update visualization/eval helpers to read TOON output via the shared parser.
  - Outline A/B training runs (JSON vs TOON) to measure convergence, accuracy, and token usage.

## Impact

- Surfaces: `src/config/schema.py`, `src/sft.py`, dense caption dataset/builder modules, prompt definitions, docs (`DATA_AND_DATASETS.md`, `REFERENCE.md`), visualization utilities, and related tests.
- Downstream: Serving/evaluation flows can continue consuming JSON because the TOON branch will be rehydrated before consumption. No contract change for summary mode.

## Validation Plan

1. **Unit tests**: Add serializer tests to ensure `toon_mode` switches outputs, and verify TOON ↔ JSON round-trips.
2. **Integration**: Run a smoke SFT job in both modes to confirm prompts, builders, and logging work end-to-end.
3. **Visualization**: Use `vis_tools/vis_qwen3.py` on TOON samples to show detections render correctly after parsing.
4. **Metrics**: Capture token counts per sample and compare convergence/accuracy between JSON and TOON runs.

## Open Questions

- ~~Do we require a dedicated `design.md` to capture tokenizer measurements, or will inline proposal notes suffice?~~
  - Inline documentation inside the proposal/code comments is sufficient; no separate `design.md` will be produced for this change.
- ~~How should we expose the parser externally (e.g., utility module vs. dataset builder method) so serving pipelines can opt-in later if needed?~~
  - The parser is exported via `src/datasets/builders/__init__.py` (`decode_toon_payload`, `encode_toon_block`, etc.), making it available to training, evaluation, and serving pipelines without duplicating code.


