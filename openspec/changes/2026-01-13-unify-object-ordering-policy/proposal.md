# Proposal: Unify object ordering policy across conversion, augmentation, and builders

## Why

Dense-caption supervision is sensitive to *ordering drift*:
- The assistant payload enumerates objects as `object_1..N`, so any instability changes the mapping between geometry/description pairs and indices.
- Augmentation can move objects; if ordering is not recomputed deterministically, training sees inconsistent `object_n` identities.
- Mixed geometry types (`bbox_2d` / `poly` / `line`) currently use different reference points for ordering, and polygon transforms can change the “first vertex”, which historically caused TL→BR reordering regressions.

To stabilize training and evaluation, we want a **single, explicit ordering policy** that is:
- deterministic (seed-invariant),
- consistent across *data conversion → preprocessing/augmentation → chat building*,
- compatible with both dense and summary modes.

## What

### 1) Sequence-level ordering (objects list)
Define a single TL→BR ordering policy based on **geometry center** (not “first vertex”):
- Primary key: `center_y` ascending
- Secondary key: `center_x` ascending
- Tie-breakers: stable, deterministic, and documented (e.g., geometry type + legacy reference point)

**Center definition**: center of the geometry’s axis-aligned bounding box (AABB):
- `bbox_2d`: AABB is the bbox itself
- `poly` / `line`: AABB computed from all vertices/points

### 2) Geometry-level ordering (polygon vertex list)
Polygons MUST be canonicalized everywhere they are produced/serialized:
- clockwise orientation
- start vertex = top-left vertex (min y, then min x)

This requirement is independent of sequence ordering (even if sequence ordering is center-based) and keeps polygon serialization stable across augmentation and builders.

### 2b) Geometry-level ordering (line/polyline direction)
Line (polyline) objects MUST use an endpoint-based canonical direction:
- Preserve the path order of intermediate vertices; **do not** rotate/re-index the sequence.
- The only allowed normalization is to reverse the entire sequence (swap start/end) when needed.
- Canonical start selection: compare the two endpoints `(x0, y0)` and `(xn, yn)` lexicographically by `(x, y)` (leftmost first, tie-break by topmost). If the end endpoint is "smaller", reverse the sequence.

This reduces label variance under flip/rotate/crop while respecting the fact that polylines are not cyclic.

### 3) Single source of truth + config-driven selection
Introduce/standardize a config-level “object ordering policy” knob that is shared by:
- data conversion (`data_conversion/`)
- training-time builders (`src/datasets/builders/`)
- any preprocessing step that needs to validate ordering

Operational default behavior SHALL be **center-based** (`center_tlbr`) while still supporting legacy reference-point ordering (`reference_tlbr`) for backward compatibility and regression diffs. The existing “preserve annotation order” escape hatch remains supported for byte-for-byte regression diffs.

Notes on defaults (important for operators):
- The repository-level wrappers/configs default to `center_tlbr` (conversion wrapper + training presets).
- The legacy mode remains available by explicitly setting `reference_tlbr`.

## Scope
- Conversion ordering: `data_conversion/utils/sorting.py` + callers in `data_conversion/pipeline/`.
- Training builder ordering: `src/datasets/builders/jsonlines.py` (and any other builder that enumerates `object_n`).
- Augmentation: ensure polygon canonicalization remains invariant after affine + clipping paths.
- Augmentation: ensure line direction canonicalization is applied after augmentation outputs are produced.
- Prompt/spec/docs: update ordering wording so “prompt contract” matches the implementation.
- Tests: add unit tests that lock in ordering and canonicalization invariants.

## Non-goals
- Changing semantic meaning of geometries (only ordering / serialization changes).
- Rewriting augmentation operators or conversion geometry extraction beyond ordering/canonicalization.
- Any changes to Stage-A output JSONL record ordering (record order in files remains explicitly non-contractual).

## Impact / Compatibility

### Behavioral changes
- By default, **conversion and training presets** emit/enumerate objects in **center-based TL→BR** order.
- Legacy behavior remains supported by setting `object_ordering_policy=reference_tlbr` explicitly (conversion + training must match).

### Observed delta on current tiny corpora (for awareness)
On the checked-in tiny samples:
- `data_new_schema/bbu_full_1024/train_tiny.jsonl`: center-based ordering differs from current reference-point ordering in **17/20 records (85%)**
- `data_new_schema/rru_full_1024_poly/train_tiny.jsonl`: differs in **11/20 records (55%)**

This confirms the policy change is *not* a no-op and should be rolled out intentionally (with explicit opt-in/compat knobs where needed).

### Backward compatibility plan
- Keep the existing `--preserve_annotation_order` (skip reordering) for regression diffs.
- Keep legacy ordering mode (`reference_tlbr`) available via config/CLI for regression diffs and older exports.
- Make center-based ordering (`center_tlbr`) the operational default for conversion wrappers and training presets.
- Update prompts/docs/specs so that the active policy is unambiguous (and does not contradict labels).

## Success criteria
- Given the same record, conversion output ordering and training builder ordering are identical.
- After augmentation, polygon vertex lists remain canonicalized (clockwise, start at top-left).
- After augmentation, line direction is canonicalized (endpoint-based direction; only full reversal allowed).
- Object ordering is deterministic and does not depend on dict/hash iteration order.
- Unit tests cover bbox/poly/line ordering and a representative augmentation case (rotate/flip + TLBR sorting).

## Risks
- Switching ordering policy may change historical `object_n` indices, impacting existing regression baselines and any downstream tooling that (incorrectly) depends on old index assignments.
- “Center” vs “top-left” ordering may change subjective readability for some edge cases; tie-breakers must be explicit to avoid surprises.
- Canonicalizing polygon/line serialization changes golden fixtures and byte-for-byte comparisons; tests/fixtures must be updated intentionally.

## Validation plan
- `openspec validate 2026-01-13-unify-object-ordering-policy --strict`
- Unit tests for ordering + canonicalization invariants
- Conversion smoke: run `data_conversion/convert_dataset.sh` on `train_tiny.jsonl` corpora and diff ordering keys
- Training smoke: build a few conversations (with augmentation on) and assert object indices match conversion ordering.
