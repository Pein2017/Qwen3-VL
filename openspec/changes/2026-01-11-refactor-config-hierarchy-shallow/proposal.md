# Proposal: Refactor training config hierarchy for shallower, layer-aligned composition

## Why

The current training configuration system under `configs/` (introduced Jan 4–5, 2026) optimizes for DRY reuse by splitting configuration into many small `configs/components/*` YAML fragments and composing runnable presets via long `extends` chains.

In practice, this has become hard to maintain and debug:
- A single run often merges **12–17 YAML files**, making it difficult to discover which values are active.
- Overrides are order-sensitive (deep merge), so tracing a single key requires jumping across many files.
- List-valued blocks (notably augmentation ops) cannot be partially overridden; experiments must either duplicate the entire list or introduce additional layering.
- Dense vs summary policies are scattered across multiple components, which obscures the training-type boundary.

We want a shallower, layer-aligned hierarchy that matches how developers think about runs:
**base runtime → runnable preset → dataset fusion → (optional) overlay**.

## What

Introduce a new, shallower config mechanism while preserving `extends`:

### 3-layer architecture (target)
1. **Base layer**: `configs/base.yaml` — shared runtime defaults and immutable settings only.
2. **Experiment layer (runnable presets)**: runnable presets under `configs/train/**` that are **self-contained** (all important/tunable defaults live in the runnable preset).
3. **Dataset fusion layer**: `custom.fusion_config: configs/fusion/...` — keep the existing fusion overlay mechanism (works well).

Optional overlays (still under `configs/train/**`) MAY extend a runnable preset and only add a small focused set of keys (e.g., GKD knobs, vLLM server mode, augmentation off).

### Key design decisions (confirmed)
- **Keep `extends`** as the composition mechanism.
- **Dense vs summary stays explicit in runnable presets** (no separate `configs/types/` layer).
- **Augmentation profiles (coarse overlays)**: represent common variants as experiment-layer overlays (e.g., `dense_2048_aug_off.yaml`) rather than ad-hoc per-experiment full-list rewrites or a separate type layer.
- **Keep `global_max_length`** as the single length knob (it remains the only supported length control in configs).
- **No backward compatibility**: old config paths/names may be removed or renamed; docs are updated accordingly.

### Operator/developer UX improvements (in-scope, non-optional)
- **Config inspection and diff tooling is mandatory**:
  - Provide a canonical workflow (preferred: a script under `scripts/config_tools/`) that prints the resolved config and the full `extends` chain (merge order).
  - Provide a diff workflow that compares two configs *after* resolving `extends` (used for both debugging and migration parity checks).
- **Parity validation workflow is mandatory**:
  - Every migrated preset MUST pass an “old vs new resolved config” parity diff, with an explicit allowlist of “allowed diffs” and a strict set of “forbidden diffs”.

## Scope
- Training YAML hierarchy used by `scripts/train.sh` / `python -m src.sft`:
  - SFT dense + SFT summary
  - GRPO dense + GRPO summary (including vLLM server-mode overlay)
  - Stage-B distill (SFT)
  - Debug/smoke presets as needed
- Documentation updates for the new hierarchy and recommended modification workflow.

## Non-goals
- No behavior changes to training, datasets, prompts, augmentation semantics, GRPO rewards, or ms-swift integration.
- No new config DSL, Hydra, or programmatic config composition (inspection/diff tooling is allowed but does not change how configs are authored or loaded).
- No changes to Stage-A/Stage-B runtime config layout (e.g., `configs/stage_b/*.yaml`) unless explicitly required by training presets.

## Impact / Breaking changes
- Existing config file paths and names may change without compatibility shims.
- Team workflows should update to start experiments from runnable presets under `configs/train/**` (and apply overlays there when needed).

## Success criteria
- Typical training runs should be understandable by opening **≤2 files** (base preset + overlay when used), plus the referenced fusion config.
- Typical runnable presets SHOULD merge **≤4 YAML files** total (including `configs/base.yaml`), excluding fusion config resolution.
- A canonical inspection workflow exists and is documented. It MUST:
  - print the full resolved `extends` chain (merge order), and
  - emit a fully-resolved training config payload in a deterministic, diffable form (canonical: **key-sorted JSON**; YAML output may be supported for readability).
- A canonical diff/parity workflow exists and is documented. It MUST:
  - compare two configs after resolving `extends`,
  - clearly indicate “allowed diffs” vs “forbidden diffs”, and
  - return a non-zero exit code when forbidden diffs are detected (so it can gate migrations/CI).
- All in-tree runnable presets validate via `scripts/validate_sft_config.py`.
- `scripts/train.sh` continues to work without adding hyperparameter CLI flags.

## Risks
- Too much duplication across runnable presets can reintroduce drift; mitigated by keeping the number of runnable presets small and using targeted overlays for variants.
- Removing compatibility may break undocumented external users; accepted per decision (no backward compat).

## Rollout plan (high-level)
1. Add config inspection/diff tooling + document the workflow (mandatory prerequisite for safe migration).
2. Inline key defaults into runnable presets under `configs/train/**` and migrate one SFT + one GRPO preset as proof.
3. Migrate remaining presets and update docs/catalog; run parity validation per preset.
4. Remove legacy `configs/components/*` and old base presets once parity is confirmed and recorded.
