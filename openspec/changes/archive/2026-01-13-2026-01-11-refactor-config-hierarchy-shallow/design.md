# Design: Shallower, layer-aligned training configs (base → preset → fusion → overlay)

## Goals
- Preserve `extends` while making configs easier to trace and modify.
- Make runnable presets the primary composition boundary (avoid a separate `configs/types/` layer).
- Reduce the number of merged YAML files per run by collapsing overly-granular components.
- Keep `global_max_length` as the single knob for length control.
- Keep fusion configs extend-aware and shallow (base + variants).
- Standardize augmentation as a small number of **coarse variants** (overlays) without reintroducing micro-component sprawl.
- Make config inspection/diff a first-class, mandatory workflow for debugging and migration safety.

## Non-goals
- No behavioral changes to training, reward functions, augmentation semantics, prompts, or dataset sampling.
- No backwards compatibility for removed config paths/names.
- No introduction of a new config DSL; keep YAML + `extends`.

## Proposed directory layout

```
configs/
  base.yaml                 # base runtime defaults (immutable choices only)
  debug.yaml                # default quick entrypoint (self-contained)

  fusion/                   # Dataset fusion configs (unchanged)
    base/
    variants/
    stage_b_distill.yaml

  train/                    # Experiment layer (runnable presets)
    sft/
      dense_1024.yaml
      dense_2048.yaml
      dense_1024_gkd.yaml
      summary_1024.yaml
      summary_2048.yaml
    grpo/
      dense_2048.yaml
      dense_2048_lowaug_t04_low_lr.yaml
      summary_1024.yaml
      summary_2048.yaml
      summary_server.yaml
    stage_b/
      distill.yaml

  smoke/
    *.yaml
```

Notes:
- This design keeps the **experiment paths** (`configs/train/**`) stable for operator ergonomics, even though backward compatibility is not required.
- `configs/components/*` is removed in the end-state (its content is folded into runnable presets under `configs/train/**`).
- Fusion configs remain extend-aware and name-merged for `targets`/`sources` (already proven).

## Composition rules

### Training configs (YAML → `ConfigLoader`)
- Runnable presets SHALL extend `configs/base.yaml` directly (no `configs/types/` layer).
- Overlays (when used) SHOULD extend exactly one runnable preset under `configs/train/**`.
- The experiment layer is responsible for all **required launch settings**:
  - `model.model`
  - `training.run_name`
  - `training.output_dir`
  - `training.logging_dir`
  - `training.num_train_epochs`
  - `training.learning_rate`
  - `training.vit_lr`
  - `training.aligner_lr`
- `global_max_length` remains the only supported length knob. Runnable presets set it explicitly, and the loader continues to proxy it into model/template/vLLM limits.
- Typical runnable presets SHOULD resolve with **≤4 merged YAML files** total (excluding fusion resolution). The intent is to keep most runs at `base + preset` (2 files) and overlays at `base + preset + overlay` (3 files).

### Fusion configs (YAML → `FusionConfig`)
- Experiments reference fusion mixes via `custom.fusion_config: configs/fusion/variants/...`.
- Fusion configs keep their existing base+variants structure and name-based list merging; no changes required.

## Mandatory config inspection/diff workflow

### Canonical tool
This change introduces a mandatory inspection/diff workflow implemented as a single canonical script:

```
scripts/config_tools/inspect_config.py
```

This is **not** a new config system: it uses the existing YAML + `extends` loader (`ConfigLoader.load_yaml_with_extends`) and only improves debuggability and migration safety.

### Inspect (single config)
Canonical command:

```bash
conda run -n ms python scripts/config_tools/inspect_config.py inspect --config configs/train/sft/dense_1024.yaml
```

Expected outputs (human-readable on stdout; machine-readable output is required for diff/parity):
- The resolved `extends` chain (merge order), including absolute or repo-relative file paths.
- A fully-resolved config payload *after* `extends` resolution, emitted as **key-sorted JSON** by default.
- Optional: support emitting YAML for readability (`--format yaml`), but JSON remains the canonical diff/parity format.

Primary use-cases:
- “What config values are actually active for this run?”
- “Which file set `custom.augmentation` / `rlhf.reward_funcs` / `global_max_length`?”
- Code review: audit changes to the final resolved config, not just the top-level experiment YAML.

### Diff (two configs after resolution)
Canonical command:

```bash
conda run -n ms python scripts/config_tools/inspect_config.py diff --left <old.yaml> --right <new.yaml> --profile parity
```

Expected outputs:
- A deterministic diff of the resolved configs (key-sorted), suitable for pasting into a PR discussion.
- A clear split between **allowed diffs** (operational/logging-only) vs **forbidden diffs** (training semantics).
- Exit code `0` when the parity profile passes.
- Exit code non-zero when forbidden diffs are detected under the parity profile (recommended: use this as a migration/CI gate).

## Parity validation (migration safety)

### Definition
“Parity” means that migrating a preset to the new hierarchy does **not** change training semantics.

Parity is evaluated by diffing the **resolved** configs (after `extends`), not the raw YAML files.

### Workflow (repeat per migrated preset)
1. Resolve and dump the old preset config.
2. Resolve and dump the new preset config.
3. Diff old vs new under the `parity` profile.
4. Record the result in a parity checklist table (old path → new path → parity status).

### Allowed diffs vs forbidden diffs

Allowed diffs (explicit allowlist; chosen for flexibility while still catching semantic drift):
- **Run identity / filesystem outputs**:
  - `training.output_dir`
  - `training.logging_dir`
  - `training.run_name`
  - `custom.dump_conversation_text`
  - `custom.dump_conversation_path`
- **Telemetry/logging verbosity** (no effect on training objective):
  - `training.logging_steps`
  - `training.logging_first_step`
  - `training.log_level`
  - `training.log_level_replica`
  - `training.report_to`
- **Checkpoint/eval scheduling** (operational differences allowed during migration as needed):
  - `training.save_strategy`
  - `training.save_steps`
  - `training.save_total_limit`
  - `training.save_last_epoch`
  - `training.eval_strategy`
  - `training.eval_steps`
  - `training.metric_for_best_model`
  - `training.greater_is_better`
- **Augmentation fields when augmentation is disabled**:
  - When `custom.augmentation.enabled` resolves to `false` in **both** configs, diffs under:
    - `custom.augmentation*`
    - `custom.augmentation_curriculum*`
    are treated as allowed diffs, because the training entrypoint ignores augmentation ops/curriculum when augmentation is disabled.

Rationale:
- We allow operational/logging/saving differences so teams can migrate with flexibility-first ergonomics.
- We still forbid changes that alter the dataset, augmentation, optimization hyperparameters, or RLHF behavior.

Forbidden diffs (everything else, including but not limited to):
- Any change to `global_max_length`.
- Any change to `model.*`, `template.*`, `data.*`, `tuner.*`, `deepspeed.*`.
- Any change to `training.*` other than the allowlist fields above (batch sizes, schedulers, optimizer hyperparameters, etc.).
- Any change to `custom.*` that affects dataset selection, prompting, augmentation, or training behavior (including `custom.use_summary`, `custom.fusion_config`, and `custom.augmentation*`), except for augmentation fields that are ignored when augmentation is disabled (see allowed diffs above).
- Any change to `rlhf.*` (reward funcs/weights, rollout settings, vLLM mode/settings, etc.).

Rationale: the goal of this refactor is *config structure and UX*, not training behavior changes. We treat semantic changes as regressions during migration.

## Augmentation profiles (coarse overlays)

### Goals
- Avoid per-experiment full-list rewrites of augmentation ops for common variants (low augmentation, augmentation disabled).
- Avoid reintroducing a large `configs/components/**` micro-fragment hierarchy.
- Keep “open 1 file to see defaults” true for the runnable preset.

### Mechanism (YAML + `extends` only)
Augmentation profiles are represented as **experiment-layer overlays** under `configs/train/**`.

Base runnable presets inline the default augmentation policy (and curriculum if applicable) under:
- `custom.augmentation`
- `custom.augmentation_curriculum`

Coarse variants SHOULD be implemented as small overlays that extend a runnable preset and override only augmentation-related keys, for example:
- `custom.augmentation.enabled: false`
- `custom.augmentation.ops: []`
- `custom.augmentation_curriculum: null`

This keeps typical runs at `base + preset` while still letting teams express “aug_off” / “lowaug” / “semantic_aug” without deep component chains.

### Profile budget
To avoid sprawl, each runnable preset SHOULD have at most **2–3** supported overlays, and new overlays should be introduced only when multiple experiments genuinely share the same policy.

### Minimal profile set (target)
This change keeps the set intentionally small and rooted in runnable presets. Examples of in-tree overlays:
- `configs/train/grpo/dense_2048_lowaug_t04_low_lr.yaml` (augmentation disabled for variance-sensitive RL)
- `configs/train/grpo/summary_server.yaml` (vLLM server mode overlay)
- `configs/train/sft/dense_1024_gkd.yaml` (GKD overlay)

## “Tunable vs fixed” guidelines

### Base layer (`configs/base.yaml`)
- Immutable runtime choices only: dtype, attention impl, template skeleton.
- No hyperparameters here.

### Runnable preset layer (`configs/train/**`)
- Defines the *policy* for that runnable preset:
  - Prompt profile defaults (`prompts.profile`, domains if applicable)
  - `custom.use_summary` and summary-specific toggles
  - Default augmentation and curriculum blocks for that preset (OK to be large)
  - Default DeepSpeed policy for that preset (e.g., ZeRO2 vs ZeRO3)
  - Default GRPO wiring for GRPO presets (`rlhf` block)
  - Default LoRA policy / tuner config for that preset
  - Default training knobs that are stable across experiments (batch sizes, scheduler defaults, etc.)
- Runnable presets SHOULD set all important/tunable fields explicitly (avoid `null` placeholders that make it easy to forget what is active).

### Overlay layer (`configs/train/**` overlays)
- “Small focused changes on top of a runnable preset”:
  - KD/GKD knobs
  - vLLM server vs colocate mode
  - disable augmentation (or apply a shared coarse variant)
- Overlays SHOULD remain minimal and should not re-specify the whole training config.

## Examples

### Runnable preset YAML (dense SFT 1024)
```yaml
# configs/train/sft/dense_1024.yaml
extends:
  - ../../base.yaml

global_max_length: 12000

custom:
  use_summary: false
  json_format: standard
  emit_norm: norm1000
  assistant_prefix_format: "<DOMAIN={domain}>, <TASK={task}>"
  augmentation: { ... }  # default dense augment (may be large)

training:
  output_dir: ./output/...
  logging_dir: ./tb/...
  run_name: ...
  num_train_epochs: 30
  learning_rate: 2.0e-4
  vit_lr: 1.0e-4
  aligner_lr: 6.0e-4

custom:
  fusion_config: configs/fusion/variants/bbu_rru_dense_1024.yaml
```

### Overlay YAML (GKD on dense 1024)
```yaml
# configs/train/sft/dense_1024_gkd.yaml
extends:
  - dense_1024.yaml

rlhf:
  rlhf_type: gkd
  teacher_model: /abs/path/to/Qwen3-VL-4B-Instruct
  beta: 0.5
  sft_alpha: 0.3
  seq_kd: true
  lmbda: 0.5
  max_completion_length: 256
  temperature: 0.9

custom:
  trainer_variant: gkd_monitor
```

## Migration strategy (incremental)
- Phase 0 (mandatory): Add `scripts/config_tools/inspect_config.py` and document inspect/diff + parity workflows.
- Phase 1: Inline key defaults into runnable presets under `configs/train/**` and migrate a minimal set of experiments (1× SFT, 1× GRPO) to validate the structure; run parity diffs.
- Phase 2: Migrate all `configs/train/**` experiments and `configs/smoke/**` presets; run parity diffs per preset and record results.
- Phase 3: Remove `configs/components/**` and legacy base presets.
