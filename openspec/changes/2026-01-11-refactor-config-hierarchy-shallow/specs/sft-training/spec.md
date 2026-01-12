# sft-training Spec Delta

## ADDED Requirements

### Requirement: Layer-aligned training config hierarchy (base → preset → fusion → overlay)
- The project SHALL keep `extends` as the YAML composition mechanism for training configs.
- Training configs SHALL be organized into the following conceptual layers:
  1) Base runtime defaults (`configs/base.yaml`)
  2) Runnable presets under `configs/train/**` that are **self-contained** (important/tunable defaults live in the runnable preset)
  3) Dataset fusion selection via `custom.fusion_config` pointing to `configs/fusion/*`
  4) Optional overlays under `configs/train/**` that extend a runnable preset and only add a small focused set of keys (e.g., GKD knobs, vLLM server mode, augmentation off)
- Experiments SHALL override required launch settings explicitly (no hidden defaults).
- Typical runnable presets SHOULD resolve with **≤4 merged YAML files** total (excluding fusion resolution) to keep config comprehension and debugging tractable.

#### Scenario: Dense SFT preset resolves in ≤2 layers
- **GIVEN** `configs/train/sft/dense_1024.yaml` that extends `configs/base.yaml`
- **WHEN** `ConfigLoader.load_yaml_with_extends` resolves `extends`
- **THEN** the resolved config includes base runtime defaults plus the runnable preset defaults without requiring additional component fragments.

### Requirement: Config inspection and diff tooling is available (mandatory)
- The project SHALL provide a canonical inspection/diff workflow for training configs, implemented as a script under `scripts/` (preferred: `scripts/config_tools/inspect_config.py`).
- The inspection tool SHALL:
  - resolve a training config after applying `extends` (using the existing loader behavior),
  - print the full resolved `extends` chain (merge order), and
  - emit a fully-resolved config payload in a deterministic, diffable representation (canonical: **key-sorted JSON**; optional YAML output may be supported for readability).
- The diff tool SHALL compare two configs *after* resolving `extends` and produce a deterministic diff suitable for review.

#### Scenario: Operator inspects resolved config and merge chain
- **GIVEN** `configs/train/sft/dense_1024.yaml`
- **WHEN** running `conda run -n ms python scripts/config_tools/inspect_config.py inspect --config configs/train/sft/dense_1024.yaml`
- **THEN** the output includes the ordered `extends` chain and the fully-resolved config payload.

#### Scenario: Operator diffs two resolved configs
- **GIVEN** two training configs with different `extends` trees
- **WHEN** running `conda run -n ms python scripts/config_tools/inspect_config.py diff --left <a.yaml> --right <b.yaml>`
- **THEN** the diff is computed on the resolved configs (not raw YAML) and is stable across runs.

### Requirement: Parity validation is required for config migration
- When migrating presets to the new hierarchy, the project SHALL validate parity by diffing **resolved configs** (after `extends`) between the old preset and the new preset.
- The parity workflow SHALL define an explicit allowlist of “allowed diffs” and MUST fail on “forbidden diffs” that change training semantics.
- Allowed diffs SHOULD cover run identity and operational telemetry/saving/eval scheduling keys, while continuing to forbid changes that alter dataset selection, augmentation policy, optimization hyperparameters, or RLHF behavior.
- The parity diff tool SHOULD return a non-zero exit code when forbidden diffs are detected so migrations can be gated.

#### Scenario: Parity diff passes when only run identity changes
- **GIVEN** an old preset and its migrated new preset
- **WHEN** running parity diff
- **THEN** the check passes if the only differences are in run identity fields (e.g., output/logging dirs) and there are no semantic diffs.

#### Scenario: Parity diff fails on training semantic change
- **GIVEN** an old preset and a new preset where `training.learning_rate` differs after resolution
- **WHEN** running parity diff
- **THEN** the check fails and reports the forbidden key change.

### Requirement: Augmentation profiles are coarse experiment-layer overlays
- The project SHALL support a small number of coarse augmentation profiles (e.g., default/low/off) as **experiment-layer overlays** under `configs/train/**` (e.g., `configs/train/grpo/dense_2048_lowaug_t04_low_lr.yaml`).
- Overlays SHOULD extend exactly one runnable preset and override only augmentation-related keys, keeping the run shallow and avoiding micro-component sprawl.
- The runnable preset MUST remain self-contained so developers can open one file to see the active defaults for that run.
- Overlay filenames SHOULD be semantic and are not required to follow a single global naming scheme.

#### Scenario: Selecting an augmentation-off overlay stays shallow and discoverable
- **GIVEN** an overlay preset extending a runnable preset (e.g., `configs/train/grpo/dense_2048_lowaug_t04_low_lr.yaml` extending `configs/train/grpo/dense_2048.yaml`)
- **WHEN** inspecting the runnable preset and the overlay
- **THEN** the augmentation policy is discoverable by opening those files and the resolved preset does not require a deep chain of micro-fragments.

## MODIFIED Requirements

### Requirement: Config overlays (GKD) apply at the experiment layer
- The project SHALL provide YAML overlays for GKD that inherit an existing SFT experiment config and only add:
  - `rlhf_type: gkd`, `teacher_model: <abs path>`, `beta`, `sft_alpha`, `seq_kd`, `lmbda`, `max_completion_length`, `temperature`
  - `custom.trainer_variant` when required for telemetry wrappers
- Launch command MUST remain `python -m src.sft --config <yaml>`.

#### Scenario: Overlay application on dense SFT experiment
- **GIVEN** `configs/train/sft/dense_1024.yaml`
- **WHEN** applying a GKD overlay that extends it
- **THEN** dataset fusion selection, augmentation, and LoRA targeting remain unchanged and only the GKD-related `rlhf` and `custom` keys differ.
