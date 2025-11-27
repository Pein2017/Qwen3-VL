# llm-kd-config Specification

## Purpose
Specify the language-model KD weighting knob (`rlhf.llm_kd_weight`), its validation rules, and how trainers apply the weight to losses/telemetry while remaining compatible with visual KD.
## Requirements
### Requirement: Configurable LM-head KD weight
The training YAML SHALL expose a non-negative `rlhf.llm_kd_weight` knob that controls how strongly the language-model KD term contributes to the total loss.

#### Scenario: Default weight attachment
- **GIVEN** a training config that omits `rlhf.llm_kd_weight`
- **WHEN** `ConfigLoader.load_training_config` constructs `train_args`
- **THEN** both `train_args` and `train_args.training_args` expose `llm_kd_weight = 1.0`
- **AND** `ConfigLoader` removes `rlhf.llm_kd_weight` from the kwargs passed to `args_cls(**args_dict)` so ms-swift constructors never see the knob
- **AND** the loader attaches the default value via `setattr` on the returned `TrainArguments` objects

#### Scenario: Positive weight supplied
- **GIVEN** a training config with `rlhf.llm_kd_weight: 0.25`
- **WHEN** the config loader resolves TrainArguments
- **THEN** the same value (`0.25`) is attached to `train_args.llm_kd_weight` and `train_args.training_args.llm_kd_weight`
- **AND** the loader pops `rlhf.llm_kd_weight` before calling `args_cls(**args_dict)` and reattaches it via `setattr`, preserving ms-swift constructor signatures even for user-supplied values
- **AND** other `rlhf` values (e.g., `beta`, `sft_alpha`) continue to flow unchanged

#### Scenario: Invalid negative weight
- **GIVEN** a training config that sets `rlhf.llm_kd_weight: -0.1`
- **WHEN** the config loader attempts to build TrainArguments
- **THEN** it raises a `ValueError` stating the weight must be non-negative
- **AND** no TrainArguments instance is returned

### Requirement: Weighted LM-head KD loss & telemetry (updates KD/CE telemetry contract)
`GKDTrainerWithMetrics` SHALL respect `llm_kd_weight` when composing losses and metrics, while remaining compatible with visual KD.

#### Scenario: Positive weight with teacher attached
- **GIVEN** a teacher model is registered, `llm_kd_weight = 0.5`, and `visual_kd.enabled = false`
- **WHEN** `compute_loss` runs on a batch with valid label tokens
- **THEN** the total loss equals `0.5 * jsd_loss + sft_alpha * ce_loss`
- **AND** `_metrics[mode]["llm_kd_loss"]` receives the weighted value (`0.5 * jsd_loss`)
- **AND** aggregated logs include `train/llm_kd_loss` (or `eval/...`) with the same weighted number

#### Scenario: Weighted value surfaces in logging.jsonl
- **GIVEN** `llm_kd_weight = 0.5` and logging is enabled
- **WHEN** inspecting `logging.jsonl`
- **THEN** each record's `llm_kd_loss` field equals the weighted value (`0.5 * jsd_loss`)
- **AND** no second field exposes the unscaled JSD term

#### Scenario: Zero weight disables LM-head KD
- **GIVEN** `llm_kd_weight = 0` and a teacher model is attached
- **WHEN** `compute_loss` executes
- **THEN** the LM-head KD branch is skipped (no addition to `total_loss`)
- **AND** the CE contribution enters with weight `1.0` (ignoring `sft_alpha`)
- **AND** no `llm_kd_loss` entries are appended to `_metrics`
- **AND** aggregated logs omit `train/llm_kd_loss` / `eval/llm_kd_loss`

#### Scenario: Visual KD remains active when LM KD disabled
- **GIVEN** `llm_kd_weight = 0`, `visual_kd.enabled = true`, and the configured visual targets exist
- **WHEN** `compute_loss` runs
- **THEN** the trainer still performs the teacher forward pass necessary for visual KD hooks
- **AND** the visual KD loss is computed, added to `total_loss`, and logged as `vision_kd_loss`
- **AND** the CE loss contributes with weight `1.0` while visual KD respects its configured `custom.visual_kd.weight`
- **AND** LM-head KD metrics remain absent

#### Scenario: Teacher forward skipped when KD inactive
- **GIVEN** `llm_kd_weight = 0`, `visual_kd.enabled = false`, and a teacher checkpoint is configured
- **WHEN** `compute_loss` runs
- **THEN** `_run_teacher_forward` is not executed, so no teacher-side compute occurs
- **AND** the total loss reduces to the unweighted CE term (plus any other active losses such as regularizers)

### Requirement: KD teacher dependency enforcement
GKD training SHALL load or skip the teacher model based on the requested KD weights.

#### Scenario: Teacher required when KD requested
- **GIVEN** the resolved config sets `custom.visual_kd.weight > 0` or `rlhf.llm_kd_weight > 0`
- **WHEN** `ConfigLoader` validates `rlhf` arguments
- **THEN** it requires `rlhf.teacher_model` to be populated with a valid checkpoint
- **AND** it raises a `ValueError` immediately if the teacher path is absent (fail-fast; no fallback warning or silent disablement)

#### Scenario: Teacher skipped when KD inactive
- **GIVEN** `custom.visual_kd.weight = 0`, `visual_kd.enabled = false`, and `rlhf.llm_kd_weight = 0`
- **WHEN** the trainer is constructed
- **THEN** it does not instantiate or load a teacher model
- **AND** the run proceeds via the standard SFT path with only the CE objective
