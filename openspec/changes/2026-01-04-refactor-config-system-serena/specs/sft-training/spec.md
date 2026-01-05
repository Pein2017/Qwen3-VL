# sft-training Spec Delta

## ADDED Requirements

### Requirement: Modular training config hierarchy
- The project SHALL organize runnable training presets under `configs/train/` and reusable components under `configs/components/`.
- Runnable presets SHALL compose component YAMLs using `extends` and only override deltas (dataset mix, output paths, run names, mode toggles).
- Component YAMLs SHALL be section-scoped (single top-level key) to keep overrides explicit.

#### Scenario: Dense preset composition
- **GIVEN** `configs/train/sft/dense_1024.yaml`
- **WHEN** `ConfigLoader.load_yaml_with_extends` resolves `extends`
- **THEN** the resolved config includes base runtime defaults, SFT training defaults, dense augmentation defaults, and a `custom.fusion_config` pointing to the 1024 variant without repeating full sections.

## MODIFIED Requirements

### Requirement: Config overlays (composable hierarchy)
- The project SHALL provide YAML overlays that inherit the base SFT preset under `configs/train/sft/base.yaml` (or its successor) and only add:
  - `rlhf_type: gkd`, `teacher_model`, `beta`, `sft_alpha`, `seq_kd`, `lmbda`, `max_completion_length`, `temperature`, `llm_kd_weight`
  - `custom.trainer_variant` (for GKD monitoring)
- Launch command MUST remain `python -m src.sft --config <yaml>`.

#### Scenario: Overlay application
- **GIVEN** `configs/train/sft/dense_1024.yaml`
- **WHEN** applying a GKD overlay
- **THEN** LoRA targets and augmentation are identical to the base preset and only the GKD-related `rlhf` and `custom` keys differ.
