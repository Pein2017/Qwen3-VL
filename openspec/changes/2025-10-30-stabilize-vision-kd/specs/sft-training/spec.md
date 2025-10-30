## ADDED Requirements

### Requirement: Vision/Aligner feature KD (optional)
- When `custom.visual_kd.enabled` is true, the trainer SHALL capture the student and teacher vision/aligner features produced by `model.model.visual` during the standard forward pass (without recomputing the encoder).
- The trainer SHALL compute a distance (`mse` at minimum) between corresponding student and teacher features for every target listed in `custom.visual_kd.targets` (`merger` and/or `deepstack`) and add `custom.visual_kd.weight * feature_loss` to the total loss.
- Feature-level KD MUST propagate gradients only through the student model; teacher tensors remain detached.

#### Scenario: deepstack + merger enabled
- GIVEN a config with `custom.visual_kd.enabled: true`, `targets: [merger, deepstack]`, and `distance: mse`
- WHEN a multimodal batch with images runs through training
- THEN the trainer reuses activations from the student/teacher forward passes, computes per-target MSE losses averaged over tokens, sums them, multiplies by `weight`, and adds the result to the batch loss.

### Requirement: Vision KD telemetry
- Training logs SHALL include `train/vision_kd_loss` (and `eval/vision_kd_loss` during evaluation) whenever the feature is enabled and images are present.
- Non-finite values MUST trigger a warning that names the metric and step, mirroring KL monitoring.

#### Scenario: logging during eval
- GIVEN `custom.visual_kd.enabled: true`
- WHEN evaluation runs on an image batch
- THEN the aggregated logs contain `eval/vision_kd_loss` with a finite scalar.

### Requirement: Disabled behavior
- With `custom.visual_kd.enabled: false` (or the key absent), the trainer SHALL skip hook registration, avoid computing extra losses, and keep training identical to the current implementation.

#### Scenario: legacy config
- GIVEN an existing Stage-3 GKD config without the new key
- WHEN loaded through `src/sft.py`
- THEN no additional hooks or losses are active; training outputs match the pre-change behavior.

### Requirement: Config interface
- `custom.visual_kd` SHALL be parsed by the config loader with defaults `enabled=false`, `weight=0.0`, `targets=[]`, `distance='mse'`.
- The loader MUST validate that `weight > 0` whenever the feature is enabled and reject unsupported `distance` or `targets` values.
- Stage configs (e.g., `stage_3_gkd.yaml` overlays) SHALL enable the feature purely via YAML edits—no CLI flags or hard-coded overrides.

#### Scenario: Stage-3 overlay enables vision KD
- GIVEN `stage_3_gkd.yaml` extended with:
  ```yaml
  custom:
    visual_kd:
      enabled: true
      weight: 0.5
      targets: [merger]
      distance: mse
  ```
- WHEN running `scripts/train.sh config=stage_3_gkd.yaml`
- THEN `src/sft.py` loads the new structure without additional CLI arguments, and the trainer activates the feature-level KD path.

### Requirement: Graceful fallback on missing visuals
- If a batch lacks `pixel_values` or the teacher/student caches are empty (e.g., text-only prompts), the trainer SHALL skip the feature KD computation and not alter the loss or metrics for that batch.

#### Scenario: evaluation on text-only samples
- GIVEN `custom.visual_kd.enabled: true`
- WHEN an eval batch contains no images (rare but possible for summary-only records)
- THEN the trainer emits no `vision_kd_loss` metric for that step and the total loss matches the CE-only path.


