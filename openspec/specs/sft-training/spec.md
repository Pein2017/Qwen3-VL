# sft-training Specification

## Purpose
Specify Qwen3‑VL SFT training behavior, including GKD integration, KD/CE telemetry, config plumbing, and stage-specific trainer expectations.
## Requirements
### Requirement: Dense-caption SFT with KL anchoring (GKD)
- When `rlhf_type == "gkd"`, the system SHALL keep the dense-caption dataset pipeline unchanged (same Stage-2/3 templates, augmentation, packing, LoRA targeting).
- A frozen `teacher_model` (base Qwen3‑VL) SHALL be loaded with matching tokenizer/template.
- The trainer SHALL minimize `loss = sft_alpha * CE(student, labels) + beta * KL(teacher||student)` and run on multimodal batches.
- The teacher MUST be no-grad and never updated.
- Student logits and labels MUST be aligned via left-shifted slicing (`logits[:, :-1]` vs `labels[:, 1:]`); rotations/wrap-around that leak BOS tokens into the final position are forbidden.
- Teacher logits MUST be gathered on the identical masked positions, and the trainer SHALL raise an error if teacher/student vocabulary sizes diverge.

#### Scenario: Valid multimodal batch under GKD
- GIVEN a dataset sample with images + chat messages
- WHEN `rlhf_type=gkd` is enabled
- THEN the forward pass computes both CE and KL without altering inputs or template behavior

#### Scenario: Token alignment regression
- GIVEN labels whose first supervised token is valid (non `-100`)
- WHEN the trainer prepares logits for KL/accuracy
- THEN the first token is ignored (no wrap-around), only the aligned next-token pairs participate, and accuracy reflects the same mask as KL

#### Scenario: Teacher vocab mismatch on forward
- GIVEN a teacher checkpoint whose vocabulary size differs from the student
- WHEN the trainer tries to compute KL
- THEN it raises an actionable `ValueError` naming both vocab dimensions instead of padding or silently copying logits

#### Scenario: Teacher mismatch
- GIVEN a teacher checkpoint with incompatible tokenizer/template
- WHEN initializing GKD
- THEN initialization fails with an actionable error (model/template mismatch)

---

### Requirement: Config overlays (no refactor)
- The project SHALL provide YAML overlays that inherit the current stage SFT configs and only add:
  - `rlhf_type: gkd`, `teacher_model: <abs path>`, `beta`, `sft_alpha`, `seq_kd`, `lmbda`, `max_completion_length`, `temperature`.
- Launch command MUST remain `python -m src.sft --config <yaml>`.

#### Scenario: Overlay application
- GIVEN `stage_3_vision_llm.yaml`
- WHEN applying `stage_3_gkd.yaml`
- THEN LoRA targets, augmentation, and packing are identical; only trainer/teacher knobs differ

---

### Requirement: KD/CE telemetry
- Training logs SHALL expose `train/llm_kd_loss`, `train/vision_kd_loss`, `train/sft_loss`, total `train/loss`, and `train/token_acc` every logging step, and emit matching `eval/*` metrics without duplicating prefixes (e.g., no `train/eval/*`).
- `logging.jsonl` SHALL include these fields; tensorboard curves SHALL be emitted when enabled.
- Evaluation mode SHALL include the teacher forward when a teacher model is configured so the same KD breakdown (`eval/llm_kd_loss`, `eval/vision_kd_loss`, `eval/sft_loss`) is reported.
- Teacher forwards MUST run without enabling `torch.autocast` when DeepSpeed is active unless the DeepSpeed config explicitly turns on autocast; otherwise dtype mismatches SHALL be avoided by casting teacher logits to the student dtype.

#### Scenario: KD spikes detection
- GIVEN a run with `beta>0`
- WHEN reading `logging.jsonl`
- THEN entries contain finite KD values; NaN must trigger a clear warning that cites the metric name and step

#### Scenario: DeepSpeed autocast guard
- GIVEN a DeepSpeed configuration that does not set `torch.autocast`
- WHEN the trainer executes the teacher forward
- THEN it skips wrapping the call in `torch.autocast`, preventing DeepSpeed's communication-precision assertion while still returning logits in the student's dtype

---

### Requirement: Backward compatibility
- With `rlhf_type` unset, behavior SHALL be identical to today’s SFT.
- Switching from SFT→GKD SHALL not require code changes; only config changes are allowed.

#### Scenario: Vanilla SFT config
- GIVEN `stage_3_vision_llm.yaml` without an `rlhf` block
- WHEN loaded through `src.sft`
- THEN the pipeline instantiates `SwiftSft`, no teacher model is loaded, and training arguments match the pre-change behavior. When the telemetry wrapper is used without a teacher, it SHALL fall back to logging only `*/sft_loss` while leaving KD metrics absent.

---

### Requirement: Smoke test
- A tiny dataset (≤32 samples) SHALL complete ≥1 epoch using GKD with `beta>0` and write checkpoints and `logging.jsonl`.
- Expected wall-clock increase (~2x fwd passes) SHALL be documented in the guide.

#### Scenario: Minimal run
- GIVEN teacher + student paths and a small JSONL
- WHEN running `--config stage_3_gkd.yaml`
- THEN training finishes, checkpoints are saved, and the KD/CE breakdown (`*/llm_kd_loss`, `*/vision_kd_loss`, `*/sft_loss`) is logged

---

### Requirement: Example overlay snippet (documentation-only)
- The documentation SHALL include a minimal YAML snippet that demonstrates the required `rlhf` keys and the custom trainer toggle for GKD overlays.

#### Scenario: Reference doc snippet
- GIVEN `docs/REFERENCE.md`
- WHEN reading the "KL Anchoring with GKD" section
- THEN a YAML fragment is present showing the GKD keys (`rlhf_type`, `teacher_model`, `beta`, `sft_alpha`, `seq_kd`, `lmbda`, `max_completion_length`, `temperature`) and the custom `trainer_variant: gkd_monitor`

---

### Requirement: Forward-only KD guidance (no on-policy)
- The documentation SHALL provide a recommended configuration for forward-only KD (no teacher/student sampling) for domain migration use-cases.
- It SHALL specify: `seq_kd: false`, `lmbda: 0.0`, recommended `sft_alpha` (≈1.0) and `beta` (≈0.1) when teacher==student base, and note that `temperature` and `max_completion_length` are ignored in this mode.

#### Scenario: Domain migration without on-policy
- GIVEN a downstream domain migration task where teacher equals the base model
- WHEN reading the GKD section in `docs/REFERENCE.md`
- THEN a YAML example with `seq_kd: false`, `lmbda: 0.0`, `sft_alpha: 1.0`, `beta: 0.1` is present with a note that generation knobs are inactive

### Requirement: Vision/Aligner feature KD (optional)
- When `custom.visual_kd.enabled` is true, the trainer SHALL capture the student and teacher vision/aligner features produced by `model.model.visual` during the standard forward pass (without recomputing the encoder).
- The trainer SHALL compute a distance (`mse` at minimum) between corresponding student and teacher features for every target listed in `custom.visual_kd.targets` (`merger` and/or `deepstack`) and add `custom.visual_kd.weight * feature_loss` to the total loss.
- Feature-level KD MUST propagate gradients only through the student model; teacher tensors remain detached.

#### Scenario: deepstack + merger enabled
- GIVEN a config with `custom.visual_kd.enabled: true`, `targets: [merger, deepstack]`, and `distance: mse`
- WHEN a multimodal batch with images runs through training
- THEN the trainer reuses activations from the student/teacher forward passes, computes per-target MSE losses averaged over tokens, sums them, multiplies by `weight`, and adds the result to the batch loss.

### Requirement: Vision KD telemetry
- Training logs SHALL include `train/vision_kd_loss` (and `eval/vision_kd_loss` during evaluation) whenever the feature is enabled and images are present, reporting the weighted term alongside `*/llm_kd_loss` and `*/sft_loss`.
- Non-finite values MUST trigger a warning that names the metric and step, mirroring KL monitoring.

#### Scenario: logging during eval
- GIVEN `custom.visual_kd.enabled: true`
- WHEN evaluation runs on an image batch
- THEN the aggregated logs contain `eval/vision_kd_loss` with a finite scalar.

#### Scenario: missing teacher fallback
- GIVEN `custom.visual_kd.enabled: true` but the active trainer has no `teacher_model`
- WHEN training initializes
- THEN the system emits a warning and disables visual KD, continuing with only the remaining loss terms.

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

### Requirement: Opt-in grouped packing without upstream changes
The SFT pipeline SHALL provide an opt-in grouped packing mode implemented within this repository (no edits to the installed ms-swift package) that packs sequences only within the same group key.

#### Scenario: Enable grouped packing
- **WHEN** training is configured with `packing_group_key: "_fusion_source"`
- **THEN** the dataloader uses the local grouped packing wrapper instead of ms-swift’s default packer
- **AND** each packed sample contains only records sharing that group key.

### Requirement: Support single- and multi-dataset training
Grouped packing SHALL work for both single-dataset (BaseCaptionDataset) and fusion (FusionCaptionDataset) runs without changing user YAML beyond the existing `packing: true` plus optional group config.

#### Scenario: Single dataset with packing
- **WHEN** `packing: true` is set and no `packing_group_key` is provided
- **THEN** the dataset is packable with unchanged behavior vs today (no grouping), and training succeeds.

#### Scenario: Fusion dataset with grouping
- **WHEN** `packing: true` and `packing_group_key` is set to a fusion field (e.g., `_fusion_domain` or `_fusion_source`)
- **THEN** packing is applied and sequences never mix records from different groups.

### Requirement: Epoch-aware packing
The grouped packing wrapper SHALL rebuild its binning each epoch so it stays aligned with Fusion per-epoch quotas/shuffle.

#### Scenario: Epoch advances
- **WHEN** the training epoch increments (callback `set_epoch`)
- **THEN** the packed bins are recomputed using the dataset’s current schedule, reflecting any new target downsampling or source resampling.

### Requirement: Configurable grouping mode
The system SHALL expose a config knob to choose the grouping key (`none|dataset|domain|custom_field`), defaulting to current ms-swift behavior when unset.

#### Scenario: Default mode
- **WHEN** the grouping knob is absent
- **THEN** packing behaves exactly like ms-swift (no grouping, same outputs as today).

#### Scenario: Domain grouping
- **WHEN** `packing_group_key: "_fusion_domain"` is set
- **THEN** packs are constrained to a single domain (e.g., `target` vs `source`).

### Requirement: Preserve default behavior when unset
The grouped packing mode SHALL be disabled by default; when `packing_group_key` is not set, packing behavior remains identical to current ms-swift packing.

#### Scenario: Group key absent
- **WHEN** `packing_group_key` is omitted
- **THEN** the pipeline uses the existing packing path with no grouping and identical outputs to today.

### Requirement: Retain partial packs (no drop_last)
The grouped packing implementation SHALL keep partial packs (no drop_last) to avoid data loss, matching current throughput expectations.

#### Scenario: Underfilled pack
- **WHEN** the final pack for a group is underfilled
- **THEN** it is still emitted rather than dropped.

### Requirement: Propagate packed group for metrics
The grouped packing implementation SHALL attach the group identifier (e.g., `_fusion_source`) to each packed batch so that per-dataset loss metrics can be computed without changing the global loss.

#### Scenario: Per-dataset loss logging
- **WHEN** a packed batch is produced under grouped packing
- **THEN** the batch includes `packed_group`, and the trainer can log mean loss by `packed_group` while keeping overall loss intact.

### Requirement: Preserve multimodal fields and masks
Packing MUST leave per-row multimodal tensors (`pixel_values`, `image_grid_thw`, etc.) and `position_ids` intact so Qwen3-VL flash-attention and padding-free paths remain correct.

#### Scenario: Packed multimodal batch
- **WHEN** a packed batch contains vision tokens
- **THEN** the collator still emits correct `position_ids/text_position_ids` (and `cu_seq_lens` where applicable) and no additional attention mask is required.

### Requirement: Strip aux metadata before model forward
Any grouping/metric metadata added for packing SHALL be removed or ignored before calling the model forward to avoid unexpected-kwargs errors.

#### Scenario: Forward pass
- **WHEN** a packed batch reaches the model
- **THEN** only model-accepted kwargs remain; `packed_group`/aux fields are consumed by the trainer or callback layer.

### Requirement: Edge-case coverage for packing
The grouped packing path SHALL be validated against extreme and boundary conditions using the local fusion config `configs/fusion/bbu_rru_lvis_coig.yaml`.

#### Scenario: Single tiny target, large sources
- **GIVEN** only one small target split (e.g., `val_tiny` for rru) and large sources
- **WHEN** packing with grouping is enabled
- **THEN** packs are built from the tiny target without crash, sources are excluded from eval, and per-group loss logging still functions.

#### Scenario: Uneven target ratios with epoch rebuild
- **GIVEN** multi-target quotas derived from ratios that change the sampled subset each epoch
- **WHEN** advancing epoch triggers `set_epoch`
- **THEN** packed bins are recomputed and reflect the new sampled indices (no reuse of stale bins).

#### Scenario: Multimodal packed batch (images present)
- **WHEN** packing a batch that includes image tokens (Qwen3-VL vision fields)
- **THEN** `text_position_ids`/`position_ids` are preserved, flash-attention runs without attention_mask errors, and image grids align with token placeholders.

#### Scenario: Long-sequence near packing_length
- **WHEN** sequences approach `packing_length` and cannot co-pack
- **THEN** they are emitted as single-item packs without truncation or drop.

#### Scenario: Fallback with grouping disabled
- **WHEN** `packing: true` but `packing_group_key` is unset
- **THEN** behavior matches ms-swift default packing (no group leakage, identical pack counts as baseline).

