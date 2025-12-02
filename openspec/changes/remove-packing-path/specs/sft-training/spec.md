## ADDED Requirements

### Requirement: Packing disabled; padding-only batching
- The SFT runner SHALL reject any config that sets `training.packing` to true or supplies packing-specific knobs (`packing_group_key`, cached-length settings), returning a clear error that packing is removed and padding is the only supported batching mode.
- The default batching path SHALL use standard padding for `per_device_train_batch_size>1`, with no bin-packing or length-cache prepass executed in training or evaluation.
- Per-dataset telemetry SHALL remain available in this padding-only mode by attaching dataset labels from sample metadata, without invoking packing logic.
- The legacy packing implementation SHALL be removed from runtime import paths and stored only under `archive/packing/` for future reference.

#### Scenario: Config tries to enable packing
- WHEN a user launches training with `training.packing: true` (or any packing knobs)
- THEN initialization fails fast with an actionable error explaining packing is removed and to delete the packing settings.

#### Scenario: Normal padded training
- WHEN packing keys are absent and `per_device_train_batch_size>1`
- THEN the dataloader uses the padded collator, training proceeds, and aggregate metrics remain unchanged.

#### Scenario: Stray packing metadata
- WHEN a config includes `custom.packing_group_key` or `custom.cached_lengths`
- THEN validation fails with guidance to remove packing metadata because the feature is no longer supported.

#### Scenario: Packing code archived
- WHEN code attempts to import packing modules from the main package
- THEN the import fails; the archived implementation lives under `archive/packing/` and is not on the runtime path.

## MODIFIED Requirements

### Requirement: Dense-caption SFT with KL anchoring (GKD)
- When `rlhf_type == "gkd"`, the system SHALL keep the dense-caption dataset pipeline unchanged (same Stage-2/3 templates, augmentation, and LoRA targeting).
- A frozen `teacher_model` (base Qwen3-VL) SHALL be loaded with matching tokenizer/template.
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
- THEN LoRA targets and augmentation are identical; only trainer/teacher knobs differ
