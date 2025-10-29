## ADDED Requirements

### Requirement: Dense-caption SFT with KL anchoring (GKD)
- When `rlhf_type == "gkd"`, the system SHALL keep the dense-caption dataset pipeline unchanged (same Stage-2/3 templates, augmentation, packing, LoRA targeting).
- A frozen `teacher_model` (base Qwen3‑VL) SHALL be loaded with matching tokenizer/template.
- The trainer SHALL minimize `loss = sft_alpha * CE(student, labels) + beta * KL(teacher||student)` and run on multimodal batches.
- The teacher MUST be no-grad and never updated.

#### Scenario: Valid multimodal batch under GKD
- GIVEN a dataset sample with images + chat messages
- WHEN `rlhf_type=gkd` is enabled
- THEN the forward pass computes both CE and KL without altering inputs or template behavior

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

### Requirement: KL/CE telemetry
- Training logs SHALL expose `train/kl_loss`, `train/sft_loss`, and total `train/loss` every logging step.
- `logging.jsonl` SHALL include these fields; tensorboard curves SHALL be emitted when enabled.

#### Scenario: KL spikes detection
- GIVEN a run with `beta>0`
- WHEN reading `logging.jsonl`
- THEN entries contain finite KL values; NaN must trigger a clear warning and step identification

---

### Requirement: Backward compatibility
- With `rlhf_type` unset, behavior SHALL be identical to today’s SFT.
- Switching from SFT→GKD SHALL not require code changes; only config changes are allowed.

#### Scenario: Vanilla SFT config
- GIVEN `stage_3_vision_llm.yaml` without an `rlhf` block
- WHEN loaded through `src.sft`
- THEN the pipeline instantiates `SwiftSft`, no teacher model is loaded, and training arguments match the pre-change behavior

---

### Requirement: Smoke test
- A tiny dataset (≤32 samples) SHALL complete ≥1 epoch using GKD with `beta>0` and write checkpoints and `logging.jsonl`.
- Expected wall-clock increase (~2x fwd passes) SHALL be documented in the guide.

#### Scenario: Minimal run
- GIVEN teacher + student paths and a small JSONL
- WHEN running `--config stage_3_gkd.yaml`
- THEN training finishes, checkpoints are saved, and KL/CE are logged

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
