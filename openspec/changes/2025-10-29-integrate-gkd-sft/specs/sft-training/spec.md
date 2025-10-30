## ADDED Requirements

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

### Requirement: KL/CE telemetry
- Training logs SHALL expose `train/kl_loss`, `train/sft_loss`, total `train/loss`, and `train/token_accuracy` every logging step, and emit matching `eval/*` metrics without duplicating prefixes (e.g., no `train/eval/*`).
- `logging.jsonl` SHALL include these fields; tensorboard curves SHALL be emitted when enabled.
- Evaluation mode SHALL omit teacher forwards and only log CE-derived metrics.
- Teacher forwards MUST run without enabling `torch.autocast` when DeepSpeed is active unless the DeepSpeed config explicitly turns on autocast; otherwise dtype mismatches SHALL be avoided by casting teacher logits to the student dtype.

#### Scenario: KL spikes detection
- GIVEN a run with `beta>0`
- WHEN reading `logging.jsonl`
- THEN entries contain finite KL values; NaN must trigger a clear warning and step identification

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
