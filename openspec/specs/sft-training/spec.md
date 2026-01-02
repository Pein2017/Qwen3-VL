# sft-training Specification

## Purpose
Specify Qwen3-VL SFT training behavior, including GKD integration, KD/CE telemetry, config plumbing, and stage-specific trainer expectations.
## Requirements
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
- Stage configs (e.g., `stage_3_gkd.yaml` overlays) SHALL enable the feature purely via YAML edits - no CLI flags or hard-coded overrides.

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

### Requirement: Optional token-type telemetry gate
- The system SHALL expose `custom.token_type_metrics` with fields `enabled` (bool, default false), `include` (list of dataset labels; default `['target', 'lvis']`), and `exclude` (list; default `['coig_lang_chat']`).
- Token-type metrics SHALL run only when `enabled` is true and `dataset_labels` intersect `include` minus `exclude`; otherwise the batch SHALL skip token-type processing and logging.
- Default behavior (enabled=false) MUST match current telemetry (no extra metrics, no new batch fields).

#### Scenario: Disabled config
- GIVEN a config without `custom.token_type_metrics.enabled: true`
- WHEN training or evaluation runs
- THEN no token-type preprocessing is executed and logs contain only the existing metrics.

#### Scenario: Included vs excluded datasets
- GIVEN `custom.token_type_metrics.enabled: true`, `include: ['target','lvis']`, `exclude: ['coig_lang_chat']`
- WHEN a mixed fusion batch contains samples labeled `target`, `lvis`, and `coig_lang_chat`
- THEN token-type metrics are computed/logged only for `target` and `lvis` samples, and skipped for `coig_lang_chat`.

---

### Requirement: Token typing and alignment
- The collator SHALL reconstruct the assistant payload text deterministically (same separators/ordering as JSONLinesBuilder), tokenize it with the active template tokenizer, and assign token types: 1=description text, 2=coordinate numbers, 3=format/structure; tokens masked by labels == -100 or padding SHALL be type 0/ignored.
- The collator SHALL align token types to supervised positions (`labels != -100`) after padding; padded/ignored positions MUST be filled with -1 and removed before model forward.
- On length mismatch between supervised tokens and computed token types, the system SHALL skip token-type metrics for that batch and emit a debug warning without failing training.

#### Scenario: Successful alignment
- GIVEN a sample containing `desc`, `bbox_2d`, and `line_points`
- WHEN the collator builds `token_types`
- THEN the number of type entries matches the count of `labels != -100` and includes at least one token of each of types 1, 2, and 3.

#### Scenario: Length mismatch fallback
- GIVEN malformed input that causes token count mismatch
- WHEN the collator detects the mismatch
- THEN it drops token-type metrics for that batch and logs a debug warning while continuing training.

---

### Requirement: Per-type accuracy and entropy metrics
- For each included dataset label, the trainer SHALL log per-type token accuracy (`desc_token_acc`, `coord_token_acc`, `format_token_acc`) and per-type entropy (`desc_entropy`, `coord_entropy`, `format_entropy`) in both train and eval modes using logits[:, :-1] vs labels[:, 1:] masked by `token_types`.
- Metrics SHALL only be emitted when token-type telemetry is enabled and the dataset label is included; excluded datasets SHALL not emit these keys.
- Metric naming SHALL stay stable across train/eval (`train/` vs `eval/` prefixes as provided by the existing custom_metrics sink).

#### Scenario: Train step with target sample
- GIVEN `token_type_metrics.enabled: true` and a batch whose `dataset_labels` include `target`
- WHEN a train logging step occurs
- THEN logs contain `target_desc_token_acc`, `target_coord_token_acc`, `target_format_token_acc`, and corresponding entropy metrics with finite values.

#### Scenario: Eval step with excluded dataset
- GIVEN `token_type_metrics.enabled: true` but a batch labeled `coig_lang_chat`
- WHEN evaluation runs
- THEN no token-type metrics are emitted for that batch.

### Requirement: Deterministic cross-rank grouped metric sync
Grouped per-dataset metrics SHALL synchronize keys deterministically across ranks and rely on a single reduction at logging time (via `MeanMetric.compute()`), avoiding per-step double reduction while still exposing groups seen only on nonzero ranks.

#### Scenario: Small group only on nonzero rank
- **WHEN** a fusion group (e.g., `lvis` or `lang_chat`) appears only on rank>0 within a logging window
- **THEN** rank0 logs still contain that group's loss/accuracy metrics with correct values aggregated once

#### Scenario: No double reduction vs single GPU
- **GIVEN** the same tiny fusion dataset is run on 1 GPU and on 2 GPUs
- **WHEN** reading per-group loss/accuracy after one logging interval
- **THEN** the multi-GPU values match the single-GPU run within floating tolerance, showing that state/count were reduced exactly once

#### Scenario: Deterministic key ordering across ranks
- **WHEN** different ranks observe metric keys in different insertion orders
- **THEN** the sync step gathers the union and instantiates missing metrics in sorted order so collective calls remain aligned and cannot bleed values across metric names

### Requirement: Summary prompt profiles SHALL separate training and inference roles.
When `custom.use_summary` is enabled, the system SHALL resolve a summary prompt profile. The default profile for training SHALL be `summary_train_min`, and it MUST include **only** format rules and task criterion (including evidence-only / no-hallucination constraints); it MUST NOT include domain knowledge, mission rules, or dataset-specific priors.

#### Scenario: Default summary training uses minimal profile
- **GIVEN** a training config with `custom.use_summary: true` and no prompt profile override
- **WHEN** prompts are resolved for summary training
- **THEN** the system selects `summary_train_min`
- **AND THEN** the resulting system prompt contains format + task criterion only (including evidence-only / no-hallucination constraints), with no BBU/RRU domain rules or mission-specific priors

#### Scenario: Runtime profile is explicit and opt-in
- **GIVEN** a config that sets `prompts.profile: summary_runtime`
- **WHEN** prompts are resolved
- **THEN** the system uses the runtime profile and allows domain knowledge injection (if a domain is provided)

---

### Requirement: Domain knowledge packs SHALL be defined as Python dataclasses and excluded from training prompts.
Domain knowledge (BBU/RRU schema hints, priors, and restrictions) SHALL be defined in Python dataclasses and composed into prompts only when the runtime profile is selected. Training profiles MUST ignore domain packs entirely.

#### Scenario: Training profile ignores domain packs
- **GIVEN** domain packs defined for BBU and RRU
- **WHEN** a training run resolves `summary_train_min`
- **THEN** the system prompt excludes domain pack content even if a domain is configured

#### Scenario: Runtime profile includes domain pack
- **GIVEN** `prompts.profile: summary_runtime` and `prompts.domain: rru`
- **WHEN** the summary system prompt is built
- **THEN** the prompt includes the RRU domain pack content and excludes BBU-only rules

---

### Requirement: Prompt profile selection SHALL be configurable and validated.
Prompt profile and domain selection SHALL be configured via the `prompts` section. `prompts.system` or `prompts.user` overrides MUST remain authoritative. If a runtime profile is selected and the domain is missing or unknown, the system MUST fail fast with an actionable error.

#### Scenario: Unknown domain is rejected
- **GIVEN** `prompts.profile: summary_runtime` and `prompts.domain: unknown`
- **WHEN** the loader resolves prompts
- **THEN** it raises a validation error describing the allowed domains

#### Scenario: Explicit prompt override bypasses profiles
- **GIVEN** `prompts.system` or `prompts.user` is set in the config
- **WHEN** prompts are resolved
- **THEN** profile composition is bypassed and the provided override is used verbatim

### Requirement: Padding-only batching and telemetry
- The SFT runner SHALL reject any config that sets `training.packing` to true or supplies packing-specific knobs (`packing_group_key`, cached-length settings), returning a clear error that packing is removed and padding is the only supported batching mode.
- The default batching path SHALL use standard padding for `per_device_train_batch_size>1`, with no bin-packing or length-cache prepass executed in training or evaluation.
- Per-dataset telemetry SHALL remain available in this padding-only mode by attaching dataset labels from sample metadata and emitting one segment per sample to the grouped-metrics reducer.
- Evaluation metrics SHALL log only datasets that provide `val` splits; source-only datasets remain absent unless their `val` path is configured.
- The legacy packing implementation SHALL be removed from runtime import paths and stored only under `archive/packing/` for future reference.

#### Scenario: Config tries to enable packing
- WHEN a user launches training with `training.packing: true` (or any packing knobs)
- THEN initialization fails fast with an actionable error explaining packing is removed and to delete the packing settings.

#### Scenario: Normal padded training
- WHEN packing keys are absent and `per_device_train_batch_size>1`
- THEN the dataloader uses the padded collator, training proceeds, and aggregate metrics remain unchanged.

#### Scenario: Per-dataset metrics with padded batches
- WHEN padded batches contain samples from multiple datasets (e.g., `bbu`, `rru`, `lvis`, `lang_chat`)
- THEN the collator attaches per-sample dataset labels and single-sample lengths, the trainer logs `train/{dataset}_loss` and `train/{dataset}_token_acc` for each dataset during logging steps, and aggregate training metrics remain unchanged.

#### Scenario: Evaluation limited to datasets with val splits
- GIVEN only target datasets provide `val` JSONL paths
- WHEN evaluation runs with padding-only batching
- THEN only those datasets produce eval metrics (e.g., `eval/bbu_loss`, `eval/bbu_token_acc`); sources without `val` remain absent.

#### Scenario: Stray packing metadata
- WHEN a config includes `custom.packing_group_key` or `custom.cached_lengths`
- THEN validation fails with guidance to remove packing metadata because the feature is no longer supported.

#### Scenario: Packing code archived
- WHEN code attempts to import packing modules from the main package
- THEN the import fails; the archived implementation lives under `archive/packing/` and is not on the runtime path.

### Requirement: Geometry JSON spacing stability
The conversation builder SHALL serialize assistant JSON with space-separated separators to preserve tokenizer distribution for geometry arrays.

#### Scenario: Geometry arrays retain spaces
- **WHEN** the JSONLinesBuilder renders assistant JSON for geometry payloads
- **THEN** JSON separators use `", "` and `": "` (spaces preserved)
- **AND** geometry arrays (bbox_2d/poly/line) are serialized with spaces between numeric elements

