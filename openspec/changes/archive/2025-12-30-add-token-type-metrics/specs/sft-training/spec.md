## ADDED Requirements

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
