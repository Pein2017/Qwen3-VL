## ADDED Requirements

### Requirement: Per-dataset telemetry without packing
- When `training.packing: false`, the SFT runner SHALL emit only `*_loss` and `*_token_acc` per dataset, using the dataset name defined in config/metadata (prefer `_fusion_source`, fall back to `dataset_name`); no separate packing/group key SHALL be required.
- The batching path SHALL stay on the standard padded collator (no bin-packing) so multi-sample batches (`per_device_train_batch_size>1`) use normal padding and keep dataloader shuffling/order unchanged.
- Training logs SHALL include per-dataset metrics for every dataset that appears in fused training batches (e.g., `train/bbu_loss`, `train/rru_token_acc`). Evaluation SHALL log only target datasets by default; source datasets MAY log if—and only if—their `val` path is provided (future-ready, currently `None`).
- The trainer SHALL reuse the grouped-metrics reducer with one segment per sample (sequence length from the padded batch) to aggregate per-dataset loss/accuracy across ranks, without enabling packing or packing-specific compilation.

#### Scenario: Training with padded batches and per-dataset metrics
- WHEN `training.packing: false`, `per_device_train_batch_size>1`, and fused batches contain samples from `bbu`, `rru`, `lvis`, `lang_chat`
- THEN the collator attaches per-sample dataset labels and single-sample lengths, the trainer logs `train/{dataset}_loss` and `train/{dataset}_token_acc` for each dataset during logging steps, and aggregate training metrics remain unchanged.

#### Scenario: Evaluation limited to target by default
- GIVEN only target datasets provide `val` JSONL paths
- WHEN evaluation runs with packing disabled
- THEN only target metrics (e.g., `eval/bbu_loss`, `eval/bbu_token_acc`) are logged; no source metrics appear.

#### Scenario: Evaluation with source val provided
- GIVEN a source dataset that now supplies a `val` path
- WHEN evaluation runs with packing disabled
- THEN per-source eval metrics (e.g., `eval/rru_loss`, `eval/rru_token_acc`) are logged alongside target metrics using the same reducer, without affecting aggregate eval metrics.
