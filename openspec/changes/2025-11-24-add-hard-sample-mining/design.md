# Design: Hard-Sample Mining Stage

## Goals
- Collect per-sample **loss** (not token_acc) after augmentation and mine hard examples once training stabilizes.
- Reweight/duplicate mined samples in subsequent epochs without changing epoch length or dataloader shape.
- Keep augmentation + fusion scheduling compatible; avoid invasive trainer changes; remain safe under DDP/DeepSpeed ZeRO-2.

## High-Level Flow
1) **Dataset emits sample metadata**: `sample_id` (stable per logical record), `dataset`, `base_idx`, `epoch`, `aug_seed`.
2) **Trainer wrapper** computes per-example CE loss (masked) and hands `(sample_id, loss, step, epoch, dataset, base_idx)` to a shared tracker; rank0-only updates to stay DS-safe.
3) **Callback** monitors training/validation metrics; after a configurable trigger (epoch >= N and plateau or eval delta < eps), it
   - aggregates per-sample losses (EMA over batches),
   - selects hard set (top-K / percentile / threshold),
   - writes a `HardSamplePlan` (weights map) to the dataset.
4) **Dataset set_epoch()** rebuilds permutation/schedule using weights while keeping length fixed (weighted choice over existing slots).
5) **Augmentation** still runs per-sample; mined samples are not forced clean unless configured (`mine_clean=false` default).

## Data Structures
- `HardSampleMiningConfig`: enable flag, `start_epoch`, `patience`, `plateau_delta`, `top_k`, `percentile`, `loss_threshold`, `max_dup_factor`, `ema_decay`, `mine_clean`, `log_top_n`, `target_ratio` (e.g., 0.7 hard / 0.3 regular).
- `HardSampleTracker`: maintains running per-sample loss stats (`count`, `ema`, `last_loss`), keyed by `sample_id` with side-car map to `(dataset, base_idx)` for logging; updates occur on rank 0 only (all-reduce or gather optional but not required).
- `HardSamplePlan`: mapping `base_idx -> weight` per dataset (target-only in fusion) + metadata (`version`, `computed_at_epoch`).

## Integration Points
- **ms-swift compatibility check**: `swift/trainers/trainers.py::Trainer.compute_loss` delegates to HF then `_compute_acc`; it returns batch-mean only. Wrapping the resolved trainer to emit a per-example loss vector (for tracking) while keeping the mean for backprop preserves ZeRO-2 flows; keep `_patch_loss_function` untouched.
- **Config**: `custom.hard_sample_mining` parsed into structured config, default disabled.
- **sft.py**: instantiate tracker + callback when config enabled; pass plan to datasets (train only). Callback added after curriculum/fusion callbacks.
- **Dataset**:
  - `BaseCaptionDataset`: add `sample_id`, accept optional `hard_sample_plan`, use weighted `choices(k=len(base_records))` in `_rebuild_perm_for_epoch`.
  - `FusionCaptionDataset`: maintain per-dataset plans; adjust `_build_train_schedule` to sample from weighted distributions **only for the target pool** while keeping per-epoch counts; source pools unchanged.
- **Trainer Wrapper**: subclass resolved trainer to override `compute_loss` → compute per-example CE (shifted labels, mask `-100`), reduce mean for training but emit vector to tracker; pop metadata from inputs before forward; keep gradient/ZeRO2 flows identical.
- **Callback**: listens to `on_epoch_end/on_evaluate`; checks plateau vs. `train/loss` or eval metric; updates dataset plan; logs top-N sample IDs and weights.

## Edge Cases
- Gradient accumulation: per-sample loss computed on current microbatch; tracker stores mean per sample across repeats.
- DDP: tracker should reduce on main process only; gather sample_ids/losses from local batch and only rank0 updates aggregator (requires distributed guard using `get_dist_setting` or `Trainer.is_world_process_zero`).
- Memory: tracker uses dict of float stats; prune when sample never observed in last M epochs if needed (not planned initially).

## Telemetry
- Logs: `hsm/triggered`, `hsm/top_loss_mean`, `hsm/num_hard`, `hsm/weights/max`, `hsm/weights/min`.
- Optional dump: write `hard_samples_epoch_{n}.json` under output_dir containing sample_id, dataset, base_idx, loss_ema.

## Alternatives Considered
- Modifying ms-swift Trainer to emit per-sample loss directly: rejected to avoid upstream dependency edits.
- Changing dataset length per epoch: rejected to keep DataLoader/static sampler compatibility.
