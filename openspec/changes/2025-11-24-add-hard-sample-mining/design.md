# Design: Hard-Sample Mining Stage

## Goals
- Collect per-sample **loss** (not token_acc) after augmentation and mine hard examples every epoch.
- Reweight/duplicate mined samples in subsequent epochs, optionally downsizing the target epoch length.
- Keep augmentation + fusion scheduling compatible; avoid invasive trainer changes; remain safe under DDP/DeepSpeed ZeRO-2.

## High-Level Flow
1) **Dataset emits sample metadata**: `sample_id` (stable per logical record), `dataset`, `base_idx`, `epoch`, `aug_seed`.
2) **Trainer wrapper** computes per-example CE loss (masked) and hands `(sample_id, loss, step, epoch, dataset, base_idx)` to a shared tracker; rank0-only updates to stay DS-safe.
3) **Callback** runs every epoch end (no plateau trigger):
   - aggregates per-sample losses (EMA over batches; mean when <3 obs),
   - selects fixed hard set (`hard_sample_size`, e.g., 500) and fixed regular set (`regular_sample_size`, e.g., 150),
   - writes a `HardSamplePlan` (weights map + `target_epoch_size`) to the dataset.
4) **Dataset set_epoch()** rebuilds permutation/schedule using weights; length can stay fixed or use a downsized target epoch length when configured for the mining stage.
5) **Augmentation** still runs per-sample; mined samples are not forced clean unless configured (`mine_clean=false` default).

## Data Structures
- `HardSampleMiningConfig`: enable flag, `start_epoch`, `hard_sample_size`, `regular_sample_size`, `ema_decay`, `mine_clean`, optional `target_epoch_size` (e.g., 650 = 500 hard + 150 regular) for mining.
- `HardSampleTracker`: maintains running per-sample loss stats (`count`, `ema`, `last_loss`), keyed by `sample_id` with side-car map to `(dataset, base_idx)` for logging; updates occur on rank 0 only (all-reduce or gather optional but not required).
- `HardSamplePlan`: mapping `base_idx -> weight` per dataset (target-only in fusion) + metadata (`version`, `computed_at_epoch`).

## Integration Points
- **ms-swift compatibility check**: `swift/trainers/trainers.py::Trainer.compute_loss` delegates to HF then `_compute_acc`; it returns batch-mean only. Wrapping the resolved trainer to emit a per-example loss vector (for tracking) while keeping the mean for backprop preserves ZeRO-2 flows; keep `_patch_loss_function` untouched.
- **Config**: `custom.hard_sample_mining` parsed into structured config, default disabled.
- **sft.py**: instantiate tracker + callback when config enabled; pass plan to datasets (train only). Callback added after curriculum/fusion callbacks.
- **Dataset**:
  - `BaseCaptionDataset`: add `sample_id`, accept optional `hard_sample_plan`, use weighted `choices(k=len(base_records))` in `_rebuild_perm_for_epoch`, or a downsized length when `target_epoch_size` is set for mining.
  - `FusionCaptionDataset`: maintain per-dataset plans; adjust `_build_train_schedule` to sample from weighted distributions **only for the target pool**; support optional downsized target length while keeping source pools/quotas unchanged.
- **Trainer Wrapper**: subclass resolved trainer to override `compute_loss` → compute per-example CE (shifted labels, mask `-100`), reduce mean for training but emit vector to tracker; pop metadata from inputs before forward; keep gradient/ZeRO2 flows identical.
- **Callback**: listens to `on_epoch_end`; updates dataset plan; logs hard counts and epoch size.

## Edge Cases
- Gradient accumulation: per-sample loss computed on current microbatch; tracker stores mean per sample across repeats (mean when <3 obs, EMA otherwise if configured).
- DDP: tracker should reduce on main process only; gather sample_ids/losses from local batch and only rank0 updates aggregator (requires distributed guard using `get_dist_setting` or `Trainer.is_world_process_zero`).
- Memory: tracker uses dict of float stats; prune when sample never observed in last M epochs if needed (not planned initially).

## Telemetry
- Logs: `hsm/triggered`, `hsm/top_loss_mean`, `hsm/num_hard`, `hsm/weights/max`, `hsm/weights/min`.
- No JSON dump required (per request).

## Alternatives Considered
- Modifying ms-swift Trainer to emit per-sample loss directly: rejected to avoid upstream dependency edits.
- Changing dataset length per epoch: rejected to keep DataLoader/static sampler compatibility.
