# Proposal: Add Hard-Sample Mining to Qwen3-VL Detection SFT

## Why
- Dense detection SFT plateaus after a few epochs with long-tail errors (small/rare objects).
- Current training samples uniformly per epoch; difficult cases are not emphasized once loss stabilizes.
- Need a built-in, YAML-driven hard-sample mining stage that re-weights or duplicates hard examples without breaking augmentation or fusion scheduling.

## What
- Collect per-sample **losses** during training (with augmentation on) and identify hard samples after convergence/plateau.
- Persist hard-sample IDs across epochs; resample/duplicate them in subsequent epochs while keeping dataset length stable and augmentation active.
- Provide configurable trigger (epoch N or metric plateau) and selection strategy (top-K, percentile, threshold).
- Expose knobs under `custom.hard_sample_mining` in YAML; add a Trainer callback to orchestrate tracking + dataset reweighting.
- Update fusion dataset schedule/_perm to honor a per-epoch hard-sample plan.

## Scope
- Training entry `src/sft.py` (wiring, config parsing).
- Datasets: `BaseCaptionDataset`, `FusionCaptionDataset` schedule/permutation and metadata for sample IDs.
- Callbacks: new `HardSampleMiningCallback` (loss tracking, trigger, schedule update) that aggregates on rank0 to stay DDP/DeepSpeed safe.
- Config schema + docs (`docs/TRAINING_PLAYBOOK.md`, `docs/UNIFIED_FUSION_DATASET.md`), optional design note; mining applies to fusion target only.

## Non-Goals
- Changing model loss functions or adding new heads.
- Offline mining or external re-labeling.
- Inference-time sampling.

## Risks / Mitigations
- **DataLoader length changes**: keep epoch length constant; use weighted/resampled perms instead of extending dataset size.
- **Augmentation noise**: aggregate losses by logical sample ID across augmented variants; optionally keep running EMA to reduce variance.
- **Trainer compatibility**: isolate per-sample loss computation in a wrapper Trainer subclass and strip metadata before model forward.

## Success Criteria
- Config flag to enable/disable hard-sample mining without code changes.
- Logs show mining stage triggered and top-K list size.
- Dataset plan reflects increased frequency for mined samples while epoch length unchanged.
- No regression to existing training runs when feature disabled.
