# Proposal: Dynamic Hard-Sample Mining for Qwen3-VL Detection SFT

## Why
- Dense detection SFT plateaus after a few epochs with long-tail errors (small/rare objects).
- Current training samples uniformly per epoch; difficult cases are not emphasized once loss stabilizes.
- Need a built-in, YAML-driven hard-sample mining stage that reweights/duplicates hard examples without breaking augmentation or fusion scheduling.

## What
- Collect per-sample **token_acc** during training（含增广），按 EMA 计算难度（acc 低即难）。
- 前 70% 训练仅记录；后 30% 每轮选目标集最难 30% 作为 hard_pool，重建下一轮调度：目标长度不变 = 30% hard（可重复）+ 70% 目标全量有放回；源样本追加为目标长度的 8%，不挖掘不重权。
- 仅目标数据集参与挖掘；源配比固定。
- Expose knobs under `custom.hard_sample_mining` in YAML; add a Trainer callback + dataset external schedule to orchestrate tracking and resampling.

## Scope
- Training entry `src/sft.py` (wiring, config parsing).
- Datasets: `BaseCaptionDataset`, `FusionCaptionDataset` schedule/permutation and metadata for sample IDs.
- Callbacks: new `HardSampleDynamicCallback` (token_acc tracking, epoch-end selection, schedule update) that aggregates on rank0 to stay DDP/DeepSpeed safe.
- Config schema + docs (`docs/training/TRAINING_PLAYBOOK.md`, `docs/UNIFIED_FUSION_DATASET.md`), optional design note; mining applies to fusion target only.

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
