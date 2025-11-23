## Proposal: Refine Unified Fusion Policy

### Problem
- MultiSourceFusionDataset relied on template cloning and per-dataset wrappers, causing OOM/unmasking issues and duplicating policy logic (augmentation, prompts, caps).
- UnifiedFusionDataset fixed OOM by using a single template, but it lost per-source controls: source prompts are not applied, sources can receive augmentation, object caps are skipped, and sources are sampled only once at init (no per-epoch refresh). Smart-resize guards from `add-detection-smart-resize` are not enforced online.

### What this change does
- Spec a unified fusion capability that keeps a single template but restores per-source policies: prompt priority (default < domain < dataset-specific), per-source augmentation/curriculum gating (sources clean by default), object caps; no geometry fallback needed.
- Define deterministic per-epoch resampling/shuffling for sources (ratio-based with replacement) while consuming the full target set each epoch; raise if a source pool is empty.
- Allow optional source eval splits while keeping target val as default.
- Require telemetry and fail-fast validation for prompt coverage, augmentation-on/off per source, and object-cap application.
- Retire the legacy multi-dataset path; unified fusion is the only supported runtime path.

### Motivations / benefits
- Eliminates template cloning risk while preserving domain-specific behavior.
- Reduces OOD risk from augmentation leaking into auxiliary sources and from missing object caps/resize guards.
- Aligns online fusion with prior specs: `update-geometry-poly-fusion` (poly contract, per-source wrappers) and `add-detection-smart-resize` (shared resize guard).
