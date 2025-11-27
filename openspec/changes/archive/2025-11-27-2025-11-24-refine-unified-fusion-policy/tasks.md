## Tasks

- [x] Review existing fusion loaders (MultiSourceFusionDataset vs UnifiedFusionDataset) and confirm current gaps: prompts, augmentation gating, object caps, epoch resampling, smart-resize guard.
- [x] Update fusion config/schema docs to encode per-source policies (prompt priority domain→dataset override, augmentation/curriculum flags, max_objects_per_image, optional source val_jsonl, seed).
- [x] Implement unified fusion loader parity: single template, prompt priority resolution, per-source augmentation/curriculum gating, object-cap preprocessor, deterministic per-epoch resampling/shuffle; raise on empty source pool.
- [x] Remove smart-resize guard from fusion path (document expectation that inputs are pre-scaled; resizing only via augmentation ops).
- [x] Add telemetry/validation: prompt coverage per source, augmentation-on/off per source, cap hit counts; add smoke tests or scripts to assert policies.
- [x] Remove the legacy MultiSourceFusionDataset path entirely, enforcing unified-only fusion and documenting migration.
