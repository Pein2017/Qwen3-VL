## Design Notes

### Architecture direction
- Keep the single-template UnifiedFusionDataset to avoid cloning risk; attach per-source policy at the record level (metadata `_fusion_source`) and select prompts dynamically.
- Build a per-epoch schedule: full target coverage each epoch; for each source, sample `round(ratio * N_target)` with replacement per epoch; shuffle with deterministic seed; support manual seed for debugging.
- Per-source policy bundle:
  - Prompts: user + system resolved by priority default < domain < dataset-specific (target can override via config).
  - Augmentation/curriculum: enabled only when the source’s policy allows; sources default to clean; target inherits augmentation/curriculum.
  - Object cap: optional per source (default on for sources; optional for target); no geometry fallback required (assumed pre-filtered JSONL).
  - No smart-resize guard: inputs are pre-scaled; resizing only via explicit augmentation ops.
- Telemetry: per-sample debug info includes dataset name, prompt id, augmentation flag, object-cap applied; aggregate counters per epoch.

### Compatibility and migration
- Maintain config compatibility with existing fusion YAML; add optional per-source policy fields (prompt override, augmentation_enabled, curriculum_enabled, max_objects_per_image, optional val_jsonl, seed), keeping existing template/ratio keys.
- Retire the MultiSourceFusionDataset path; unified fusion is the sole runtime loader for fusion configs.

### Related specs
- Aligns with `update-geometry-poly-fusion` (canonical poly, per-source wrappers) and `add-detection-smart-resize` (shared resize guard). No changes to augmentation semantics; only policy routing.
