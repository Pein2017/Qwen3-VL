# Documentation & Pipeline Changelog

Status: Active
Scope: Recent documentation and pipeline highlights for Qwen3-VL.
Owners: Documentation
Last updated: 2026-01-02
Related: [overview/ARCHITECTURE.md](ARCHITECTURE.md), [data/DATA_JSONL_CONTRACT.md](../data/DATA_JSONL_CONTRACT.md), [data/DATA_AUGMENTATION.md](../data/DATA_AUGMENTATION.md)

## Recent Updates

### v1.1.3 - RRU Support & Canonical Polygons (Nov 2025)
- Unified converter now handles RRU raw annotations: taxonomy additions (`ground_screw`, 尾纤/接地线标签与套管保护, 站点距离=数字), group membership encoded in `desc` via `组=<id>` (no `groups` field). Summaries are JSON strings with per-category stats.
- Polygon vertices are canonicalized offline (clockwise, top-most then left-most first) and `vis_tools` mirrors the ordering to avoid self-crossing during visualization.

### v1.1.2 - Config & Telemetry Contracts (Oct 2025)
- YAML loader now builds frozen dataclasses (`TrainingConfig`, `CustomConfig`, `SaveDelayConfig`, `VisualKDConfig`) with early validation and deterministic merging.
- Datasets adopt shared contracts for records/geometry; augmentation telemetry is a typed dataclass surfaced to preprocessors and debug logging.
- Stage-A CLI wraps runtime flags in `StageAConfig`, catching invalid missions/paths before inference launches.

### v1.1.1 - Quad Truncation Refinement (Oct 2025)
- Fixed rotate+crop polygon handling: rotated polygons now maintain rotation after crop.
- Added polygon simplification to preserve true polygon corners.
- Perfect boundary truncation without spurious AABB conversion.

### v1.1.0 - Smart Cropping with Label Filtering (Oct 2025)
- **RandomCrop** operator with automatic label filtering and geometry truncation.
- Perfect visual-label alignment for dense detection captioning.
- Completeness field tracking: `可见性=完整` ↔ `可见性=部分` updates.
- Quad rotation fix + redundancy cleanup (removed CenterCrop, Equalize).

---

## Major Change: Geometry Schema Overhaul (Nov 2025)

- The pipeline now publishes `poly` geometry entries everywhere (replacing the previous 4-point geometry key). Internally we still emit 4-point polygons today, but the schema and prompts are ready to hold arbitrary vertex counts going forward.
- All documentation, builders, and augmentation ops now expect `poly` (even-length list ≥8 values / ≥4 points; current runtime validation) as one of the three canonical geometry keys (`bbox_2d`, `poly`, `line`).
- This change affects data conversion, dataset builders, augmentation telemetry, Stage-A/B workflows, and training prompts. Regenerate derived artifacts and re-validate dataset probes if conversion scripts are re-run.
