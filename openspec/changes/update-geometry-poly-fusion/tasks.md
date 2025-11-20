- [ ] Update `openspec/specs/data-augmentation/spec.md` via this change to replace `quad` with `poly` where appropriate and clarify that `poly` is the generic polygon type (4-point and N-point).
- [ ] Ensure all geometry helpers and augmentation paths in `src/` consistently use `bbox_2d` / `poly` / `line` and enforce the single-geometry-field invariant (verify no remaining `quad` references in code).
- [ ] Align `/data/public_data` converters (e.g., LVIS/COCO) and validators to emit `poly` + `poly_points` instead of `quad`, with a configurable max-point policy to fall back to `bbox_2d` for very complex polygons.
- [ ] Introduce an offline fusion builder that reads a small YAML/JSON config describing target and auxiliary datasets (paths, weights, augmentation flags) and produces a fused training JSONL with `metadata.dataset` provenance.
- [ ] Add a source-aware augmentation preprocessor that toggles augmentation per dataset source (e.g., apply augmentations only to BBU records) without changing `DenseCaptionDataset`.
- [ ] Wire training configs to point `custom.train_jsonl` at the fused JSONL and `custom.val_jsonl` at the BBU target val JSONL by default, keeping evaluation scoped to the target domain.
- [ ] Update relevant docs (`docs/DATA_AND_DATASETS.md`, `docs/DATA_AUGMENTATION.md`, `/data/public_data/README.md`/`POLYGON_SUPPORT.md`) to describe the `poly` geometry contract and the multi-dataset fusion pipeline.

