- [x] Remove `DynamicPairDataset` and flatten `DenseCaptionDataset` to single-record behavior.
- [x] Delete `images_per_user_turn` from schema, config loaders, and configs; guard validation with explicit error.
- [x] Rewrite `JSONLinesBuilder` (dense + summary modes) to emit minimal object hierarchies without `图片_{n}` wrappers.
- [x] Update prompts/templates and docs (`docs/training/REFERENCE.md`, `docs/DATA_AND_DATASETS.md`, `src/README.md`) to reflect the new structure.
- [x] Refresh unit/integration tests and demo/vis tooling for the minimal schema.
- [ ] Run smoke SFT training and targeted eval set to confirm regressions resolve.

