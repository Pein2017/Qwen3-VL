# Tasks: add-multi-object-line-copy-paste

- [ ] Review existing augmentation specs and tests for PatchOps, especially `small_object_zoom_paste` and `RandomCrop`.
- [ ] Implement shared occupancy-grid and patch-placement helpers in `src/datasets/augmentation/ops.py`.
- [ ] Extend `SmallObjectZoomPaste` with optional background-aware placement, multi-copy mode, and a small-object patch bank while preserving current default behavior.
- [ ] Add `ObjectClusterCopyPaste` PatchOp with cluster construction, patch extraction, and background-aware placement, reusing shared helpers and (optionally) the patch bank.
- [ ] Add `LineSegmentCopyPaste` PatchOp for line/segment copy-paste with mild scale+translate transforms and IoU/coverage safeguards, reusing shared helpers and the patch bank.
- [ ] Wire new ops into at least one training config under `configs/` (e.g., 1024 `sft_base.yaml`) with conservative default probabilities and caps.
- [ ] Update `docs/data/DATA_AUGMENTATION.md` with descriptions, diagrams, and configuration examples for the new copy-paste capabilities.
- [ ] Add or extend unit tests under `tests/augmentation/` to cover occupancy behavior, bank determinism, and geometry correctness for the new ops.
- [ ] Validate visually via `vis_tools/vis_augment_compare.py` on representative samples to confirm recognizability of small objects and line segments after augmentation.
- [ ] Run `openspec validate add-multi-object-line-copy-paste --strict` and ensure all checks pass.

