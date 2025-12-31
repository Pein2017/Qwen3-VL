# Design: Irrelevant summary target for fusion

## Dataset semantics
- Records: one image per record, required `summary: 无关图片`, width/height read from the image, and a single dummy full-frame `bbox_2d` object to satisfy the canonical JSONL contract; image paths stored relative to the JSONL.
- Prompts: use the existing `bbu_summary` template to align output format (`无关图片`).
- Augmentation/curriculum: disabled to keep negatives deterministic.

## Integration
- Fusion config adds a new **target** entry named `irrelevant_summary` that reuses the existing `bbu` wrapper with `template: bbu_summary` and `mode: summary`.
- Default ratio is `1` (user-tunable) so the target quota is `round(len(irrelevant_pool) * 1)`, i.e., each irrelevant image appears once per epoch.
- This is intentionally a **target** (not a source): source ratios are computed against `total_target_quota`, while targets scale by their own pool size.
- Set `augmentation_enabled: false` and `curriculum_enabled: false` on this entry for forward safety even if global augmentation is later enabled.
- Set `val_jsonl` to the same path as `train_jsonl` (`data/irrelevant_summary/train.jsonl`) to satisfy target split requirements without introducing a separate tiny validation set.
- Helper script: scans `data/irrelevant_summary/images/*.jpeg` (and optionally `*.jpg`), writes `data/irrelevant_summary/train.jsonl`, keeping filenames and EXIF-aware dimensions; sorts paths deterministically and skips files that fail to load.
- Eval note: the fusion eval schedule concatenates *all* targets; with `val_jsonl=train_jsonl`, this target will appear in eval metrics unless the run config overrides eval behavior.

## Validation
- Spec validation via `openspec validate add-irrelevant-summary-source --strict`.
- Optional config dry-run: load fusion config in Python to confirm wrapper resolution and summary-mode encoding.
