# Proposal: add-irrelevant-summary-source

## Why
- Summary SFT is overfitting to BBU scene priors; model hallucinates BBU content on irrelevant images.
- Need a lightweight "irrelevant" negative dataset that always yields summary `无关图片` to regularize hallucinations.
- Keep the existing summary template/prompt pipeline intact while adding a clean negative stream.

## What Changes
- Add an "irrelevant_summary" **fusion target entry** that reuses the existing `bbu` dataset wrapper while running in `mode: summary` with the `bbu_summary` template.
- This target placement is intentional: fusion target ratios scale by the dataset's own pool size, so `ratio: 1` means "use each irrelevant image once per epoch" without introducing a new ratio semantics.
- Note: fusion *source* ratios are computed against the total target quota (`round(source_ratio * total_target_quota)`), not the source pool size; using a target entry preserves the "ratio scales by its own dataset length" expectation.
- Provide a helper that converts `data/irrelevant_summary/images/*.jpeg` into JSONL records with `summary: "无关图片"`, width/height filled, and image paths relative to the JSONL.
- Keep records compatible with the canonical JSONL contract by emitting a single dummy full-frame `bbox_2d` object per image.
- Extend the existing summary fusion config to mix this target with a ratio knob (default `1`, user-tunable).
- Set `augmentation_enabled: false` and `curriculum_enabled: false` on this entry to keep negatives deterministic even if future runs enable augmentation globally.
- Use `val_jsonl: data/irrelevant_summary/train.jsonl` so validation can build without requiring a separate split for this tiny negative set.

## Scope / Non-goals
- Scope: summary-mode only; negative/irrelevant images; no real annotations required (dummy bbox only).
- Out of scope: new prompts, augmentation, Stage-A/B runtime changes, or changes to the global JSONL contract/validators.

## Risks
- Oversampling a tiny pool could skew training if `ratio > 1` (target upsampling); mitigate by defaulting to `ratio: 1` and keeping the knob explicit/user-tunable.
- Eval mixes all targets: if the training run builds eval with the fusion config, this target's `val_jsonl` will be included in the eval schedule (we set `val_jsonl=train_jsonl` for simplicity).

## Validation
- `openspec validate add-irrelevant-summary-source --strict`
- JSONL sanity: `python scripts/validate_dense_jsonl_contract.py --jsonl data/irrelevant_summary/train.jsonl --limit 0`
- Config lint/smoke: load the updated fusion config in a dry run (no training) once implemented.
