# Tasks: update-desc-summary-contract

- [x] Audit existing desc parsing and augmentation completeness updates for '/' dependence; list required code touch points.
- [x] Implement new key=value desc builder in data_conversion (BBU + RRU), including OCR rules and negative-precedence conflict resolution.
- [x] Remove need-review (需复核) rewrite path and ensure remarks remain BBU-only.
- [x] Drop occlusion (遮挡) judgments from desc and summary outputs.
- [x] Update summary_builder to emit JSON string summaries for BBU/RRU with per-category stats, group breakdowns, and standard separators.
- [x] Update sanitizers to keep '-' and '/' in OCR while protecting commas/equals/pipes.
- [x] Update JSONLinesBuilder ref extraction + class-whitelist parsing to read 类别= and keep geometry JSON spacing.
- [x] Update Stage-A/B prompt text and summary parsing (Stage-A inference/postprocess, Stage-B prompts/reflection) for JSON summaries with legacy fallback.
- [x] Update augmentation completeness updates to handle 可见性=显示完整 → 可见性=只显示部分.
- [x] Update docs/readmes: DATA_JSONL_CONTRACT, DATA_PREPROCESSING_PIPELINE, DATA_AND_DATASETS, DATA_AUGMENTATION, Stage-B guidance, data_conversion README.
- [x] Update tests to reflect JSON summary strings and spacing (builder + summary normalizer).
- [x] Expand OpenSpec deltas with full requirement scenarios (detection-preprocessor, data-augmentation, sft-training) and run `openspec validate update-desc-summary-contract --strict`.
- [x] Normalize 站点距离 to integer token in converter sanitization.
- [x] Allow irrelevant-image summaries to remain `无关图片` (update spec/doc contract).
- [x] Re-run `openspec validate update-desc-summary-contract --strict` after the spec/doc update.
- [x] Remove “重复键” from summary schema, prompts, tests, and converter logic.
