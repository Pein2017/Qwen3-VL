# Tasks: add-summary-grpo-post-training

- [x] Draft `summary-grpo-post-training` spec with format contract, reward wiring, header matching, JSON-equivalence rules, rollout settings, and launch path via `scripts/train.sh` + `rlhf` block.
- [x] Add deprecation delta for `grpo-integration` (mark as deprecated and superseded by summary GRPO spec).
- [x] Document per-epoch irrelevant prompt alternation (~50/50), fusion-based dataset toggles, and single-line irrelevant output in the new spec.
- [x] Implement GRPO reward functions (format/header/parse/content) and register them for ms-swift `reward_funcs`.
- [x] Ensure reward kwargs include `metadata._fusion_domain_token` and ground-truth summary (`metadata.summary_ref`) for reward functions.
- [x] Ensure irrelevant summary samples suppress `assistant_prefix_format` so labels are single-line `无关图片`.
- [x] Add a GRPO summary config example under `configs/` with `rlhf` block and fusion-based dataset toggle.
- [x] Add a GRPO dry-run/smoke recipe (minimal dataset + 1–2 steps) and expected success criteria.
- [x] Add deprecation header to `openspec/specs/grpo-integration/spec.md` once the new spec is approved.
- [x] Validate with `openspec validate add-summary-grpo-post-training --strict`.
