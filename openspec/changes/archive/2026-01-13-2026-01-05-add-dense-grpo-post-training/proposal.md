# Proposal: Mixed-mode dense GRPO post-training (BBU/RRU detection) with reward shaping

## Why
- Current GRPO post-training is summary-only (`configs/train/grpo/summary_2048.yaml`) and optimizes summary stability, not dense detection performance.
- Dense captioning already exists as a training mode (BBU/RRU targets with `bbox_2d`/`poly`/`line` + `desc`), but there is no GRPO reward surface for improving dense localization/attributes.
- Detection quality requires objective geometry-aware rewards:
  - **Localization-first** (geometry overlap / recall+precision)
  - **Category-second** (类别 correctness on matched objects)
  - **Attribute correctness** (key=value fields), with business-driven weighting (e.g., `可见性` is noisy and should be down-weighted).
- Mixed-mode GRPO (dense targets + summary regularization sources) is needed to preserve summary behavior while tuning dense detection, without introducing LVIS/chat sources in this post-training phase.

## What Changes
- Add a **new dense GRPO post-training capability** for BBU/RRU dense captioning, implemented as:
  - A new dense reward namespace (`dense.*`) for:
    - strict two-line format + header validation (`<TASK=DETECTION>`)
    - strict JSON/schema validation (no duplicate keys; one geometry per object)
    - geometry matching rewards using **exact poly and line scoring** (no bbox-only fallback)
    - attribute rewards using **exact string match** on `desc` key=value fields with business weighting
    - category-aware metrics as a secondary objective
  - A GRPO fusion dataset mix that samples:
    - dense targets: `bbu_dense` ratio `1.0`, `rru_dense` ratio `1.0`
    - summary sources: `bbu_summary` ratio `0.5`, `rru_summary` ratio `0.5`
    - irrelevant summary: `irrelevant_summary` ratio `0.2`
    - explicitly **excludes** `lvis` and `lang_chat` for this post-training stage
- Keep existing summary GRPO training compatible and unchanged by default.
- Add offline evaluation extensions to measure attribute/OCR/site-distance correctness on dense dumps, aligned with the reward definitions.

## Scope
- New dense reward implementations and registry wiring under `src/rlhf/grpo/rewards/`.
- New GRPO fusion config variants under `configs/fusion/` for dense+summary mixing, excluding LVIS/chat.
- New GRPO training preset(s) under `configs/train/grpo/` and a new RLHF overlay under `configs/components/rlhf/`.
- Offline evaluation extension under `vis_tools/` for attribute/OCR/site-distance metrics on `gt_vs_pred.jsonl` (norm1000).

## Non-goals
- Changing the JSONL data contract (`images/objects/summary/width/height`).
- Introducing new geometry primitives beyond `bbox_2d`, `poly`, and `line`.
- Adding LVIS/chat sources to GRPO post-training (explicitly excluded for this stage).
- Replacing the existing summary GRPO presets; summary-only workflows remain supported.
- Adding new CHORD-mixing requirements specific to dense GRPO; any CHORD behavior remains governed by the existing GRPO/summary CHORD integration and is not modified by this change.

## Impact / Breaking changes
- Additive change for configs and reward functions.
- A compatibility fix is expected for reward-call conventions: custom ORMs SHALL accept ms-swift GRPO calling conventions (`reward_func(completions, **kwargs)`).

## Success criteria
- A mixed-mode GRPO run initializes and trains without reward exceptions:
  - dense rewards apply only to dense samples
  - summary rewards apply only to summary/irrelevant samples
- Offline evaluation shows improvements on dense targets:
  - missing-object rate decreases (primary; recall-biased objective)
  - localization quality metrics (e.g., mean-Fβ/mean-F1) increase (primary)
  - category mean-F1 does not regress materially (secondary)
  - attribute weighted match improves (excluding noisy `可见性` down-weight)
  - RRU `站点距离` exact-match accuracy improves
  - OCR/备注 match rates increase when present, without penalizing mismatches

## Risks
- Mixed-mode ratios (0.5/0.5/0.2) increase non-dense samples per epoch; dense localization reward weights must dominate to avoid over-regularizing toward summary output.
- Exact poly/line scoring is computationally heavier than bbox-only; reward functions must avoid redundant parsing/matching work across multiple rewards.
- GT annotations can be incomplete; overly strong false-positive penalties may reduce recall and hide improvements as “hallucinations”.
- Attribute matching assumes stable, exact strings; drift in formatting (spaces, punctuation) can reduce reward signal if not normalized consistently.

## Rollout plan (high-level)
1. Define dense GRPO capability spec and summary reward compatibility deltas (this change).
2. Implement dense reward module + registry wiring and ensure mode-gated execution.
3. Add GRPO configs (fusion + training presets) that exclude LVIS/chat and encode the requested ratios.
4. Extend offline evaluation to include attribute/OCR/site-distance metrics aligned with reward definitions.
5. Add targeted unit tests for parsing, matching, and attribute scoring.
6. Run a small GRPO smoke config (few steps) to confirm end-to-end wiring and reward logging.
