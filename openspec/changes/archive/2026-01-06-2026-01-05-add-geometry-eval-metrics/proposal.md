# Proposal: Geometry- and category-aware evaluation for dense captioning (bbox/poly/line)

## Why
- `eval_loss` is a weak proxy for dense captioning / detection quality. The training objective can improve while geometry placement, object recall, or attribute correctness regresses.
- Qwen3‑VL’s target domains rely on three geometry primitives (`bbox_2d`, `poly`, `line`) and structured descriptions (`desc`). Objective evaluation must reflect both geometry overlap and semantic correctness.
- Existing dump-based evaluation provides geometry-only, same-type matching, which does not reflect business usage where `bbox_2d` and `poly` are both valid region representations and where category correctness matters.

## What
- Introduce an **offline evaluation protocol** over `gt_vs_pred.jsonl` (norm1000) that reports:
  - **Region overlap** via filled-shape IoU, allowing `bbox_2d` ↔ `poly` cross-matching (“area intersection is the ruler”).
  - **Line overlap** via **TubeIoU**: rasterized mask IoU between buffered polylines at a fixed norm1000 tolerance.
  - **Three evaluation modes** (same underlying overlap + matching):
    1) Localization-only (ignore `desc`)
    2) Phase-aware end-to-end (coarse “phase/head” label constraint)
    3) Category-aware end-to-end (fine category constraint)
  - **COCO-like threshold sweep** (0.50:0.95 step 0.05) with `Precision/Recall/F1` and mean-F1.
  - Optional **attribute/OCR scoring** on matched pairs (key=value and legacy slash formats normalized).
- Produce both:
  - a **machine-readable** JSON report (for regression gating / dashboards), and
  - a **human-readable** console summary (for quick iteration).

## Scope
- `vis_tools/` evaluation utilities and CLI surface (dump reader + metric aggregation).
- Minimal domain-aware `desc` parsing/normalization needed for phase/category (and optional attributes).
- Documentation updates to record the evaluation protocol and recommended thresholds/tolerances.

## Non-goals
- Changing training prompts, JSONL schema, or the Stage‑A/B runtime contracts.
- Introducing new annotation types beyond `bbox_2d`, `poly` (convex quads), and `line`.
- Replacing visualization workflows; evaluation only consumes the existing `gt_vs_pred.jsonl` dumps.

## Impact / Breaking changes
- No breaking change required. This is an additive evaluation workflow.
- Optional: later follow-ups may add metric gates to Stage-A validation scripts, but that is out of scope for this change proposal.

## Success criteria
- Running the evaluator on existing `gt_vs_pred.jsonl` produces stable, deterministic metrics.
- Metrics separate regressions across geometry types (bbox/poly/line) and across domains (BBU/RRU).
- Cross-type region matching (`bbox_2d`↔`poly`) reduces false “misses” when a model outputs a valid alternative region encoding.
- Line metrics reflect both missing segments and hallucinated extra segments via TubeIoU (area IoU).

## Risks
- Mask IoU introduces discretization effects. The protocol must define a deterministic rasterization space (norm1000 grid) and tight-window rasterization to keep runtime practical.
- Desc normalization across legacy vs key=value formats requires explicit rules to avoid inflating errors due to superficial formatting differences.

## Rollout plan (high-level)
1. Define the evaluation spec (this change).
2. Implement overlap functions: region IoU (bbox/poly) and line TubeIoU.
3. Implement desc parsing for phase/category (and optional attributes).
4. Implement matching + metric aggregation + reporting outputs.
5. Add unit tests and a small golden dump for smoke validation.
6. Document usage in the docs/tooling index.

