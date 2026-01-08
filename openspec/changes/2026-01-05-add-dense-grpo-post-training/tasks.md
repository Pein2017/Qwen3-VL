# Tasks: Dense GRPO post-training (BBU/RRU detection) with reward shaping

- [x] Pre-implementation verification (recommended)
  - [x] Confirm `metadata._fusion_mode` exists for fusion samples and values are exactly `dense|summary`.
  - [x] Confirm TubeIoU implementation matches `vis_tools/geometry_eval_metrics.py:tube_iou_line` and uses `DEFAULT_LINE_TOL=8.0` by default.
  - [x] Ensure any new `DenseSample` / config dataclasses follow Schema Constitution (`dataclass(frozen=True)` for internal state).

- [x] Spec + design
  - [x] Confirm reward-call contract with ms-swift GRPO (`reward_func(completions, **kwargs)`) and codify it in specs.
  - [x] Define dense-vs-summary mode gating rules using `metadata._fusion_mode` and `metadata._fusion_source`.
  - [x] Define strict geometry validity rules (poly/line) and “no bbox-only fallback” rule for rewards.

- [x] Dense reward infrastructure (`dense.*`)
  - [x] Add `src/rlhf/grpo/rewards/dense/context.py` (`DenseSample`) with cached parsing + matching artifacts.
  - [x] Add `src/rlhf/grpo/rewards/dense/parsing.py`:
    - [x] strict two-line parsing + header validation (`<TASK=DETECTION>`)
    - [x] strict JSON parsing with duplicate-key rejection
    - [x] object schema validation (one geometry key + desc)
    - [x] `desc` key=value parsing (exact string semantics)
  - [x] Add `src/rlhf/grpo/rewards/dense/matching.py`:
    - [x] exact region IoU for bbox/poly using norm1000 mask IoU (supports non-convex polys)
    - [x] exact line overlap metric (TubeIoU on norm1000 grid; see `vis_tools/geometry_eval_metrics.py`) with configurable tolerance
    - [x] greedy 1-to-1 matching + IoU sweep reuse
  - [x] Add `src/rlhf/grpo/rewards/dense/rewards.py`:
    - [x] `dense.format`
    - [x] `dense.parse_schema_strict`
    - [x] `dense.loc_mean_fbeta` (primary; recall-biased to reduce missing objects; default β=2.0)
    - [x] `dense.loc_soft_recall` (recall shaping; weak/optional FP penalty)
    - [x] `dense.cat_mean_f1` (secondary)
    - [x] `dense.attr_weighted_recall` (exact string; down-weight `可见性`; bonus-only OCR/备注; RRU站点距离 strict)

- [x] Summary reward compatibility (mixed-mode safe)
  - [x] Update summary ORMs to accept ms-swift call signature (`completions` positional).
  - [x] Add mode gating so summary rewards no-op on dense samples (avoid `_fusion_template` mapping errors).

- [x] Reward registry wiring
  - [x] Register dense rewards alongside summary rewards in `src/rlhf/grpo/rewards/registry.py`.
  - [x] Update `src/rlhf/grpo/rewards/names.py` to include dense reward identifiers.

- [x] Configs (GRPO post-training presets)
  - [x] Add `configs/fusion/base/dense_grpo.yaml` (dense targets only; no summary mixing).
  - [x] Add `configs/fusion/variants/bbu_rru_dense_grpo_2048.yaml` (2048 JSONL paths).
  - [x] Add `configs/components/rlhf/dense_grpo.yaml` (reward funcs + weights; dense-only).
  - [x] Add `configs/train/grpo/dense_base.yaml` (base training preset).
  - [x] Add `configs/train/grpo/dense_2048.yaml` (select fusion variant).

- [x] Offline evaluation alignment
  - [x] Extend `vis_tools/geometry_eval_metrics.py` to report:
    - [x] attribute weighted match on matched pairs (desc key=value exact match)
    - [x] OCR/text match rate for `文本` (bonus-only semantics)
    - [x] notes match for `备注` (bonus-only semantics)
    - [x] RRU `站点距离` exact match accuracy
  - [x] Ensure evaluation uses the same geometry rulers as dense rewards (poly/line exact; no bbox-only fallback for invalid inputs).
  - [x] Update region IoU in offline evaluation to support non-convex polys (norm1000 mask IoU), aligned with reward computation.

- [ ] Tests + smoke
  - [x] Unit tests for dense parsing/schema/desc parsing (including duplicate keys).
  - [x] Unit tests for strict poly/line validity enforcement (no AABB fallback).
  - [x] Unit tests for attribute weighting rules (可见性 down-weight; 文本/备注 bonus-only; 站点距离 strict).
  - [x] Add a tiny GRPO smoke config (`configs/smoke/grpo_dense.yaml`) and a config-load test.
  - [ ] Run the smoke config (1–2 steps) to confirm trainer init + reward logging (requires GPU + checkpoint).

- [x] Docs
  - [x] Update `docs/training/TRAINING_PLAYBOOK.md` and `docs/training/REFERENCE.md` to include the dense GRPO preset and the reward intent (localization-first).
  - [x] Add an operator recipe for offline evaluation of dense dumps with the new attribute metrics.
