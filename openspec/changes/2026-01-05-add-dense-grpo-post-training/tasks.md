# Tasks: Mixed-mode dense GRPO post-training (BBU/RRU detection) with reward shaping

- [ ] Pre-implementation verification (recommended)
  - [ ] Confirm `metadata._fusion_mode` exists for fusion samples and values are exactly `dense|summary`.
  - [ ] Confirm TubeIoU implementation matches `vis_tools/geometry_eval_metrics.py:tube_iou_line` and uses `DEFAULT_LINE_TOL=8.0` by default.
  - [ ] Ensure any new `DenseSample` / config dataclasses follow Schema Constitution (`dataclass(frozen=True)` for internal state).

- [ ] Spec + design
  - [ ] Confirm reward-call contract with ms-swift GRPO (`reward_func(completions, **kwargs)`) and codify it in specs.
  - [ ] Define dense-vs-summary mode gating rules using `metadata._fusion_mode` and `metadata._fusion_source`.
  - [ ] Define strict geometry validity rules (poly/line) and “no bbox-only fallback” rule for rewards.

- [ ] Dense reward infrastructure (`dense.*`)
  - [ ] Add `src/rlhf/grpo/rewards/dense/context.py` (`DenseSample`) with cached parsing + matching artifacts.
  - [ ] Add `src/rlhf/grpo/rewards/dense/parsing.py`:
    - [ ] strict two-line parsing + header validation (`<TASK=DETECTION>`)
    - [ ] strict JSON parsing with duplicate-key rejection
    - [ ] object schema validation (one geometry key + desc)
    - [ ] `desc` key=value parsing (exact string semantics)
  - [ ] Add `src/rlhf/grpo/rewards/dense/matching.py`:
    - [ ] exact region IoU for bbox/poly using norm1000 mask IoU (supports non-convex polys)
    - [ ] exact line overlap metric (TubeIoU on norm1000 grid; see `vis_tools/geometry_eval_metrics.py`) with configurable tolerance
    - [ ] greedy 1-to-1 matching + IoU sweep reuse
  - [ ] Add `src/rlhf/grpo/rewards/dense/rewards.py`:
    - [ ] `dense.format`
    - [ ] `dense.parse_schema_strict`
    - [ ] `dense.loc_mean_fbeta` (primary; recall-biased to reduce missing objects; default β=2.0)
    - [ ] `dense.loc_soft_recall` (recall shaping; weak/optional FP penalty)
    - [ ] `dense.cat_mean_f1` (secondary)
    - [ ] `dense.attr_weighted_recall` (exact string; down-weight `可见性`; bonus-only OCR/备注; RRU站点距离 strict)

- [ ] Summary reward compatibility (mixed-mode safe)
  - [ ] Update summary ORMs to accept ms-swift call signature (`completions` positional).
  - [ ] Add mode gating so summary rewards no-op on dense samples (avoid `_fusion_template` mapping errors).

- [ ] Reward registry wiring
  - [ ] Register dense rewards alongside summary rewards in `src/rlhf/grpo/rewards/registry.py`.
  - [ ] Update `src/rlhf/grpo/rewards/names.py` to include dense reward identifiers.

- [ ] Configs (GRPO post-training presets)
  - [ ] Add `configs/fusion/base/dense_grpo_mixed.yaml` (dense targets + summary sources; exclude LVIS/chat; include irrelevant).
  - [ ] Add `configs/fusion/variants/bbu_rru_dense_grpo_mixed_2048.yaml` (2048 JSONL paths).
  - [ ] Add `configs/components/rlhf/dense_summary_grpo_mixed.yaml` (reward funcs + weights; max_completion_length=2048).
  - [ ] Add `configs/train/grpo/dense_summary_mixed_base.yaml` (base training preset).
  - [ ] Add `configs/train/grpo/dense_summary_mixed_2048.yaml` (select fusion variant).

- [ ] Offline evaluation alignment
  - [ ] Extend `vis_tools/geometry_eval_metrics.py` to report:
    - [ ] attribute weighted match on matched pairs (desc key=value exact match)
    - [ ] OCR/text match rate for `文本` (bonus-only semantics)
    - [ ] notes match for `备注` (bonus-only semantics)
    - [ ] RRU `站点距离` exact match accuracy
  - [ ] Ensure evaluation uses the same geometry rulers as dense rewards (poly/line exact; no bbox-only fallback for invalid inputs).
  - [ ] Update region IoU in offline evaluation to support non-convex polys (norm1000 mask IoU), aligned with reward computation.

- [ ] Tests + smoke
  - [ ] Unit tests for dense parsing/schema/desc parsing (including duplicate keys).
  - [ ] Unit tests for strict poly/line validity enforcement (no AABB fallback).
  - [ ] Unit tests for attribute weighting rules (可见性 down-weight; 文本/备注 bonus-only; 站点距离 strict).
  - [ ] Add a tiny GRPO smoke config and verify training initializes and emits reward metrics.

- [ ] Docs
  - [ ] Update `docs/training/TRAINING_PLAYBOOK.md` and `docs/training/REFERENCE.md` to include the new mixed-mode GRPO preset and the reward intent (localization-first).
  - [ ] Add an operator recipe for offline evaluation of dense dumps with the new attribute metrics.
