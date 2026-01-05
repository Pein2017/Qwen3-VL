# dense-grpo-post-training Specification (Change Proposal)

## Purpose
Define mixed-mode GRPO post-training for BBU/RRU dense captioning (detection) with summary-mode regularization. The primary optimization target is dense localization quality (geometry), with category and attribute correctness as secondary objectives.

## Non-goals (this change)
- Adding LVIS or language-chat sources to this GRPO post-training stage.
- Changing the JSONL schema or introducing new geometry primitives beyond `bbox_2d`, `poly`, and `line`.
- Semantic/fuzzy matching of attributes; dense attribute rewards use exact string match rules.

## ADDED Requirements

### Requirement: Mixed-mode fusion dataset composition excludes LVIS/chat
Dense GRPO post-training SHALL be executed on a fusion dataset that samples:
- dense targets: `bbu_dense` ratio `1.0` and `rru_dense` ratio `1.0` (mode `dense`)
- summary sources: `bbu_summary` ratio `0.5`, `rru_summary` ratio `0.5` (mode `summary`)
- irrelevant summary: `irrelevant_summary` ratio `0.2` (mode `summary`)
and SHALL NOT include `lvis` or `lang_chat` sources in this stage.

#### Scenario: Fusion config uses the requested dataset mix
- **GIVEN** a GRPO config that references a fusion file for dense post-training
- **WHEN** the fusion config is parsed
- **THEN** the targets include `bbu_dense` and `rru_dense` in `dense` mode
- **AND** the sources include `bbu_summary` (ratio 0.5), `rru_summary` (ratio 0.5), and `irrelevant_summary` (ratio 0.2) in `summary` mode
- **AND** the fusion config contains no `lvis` or `lang_chat` sources

### Requirement: Dense GRPO rollouts use max_completion_length=2048
Dense GRPO configs for this mixed-mode post-training stage SHALL set `max_completion_length = 2048` for rollouts.

#### Scenario: Rollout max completion length is 2048
- **GIVEN** a dense GRPO config for mixed-mode post-training
- **WHEN** the trainer is constructed
- **THEN** the rollout `max_completion_length` equals `2048`

### Requirement: Dense outputs use the two-line header + object-dict JSON contract
For dense-mode samples, GRPO SHALL enforce the two-line output contract:
- Line 1: `<DOMAIN=BBU|RRU>, <TASK=DETECTION>`
- Line 2: a single-line JSON object mapping `object_{n}` keys to object entries that contain:
  - `desc` (string)
  - exactly one geometry key: `bbox_2d` or `poly` or `line`
  - optional `line_points` when the geometry key is `line`

Geometry representation in the assistant output SHALL follow the dense prompt contract:
- `bbox_2d`: flat array `[x1, y1, x2, y2]` (norm1000 integers)
- `poly`: a list of point pairs `[[x1, y1], [x2, y2], ...]` (at least 3 points)
- `line`: a list of point pairs `[[x1, y1], [x2, y2], ...]` (at least 2 points)

For robustness, reward parsing MAY accept `poly`/`line` geometries emitted as flat arrays `[x1, y1, ...]` when they are even-length and meet the minimum point count.

#### Scenario: Dense header task token is DETECTION
- **GIVEN** a dense-mode sample from the BBU domain
- **WHEN** the model emits `<DOMAIN=BBU>, <TASK=DETECTION>` on line 1
- **THEN** header reward is positive
- **WHEN** the model emits `<TASK!=DETECTION>` for a dense-mode sample
- **THEN** header reward is zero (or negative) and downstream dense rewards are gated off

### Requirement: Reward functions are mode-gated (dense vs summary)
Dense GRPO reward implementations SHALL apply dense rewards only to samples where `metadata._fusion_mode == "dense"` and SHALL apply summary rewards only to samples where `metadata._fusion_mode == "summary"` (including irrelevant summary samples).

Mode-mismatched reward functions SHALL return a neutral per-sample reward value of `0.0`.

#### Scenario: Dense rewards do not parse summary outputs
- **GIVEN** a summary-mode sample (`metadata._fusion_mode == "summary"`)
- **WHEN** a `dense.*` reward function is invoked
- **THEN** it returns `0.0` and does not attempt dense JSON parsing or geometry matching

#### Scenario: Mode-mismatched rewards return zeros
- **GIVEN** a batch containing both dense-mode and summary-mode samples
- **WHEN** a reward function is invoked on a sample whose `metadata._fusion_mode` does not match the reward namespace
- **THEN** that sample’s reward is `0.0`

### Requirement: Exact poly/line geometry scoring without bbox-only fallback
Dense GRPO localization rewards SHALL compute geometry overlap using exact rulers:
- Region family (`bbox_2d`, `poly`): filled-shape IoU with cross-type bbox↔poly matching supported.
- Line family (`line`): TubeIoU on the norm1000 grid with a stability-first tolerance (default `tol=8.0` in norm1000 space), computed by rasterizing each polyline into a tube mask of width `round(2*tol)` and then taking `|A∩B|/|A∪B|`.

Rewards SHALL NOT approximate invalid `poly`/`line` geometries using bbox/AABB fallback; invalid geometries SHALL be treated as schema failures and excluded from matching.

#### Scenario: Invalid poly is rejected (no AABB fallback)
- **GIVEN** a predicted object with `poly` points that are neither:
  - a list of point pairs `[[x, y], ...]` with at least 3 points
  - nor a flat even-length array `[x1, y1, ...]` with at least 3 points
- **WHEN** localization rewards are computed
- **THEN** the object is treated as invalid
- **AND** matching does not use bbox/AABB approximations for that object

### Requirement: Localization rewards bias toward recall (missing-object reduction) with weak hallucination penalties
Dense GRPO reward shaping SHALL prioritize reducing missing objects and SHALL avoid strong penalties on unmatched predictions, because GT annotations may be incomplete.

Concretely:
- Localization aggregation SHOULD be recall-biased (default mean-F2 across thresholds; β=2 implies FN weight `β²=4×` relative to FP inside Fβ).
- Precision/false-positive penalties SHALL be small and bounded (including possibly zero).
- Category and attribute rewards SHALL be computed on matched pairs only.

#### Scenario: Extra predicted objects do not dominate negative reward
- **GIVEN** a dense sample with incomplete GT (some true objects are missing from annotation)
- **WHEN** the model predicts additional valid objects beyond the annotated GT
- **THEN** localization rewards do not apply a large negative penalty solely due to unmatched predictions
- **AND** missing-object reduction remains the primary optimization signal

### Requirement: Localization is the primary objective; category is secondary
Dense GRPO configurations SHALL include:
- at least one localization-first reward (geometry-only matching, recall-biased mean-Fβ across thresholds; default β=2.0)
- at least one category-aware reward (requires `类别` equality on matched pairs)
and SHALL weight localization rewards higher than category rewards.

#### Scenario: Reward weighting prioritizes localization
- **GIVEN** a dense GRPO config with reward functions and weights
- **WHEN** reward aggregation is performed
- **THEN** the combined weight of localization rewards exceeds the combined weight of category rewards

### Requirement: Attribute scoring uses exact key=value matches with business weighting
For matched dense objects, attribute rewards SHALL treat `desc` as comma-separated `key=value` terms and SHALL score exact string matches with the following semantics:
- Attribute scoring SHALL use weighted recall over GT attributes (prediction extra keys not present in GT are neutral).
- Keys and values SHALL be normalized by removing all whitespace characters before comparison (e.g., `品牌=华为` equals `品牌 = 华为`).
- Default per-key weights:
  - Normal non-OCR keys: `1.0`
  - `可见性`: `0.1` (least important; noisy annotation)
  - `站点距离`: `4.0` (important; exact integer match only)
- OCR/notes keys (`文本`, `备注`):
  - bonus-only with default bonus weight `6.0` each
  - reward only when the key exists in GT and the predicted value matches exactly (after whitespace removal)
  - no penalty when missing or mismatch

#### Scenario: 可见性 is down-weighted, OCR/备注 are bonus-only
- **GIVEN** two matched objects with identical `类别`
- **WHEN** all non-OCR attributes match except `可见性`
- **THEN** attribute reward decreases only mildly (down-weighted penalty)
- **WHEN** `文本` or `备注` mismatches
- **THEN** no negative penalty is applied for those keys

### Requirement: RRU site distance is strict and high-signal
For RRU dense objects where `类别=站点距离`, rewards SHALL require an exact integer match for `站点距离=<int>` and SHALL weight this key as high-signal relative to other attributes.

#### Scenario: 站点距离 exact match required
- **GIVEN** a GT object with `类别=站点距离,站点距离=123`
- **WHEN** the prediction emits `类别=站点距离,站点距离=123`
- **THEN** attribute reward is strongly positive
- **WHEN** the prediction emits a different integer (or omits the key)
- **THEN** attribute reward for this key is zero (no penalty), and no fuzzy matching is applied

### Requirement: Reward-call compatibility with ms-swift GRPO
Custom dense GRPO rewards SHALL be compatible with ms-swift GRPO reward invocation and SHALL accept a positional `completions` argument plus batched kwargs.

#### Scenario: Reward receives completions and batched metadata
- **WHEN** ms-swift invokes a `dense.*` reward during GRPO training
- **THEN** the reward receives `completions` as a positional list of strings
- **AND** receives `metadata` and `assistant_payload` in kwargs as batched per-row lists

### Requirement: Offline evaluation reports dense attribute/OCR/site-distance metrics
The offline `gt_vs_pred.jsonl` evaluator SHALL report, in addition to localization and category metrics, dense attribute diagnostics aligned with reward semantics:
- weighted attribute match (with `可见性` down-weighted)
- OCR/text match rate for `文本` (bonus-only)
- notes match rate for `备注` (bonus-only)
- RRU `站点距离` exact match accuracy

#### Scenario: Evaluation includes attribute diagnostics
- **GIVEN** a dense `gt_vs_pred.jsonl` dump
- **WHEN** evaluation runs
- **THEN** the JSON report includes localization and category mean-F1 metrics
- **AND** includes the attribute/OCR/site-distance diagnostic metrics defined above
