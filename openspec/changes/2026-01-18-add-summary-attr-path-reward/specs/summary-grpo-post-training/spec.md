# summary-grpo-post-training (Delta)

## ADDED Requirements

### Requirement: Summary GRPO provides an optional nested attribute-path completeness reward
The system SHALL provide an optional summary GRPO reward function (reward id: `summary.attr_path_recall`) that scores nested attribute **paths** (depth=2) derived from `metadata.summary_ref`, intended to improve schema completeness for stable semantic fields (e.g., `可见性/完整`, `符合性/符合`, `捆扎/整齐`, `弯曲半径/半径合理`).

The reward SHALL:
- Operate on `统计[*]` entries and require a string `类别` field per entry.
- Score **completeness only for categories that the model chooses to emit and that also exist in the ground-truth summary** (i.e., scoring categories are `pred_categories ∩ ref_categories`). This reward MUST NOT directly penalize omitting a category (category recall is handled by other rewards).
- Consider only a curated set of stable parent keys for recursion (implementation-defined), stored as code constants and covered by unit tests (at minimum: `捆扎`, `可见性`, `符合性`, `弯曲半径`).
- Exclude `文本` and OCR/free-text-like dynamic keys using explicit minimum heuristics:
  - SHALL exclude: digit-only child keys (e.g., `"263"`).
  - SHALL exclude: any parent key named `文本` (entire field ignored).
  - SHOULD exclude: child keys that contain `=`.
  - SHOULD exclude: child keys longer than 32 characters.
- Be compatible with both BBU and RRU summary templates.

#### Scenario: Reward improves completeness for emitted categories
- **GIVEN** a summary sample with a ground-truth `summary_ref` that includes `电线` with `捆扎: {整齐: 1}`
- **WHEN** the model emits category `电线` but omits `捆扎/整齐`
- **THEN** the nested attribute-path reward is lower than for a prediction that includes `捆扎/整齐`

#### Scenario: Omitting a category does not incur a direct penalty from this reward
- **GIVEN** a summary sample with a ground-truth `summary_ref` that includes category `电线`
- **WHEN** the model does not emit category `电线`
- **THEN** `summary.attr_path_recall` does not directly penalize the omission (it scores only categories in `pred ∩ ref`)

#### Scenario: Dynamic child keys are excluded
- **GIVEN** a summary sample whose `summary_ref` includes fields with dynamic digit keys (e.g., `站点距离: {"263": 1}`)
- **WHEN** computing the nested attribute-path reward
- **THEN** the reward does not require reproducing digit-only child keys
