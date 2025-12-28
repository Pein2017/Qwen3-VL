# summary-grpo-post-training Specification (Delta)

## ADDED Requirements

### Requirement: Summary GRPO enforces irrelevant single-line output
Summary-mode GRPO SHALL treat samples with `metadata._fusion_source == "irrelevant_summary"` as irrelevant and SHALL reward only the exact single-line output `无关图片` (no header line, no JSON).

#### Scenario: Irrelevant sample outputs one line
- **GIVEN** a summary-mode sample with `metadata._fusion_source = "irrelevant_summary"`
- **WHEN** computing rewards
- **THEN** only the exact single-line output `无关图片` receives a positive format reward
- **AND** any header line or JSON output is treated as a format failure

### Requirement: Non-irrelevant summaries keep the two-line header + JSON contract
For samples that are not irrelevant, GRPO SHALL enforce the two-line output format with line 1 `<DOMAIN=...>, <TASK=...>` and line 2 as a single-line JSON summary string ending in `}`.

#### Scenario: Non-irrelevant output format
- **GIVEN** a non-irrelevant summary-mode sample
- **WHEN** the model emits output
- **THEN** format reward is positive only if line 1 is the header and line 2 is a valid JSON string ending in `}`

### Requirement: Header matching uses `_fusion_domain` + `mission`
For non-irrelevant samples, header rewards SHALL validate `<DOMAIN>` against `metadata._fusion_domain` and `<TASK>` against `mission`.

#### Scenario: Header token match
- **GIVEN** `_fusion_domain = "bbu"` and `mission = "挡风板安装检查"`
- **WHEN** the model emits `<DOMAIN=bbu>, <TASK=挡风板安装检查>`
- **THEN** header reward is positive

### Requirement: JSON validity and order-invariant equivalence
Content rewards SHALL parse the JSON summary and compare against the ground-truth summary using order-invariant equivalence. The comparison SHALL treat `统计` and `备注` as order-insensitive multisets and SHALL ignore key ordering at the top level.

#### Scenario: Order-invariant summary match
- **GIVEN** a ground-truth summary JSON and a model JSON with the same key/value content but different key order or list order
- **WHEN** computing content accuracy
- **THEN** the summaries are treated as equivalent

### Requirement: RRU-specific fields and deprecated keys
`分组统计` SHALL be allowed for RRU summaries and disallowed for BBU summaries. `备注` SHALL be disallowed for RRU summaries. The reward system SHALL ignore any `异常` field (no reward or penalty).

#### Scenario: RRU allows 分组统计
- **GIVEN** an RRU summary that includes `分组统计`
- **WHEN** computing content rewards
- **THEN** the field is accepted as valid

### Requirement: Reward set includes gated parse penalty and no length penalty
GRPO SHALL use multiple rewards including format, header, parsing validity, and content accuracy. Parse error penalties SHALL be applied only when JSON parsing fails. No length penalty SHALL be applied.

#### Scenario: Parse error penalty is gated
- **GIVEN** a non-irrelevant output with invalid JSON
- **WHEN** computing rewards
- **THEN** parse error penalty is applied and content accuracy is skipped

### Requirement: Irrelevant prompt alternates per epoch
Irrelevant summary samples SHALL alternate between the `summary_bbu` and `summary_rru` prompt templates per epoch using a deterministic epoch-salted policy, while keeping `_fusion_source = "irrelevant_summary"`.

#### Scenario: Irrelevant template alternation
- **GIVEN** two successive epochs
- **WHEN** sampling an irrelevant record
- **THEN** the assigned template may change across epochs but stays stable within the epoch

### Requirement: Rollout settings are fixed
GRPO rollouts SHALL use `num_generations = 3`, `temperature = 0.3`, and `max_length = 2048`, while other hyperparameters default to the current ms-swift version.

#### Scenario: Rollout settings applied
- **GIVEN** GRPO training initialization
- **WHEN** the trainer is constructed
- **THEN** the rollout settings match the fixed values above

### Requirement: Base checkpoint is merged; output is LoRA-only
GRPO summary post-training SHALL load from a merged (full) checkpoint and save only the LoRA adapter outputs for downstream re-merge.

#### Scenario: LoRA-only outputs
- **GIVEN** a merged checkpoint path in configuration
- **WHEN** training completes
- **THEN** only LoRA adapter weights are persisted
