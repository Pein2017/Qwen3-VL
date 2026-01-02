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

### Requirement: Header matching uses dataset domain token and TASK=SUMMARY
For non-irrelevant samples, header rewards SHALL validate `<DOMAIN>` against the dataset domain token (`BBU` or `RRU`) surfaced in metadata and SHALL validate `<TASK>` as the fixed token `SUMMARY` for summary-mode outputs.

#### Scenario: Header token match
- **GIVEN** a summary-mode sample with `metadata._fusion_domain_token = "BBU"`
- **WHEN** the model emits `<DOMAIN=BBU>, <TASK=SUMMARY>`
- **THEN** header reward is positive

### Requirement: JSON validity and order-invariant equivalence
Content rewards SHALL parse the JSON summary and compare against the ground-truth summary from `metadata.summary_ref` using order-invariant equivalence. The comparison SHALL treat `统计` and `备注` as order-insensitive multisets and SHALL ignore key ordering at the top level.

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
Irrelevant summary samples SHALL alternate between the `summary_bbu` and `summary_rru` prompt templates per epoch using a deterministic epoch-salted policy (roughly 50/50 within an epoch), while keeping `_fusion_source = "irrelevant_summary"`. Evaluation SHALL use a deterministic mapping that is stable across runs.

#### Scenario: Irrelevant template alternation
- **GIVEN** two successive epochs
- **WHEN** sampling an irrelevant record
- **THEN** the assigned template may change across epochs but stays stable within the epoch

### Requirement: Rollout settings are fixed
GRPO rollouts SHALL use `num_generations = 3`, `temperature = 0.3`, and `max_completion_length = 2048`, while other hyperparameters default to the current ms-swift version.

#### Scenario: Rollout settings applied
- **GIVEN** GRPO training initialization
- **WHEN** the trainer is constructed
- **THEN** the rollout settings match the fixed values above

### Requirement: GRPO launch uses shared modules and `rlhf` block
Summary GRPO SHALL be launched via `scripts/train.sh` (shared training entrypoint) and SHALL configure GRPO entirely under the `rlhf` block (including `rlhf_type=grpo`, reward functions/weights, and rollout settings).

#### Scenario: Shared GRPO launch path
- **GIVEN** a summary GRPO config
- **WHEN** training is launched
- **THEN** `scripts/train.sh` invokes the standard training entrypoint with `rlhf` settings and no custom trainer

### Requirement: Fusion-based dataset toggle is required
Summary GRPO SHALL use `custom.fusion_config` pointing to a `configs/fusion/*` definition. Summary targets SHALL set `mode: summary` (or inherit `custom.use_summary: true`) so that all GRPO targets are in summary mode.

#### Scenario: Fusion config summary toggle
- **GIVEN** a GRPO summary config referencing a fusion file
- **WHEN** dataset loading begins
- **THEN** summary targets are loaded in summary mode via the fusion definition

### Requirement: Summary prompt profile and assistant prefix are required
Summary GRPO configs SHALL set `prompts.profile: summary_runtime` and SHALL set `custom.assistant_prefix_format` to `<DOMAIN={domain}>, <TASK={task}>` for non-irrelevant summaries. Irrelevant summaries SHALL not include the header line and SHALL be scored as format failures if a header is generated.

#### Scenario: Prompt profile alignment
- **GIVEN** a GRPO summary config
- **WHEN** prompts are resolved
- **THEN** the summary_runtime profile and assistant prefix format are applied

### Requirement: Base checkpoint is merged; output is LoRA-only
GRPO summary post-training SHALL load from a merged (full) checkpoint and save only the LoRA adapter outputs for downstream re-merge.

#### Scenario: LoRA-only outputs
- **GIVEN** a merged checkpoint path in configuration
- **WHEN** training completes
- **THEN** only LoRA adapter weights are persisted

### Requirement: Stage-B compatibility acceptance
Post-training evaluation SHALL confirm that Stage-B inputs tolerate summary outputs that are either (a) `<DOMAIN=...>, <TASK=SUMMARY>` plus JSON on line 2, or (b) single-line `无关图片` with no header.

#### Scenario: Mixed-format Stage-B inputs
- **GIVEN** a Stage-B input batch containing prefixed summaries and single-line `无关图片`
- **WHEN** Stage-B prompt assembly runs
- **THEN** no format-related rejection or parse failure occurs
