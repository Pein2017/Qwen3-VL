# summary-grpo-post-training Specification

## Purpose
Define summary-mode GRPO post-training for BBU/RRU summary outputs and irrelevant negatives. This spec fully supersedes summary-mode GRPO guidance in `grpo-integration`.
## Requirements
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
For non-irrelevant samples, header rewards SHALL validate `<DOMAIN>` against a domain token derived from `_fusion_template` (`summary_bbu` → `BBU`, `summary_rru` → `RRU`) and SHALL validate `<TASK>` as the fixed token `SUMMARY` for summary-mode outputs. `_fusion_source == "irrelevant_summary"` SHALL suppress header checks entirely (even if `_fusion_template` is `summary_bbu`).

#### Scenario: Header token match via fusion template
- **GIVEN** a summary-mode sample with `_fusion_template = "summary_bbu"`
- **WHEN** the model emits `<DOMAIN=BBU>, <TASK=SUMMARY>`
- **THEN** header reward is positive

### Requirement: JSON validity and order-invariant equivalence
Content rewards SHALL parse the JSON summary and compare against the ground-truth summary from `metadata.summary_ref` using order-invariant equivalence. The comparison SHALL treat `统计` and `备注` as order-insensitive multisets and SHALL ignore key ordering at the top level. For RRU summaries, `分组统计` SHALL be included in content scoring (group_id → count, order-insensitive by key). The JSON MUST NOT contain a `dataset` key; its presence SHALL invalidate the sample for content rewards.

#### Scenario: Dataset key invalidates summary JSON
- **GIVEN** a summary-mode sample whose JSON contains `"dataset"`
- **WHEN** computing content rewards
- **THEN** the summary is treated as invalid and content rewards are zeroed (or the sample fails fast per implementation)

#### Scenario: Group statistics are scored for RRU
- **GIVEN** an RRU summary reference that includes `分组统计`
- **WHEN** a prediction omits or mismatches the `分组统计` counts
- **THEN** the content reward reflects the mismatch (lower than a matching prediction)

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

### Requirement: CHORD mixing toggle is GRPO-only and fails fast otherwise
The `custom.grpo.chord.enabled` toggle SHALL only be valid when `rlhf.rlhf_type == "grpo"`. If enabled when `rlhf.rlhf_type != "grpo"`, training initialization SHALL fail fast with a clear configuration error.

#### Scenario: Enabled under non-GRPO fails fast
- **GIVEN** a config with `custom.grpo.chord.enabled: true`
- **AND** `rlhf.rlhf_type` is not `grpo`
- **WHEN** training initialization begins
- **THEN** the process fails fast with a clear configuration error message

### Requirement: Optional CHORD SFT mixing provides a supervised fallback signal
Summary GRPO post-training SHALL support an optional CHORD-style mixed loss that combines GRPO loss with a supervised (SFT) loss to provide training signal when rollouts are identical.

#### Scenario: CHORD mixing disabled preserves baseline behavior
- **GIVEN** a summary GRPO config with `custom.grpo.chord.enabled: false` (or absent)
- **WHEN** training runs
- **THEN** the trainer computes GRPO loss only
- **AND** no CHORD SFT dataset is attached

#### Scenario: CHORD mixing enabled attaches expert data and mixes losses
- **GIVEN** a summary GRPO config with `custom.grpo.chord.enabled: true` and valid CHORD schedule parameters
- **WHEN** training runs
- **THEN** the trainer computes `loss = (1 - mu) * grpo_loss + mu * chord_sft_loss`
- **AND** the expert dataset is attached for CHORD SFT loss computation

### Requirement: CHORD expert dataset defaults to the GRPO training dataset
When CHORD mixing is enabled for summary GRPO post-training, the system SHALL default the CHORD expert dataset to the same fusion dataset used for GRPO training so that supervised targets match the summary output contract.

#### Scenario: Expert dataset defaults to fusion train dataset
- **GIVEN** summary GRPO training launched with a fusion dataset and CHORD mixing enabled
- **WHEN** the trainer is constructed
- **THEN** the CHORD expert dataset uses the same fusion train dataset records as the GRPO training dataset
- **AND** irrelevant samples (`metadata._fusion_source == "irrelevant_summary"`) are included without filtering
- **AND** irrelevant targets remain the single-line `无关图片` contract

### Requirement: CHORD mixing is config-toggleable and validated
The system SHALL provide a single YAML toggle to enable/disable CHORD mixing for summary GRPO post-training via `custom.grpo.chord.enabled`. When enabled, required CHORD schedule parameters SHALL be validated at startup, and the trainer SHALL receive the corresponding ms-swift CHORD arguments.

Required fields when enabled:
- `custom.grpo.chord.mu_warmup_steps` (int >= 0)
- `custom.grpo.chord.mu_decay_steps` (int >= 0)
- `custom.grpo.chord.mu_peak` (float in [0, 1])
- `custom.grpo.chord.mu_valley` (float in [0, 1])
- `custom.grpo.chord.sft_per_device_train_batch_size` (int > 0)

Optional fields:
- `custom.grpo.chord.enable_phi_function` (bool; default false)

When enabled, the trainer args SHALL be populated as:
- `chord_mu_warmup_steps = custom.grpo.chord.mu_warmup_steps`
- `chord_mu_decay_steps = custom.grpo.chord.mu_decay_steps`
- `chord_mu_peak = custom.grpo.chord.mu_peak`
- `chord_mu_valley = custom.grpo.chord.mu_valley`
- `chord_sft_per_device_train_batch_size = custom.grpo.chord.sft_per_device_train_batch_size`
- `chord_enable_phi_function = custom.grpo.chord.enable_phi_function`

#### Scenario: Missing schedule parameters fails fast
- **GIVEN** CHORD mixing is enabled but one or more required schedule fields are missing
- **WHEN** training initialization begins
- **THEN** the process fails fast with a clear configuration error message

### Requirement: Reward identifiers use namespaced dot form
Summary GRPO configs SHALL use namespaced dot identifiers for reward functions (e.g., `summary.format`, `summary.header`, `summary.parse`). Legacy snake-case identifiers (e.g., `summary_format`) are unsupported and SHALL fail validation.

#### Scenario: Legacy reward identifier is rejected
- **GIVEN** a GRPO config whose `rlhf.reward_funcs` includes `summary_format`
- **WHEN** configuration validation runs
- **THEN** validation fails with a clear error that legacy reward identifiers are unsupported

### Requirement: GRPO CHORD configuration uses `custom.grpo.chord`
When CHORD is enabled for summary GRPO, configs SHALL define it under `custom.grpo.chord` and SHALL include required fields: `sft_per_device_train_batch_size`, `mu_warmup_steps`, `mu_decay_steps`, `mu_peak`, `mu_valley`, and `enable_phi_function`. The legacy `custom.grpo_chord` field is unsupported and SHALL fail validation.

#### Scenario: CHORD config is validated
- **GIVEN** a GRPO config with `custom.grpo.chord.enabled: true`
- **WHEN** configuration validation runs
- **THEN** all required CHORD fields must be present and valid
- **AND** validation fails if only `custom.grpo_chord` is provided

