# grpo-integration Specification

> **Deprecated**: Superseded by `summary-grpo-post-training` for summary-mode GRPO. New work should target the newer spec.

## Purpose
Define the Stage-A (image summary) → Stage-B (text-only GRPO) pipeline, including image ingestion/aggregation, strict two-line verdict format, reward wiring, and GRPO training/inference expectations for group-level reasoning.
## Requirements
### Requirement: Per-image inference and grouped aggregation
- The Stage-A engine SHALL accept an input directory containing multiple groups with arbitrarily many images (N ≥ 1).
- Supported extensions: {jpg, jpeg, png} (case-insensitive); files MUST be discovered deterministically with natural sort.
- The engine SHALL produce a Chinese single-line summary per image; implementations MAY batch multiple images per forward pass for throughput.
- A group aggregator SHALL construct a single JSONL record per group with `per_image` keys `image_{i}` aligned to the deterministic image order.
- Group ID SHALL be derived from filenames using the pattern `^(QC-[A-Za-z]+-[0-9]{8}-[0-9]+)` when present; otherwise the immediate subdirectory name is used.
- Output JSONL fields: `group_id`, `mission`, `label` (`pass|fail`), `images` (list), and `per_image` (dict by `image_{i}`).

#### Scenario: Basic directory with 3 images in one group
- **GIVEN** files: QC-TEMP-20250118-0015956-001.jpeg, -002.jpeg, -010.jpeg
- **WHEN** running the Stage-A engine
- **THEN** the JSONL has one record with `images` length 3 and `per_image` keys: `image_1`, `image_2`, `image_3`

#### Scenario: Multi-group directory processing
- **GIVEN** subdirectories with different group IDs
- **WHEN** processing recursively
- **THEN** each group produces one JSONL record with correctly sorted `image_{i}` keys

### Requirement: image_{i} alignment and coverage (strict validation)
- The aggregator MUST assign `image_{1..N}` to match the sorted input order exactly.
- If any `image_{i}` index is missing or extra relative to N, the engine SHALL raise a ValueError and abort the current group.
- All discovered images MUST yield a non-empty cleaned summary; otherwise the engine SHALL raise ValueError and abort the group.

#### Scenario: Mismatch in image_{i} indices
- **GIVEN** 2 input images for a group but per_image has keys `image_1` and `image_3`
- **THEN** the engine raises ValueError and does not write a partial record for that group

#### Scenario: Empty summary validation
- **GIVEN** an image that returns an empty string after cleaning
- **THEN** the engine raises ValueError for that group and writes no record

### Requirement: Chinese prompting and decoding
- The system SHALL use the model's native chat_template; no manual special tokens.
- Prompts and responses SHALL be in Chinese; prompt wording is pluggable and may be tailored per task type.
- The system SHALL save both raw and cleaned text per image; aggregator assembles per_image mapping without requiring model-emitted `image_{i}`.

#### Scenario: Task-specific prompting
- **GIVEN** a mission name like "BBU安装方式检查（正装）"
- **WHEN** generating prompts
- **THEN** the system uses mission-specific Chinese instructions

### Requirement: Stage-A SHALL expose a sharding mode switch (per_group vs per_image)
Stage-A SHALL provide a CLI option `--sharding_mode` with values:
- `per_group` (default): shard work at group granularity; a group's images are processed by a single rank.
- `per_image`: shard work at image granularity; a group's images MAY be processed by multiple ranks.

#### Scenario: Default behavior remains per_group
- **WHEN** Stage-A is run without an explicit `--sharding_mode`
- **THEN** Stage-A uses `per_group`

### Requirement: Stage-A SHALL NOT provide backward-compatible aliases for removed batching modes
Stage-A SHALL expose only `--sharding_mode` for controlling execution strategy and SHALL reject legacy batching-mode flags.

#### Scenario: Legacy batching flag is rejected
- **WHEN** Stage-A is invoked with a removed `--batching_mode` flag
- **THEN** Stage-A fails fast with a clear CLI validation error

### Requirement: Group sampling SHALL occur before image job flattening
When `pass_group_number`/`fail_group_number` sampling is enabled, Stage-A SHALL sample at the group level before creating per-image jobs.

#### Scenario: Sampling preserves group counts in per_image mode
- **GIVEN** `pass_group_number=P` and `fail_group_number=F`
- **WHEN** Stage-A runs with `--sharding_mode per_image`
- **THEN** at most P pass groups and at most F fail groups are selected
- **AND** only images belonging to those selected groups are processed

### Requirement: Sharding modes define alignment/coverage equivalence, not text identity
When comparing `per_group` vs `per_image` runs over the same discovered+sampled groups, the correctness criteria SHALL be:
- identical selected group set
- unchanged group-level output schema and strict `image_{i}` alignment/coverage

The generated summary strings MAY differ across modes and across runs under stochastic decoding.

#### Scenario: Same groups, different texts are allowed
- **GIVEN** a fixed input directory and fixed sampling seed
- **WHEN** Stage-A runs with `--sharding_mode per_group` and `--sharding_mode per_image`
- **THEN** the set of output groups (as `group_id::label`) is identical
- **AND** each output record satisfies strict `image_{1..N}` coverage
- **AND** per-image summary text is allowed to differ

### Requirement: per_group mode SHALL avoid cross-group image batches
In `per_group` mode, Stage-A SHALL batch only within a group and SHALL NOT form a forward-pass batch containing images from multiple groups.

#### Scenario: per_group batching is group-local
- **GIVEN** two groups A and B
- **WHEN** Stage-A runs with `--sharding_mode per_group` and `batch_size>1`
- **THEN** no forward pass contains both A and B images

### Requirement: per_image mode SHALL shard images across ranks and merge on rank 0
In single-process or distributed runs, `per_image` mode SHALL:
- flatten selected groups into deterministic image jobs with `(group_seq, image_index, path)`
- shard image jobs across ranks deterministically when `WORLD_SIZE>1` (e.g., round-robin)
- execute per-rank inference in batches while keeping at most `batch_size` images in flight
- write per-rank intermediate per-image results
- merge on rank 0 into group-level JSONL records

#### Scenario: per_image sharding improves balance without changing group contract
- **GIVEN** group A has 20 images and group B has 1 image
- **WHEN** Stage-A runs with `--sharding_mode per_image` under `WORLD_SIZE=2`
- **THEN** images from group A MAY be processed on both ranks
- **AND** the final output still contains exactly one JSONL record per successfully completed group with `images` and `per_image` aligned

### Requirement: per_image merge SHALL mark incomplete groups as failed and continue
In `per_image` mode, rank 0 merge SHALL treat any group with missing/failed image summaries as a group failure:
- the group SHALL NOT emit a partial JSONL record
- merge SHALL continue for other groups

#### Scenario: Missing image result causes group failure only
- **GIVEN** group A has 3 images but only image_1 and image_2 summaries are present at merge time
- **WHEN** Stage-A merges results
- **THEN** group A emits no record
- **AND** other complete groups are still written

### Requirement: per_image intermediate outputs SHALL be deleted by default and optionally retained
In `per_image` mode, Stage-A SHALL delete intermediate per-rank per-image output files after a successful merge by default, and SHALL provide an option to retain them for debugging.

#### Scenario: Default cleanup of intermediate files
- **WHEN** Stage-A completes a `per_image` run successfully without a keep-intermediate option
- **THEN** intermediate per-image per-rank files are removed

#### Scenario: Keep intermediate files for debugging
- **WHEN** Stage-A runs with an explicit keep-intermediate option
- **THEN** intermediate per-image per-rank files are preserved after merge

### Requirement: Final Stage-A output ordering MAY vary
Stage-A outputs (including merged outputs under distributed runs) MAY be written in any order; consumers SHALL NOT depend on JSONL record ordering for correctness.

#### Scenario: Merge order is not a correctness criterion
- **GIVEN** a fixed input directory and fixed sampling seed
- **WHEN** Stage-A runs with `--sharding_mode per_group` and `--sharding_mode per_image`
- **THEN** the set of output groups is identical
- **AND** record ordering differences are permitted


<!-- Phase 2: Stage-B Group-Level Dataset -->

### Requirement: One group per sample (text-only, no images)
- Each Stage-B sample SHALL represent one group; required fields:
  - `group_id`: string
  - `mission`: string
  - `label`: `pass` | `fail`
  - `stage_a_summaries`: dict {image_i: 中文单行摘要}
  - `messages`: list[dict] (system in Chinese; user includes task focus and Stage-A summaries as plain text)
- Stage-B inputs SHALL NOT include any image data (text-only reasoning).

#### Scenario: Minimal valid sample
- **GIVEN** a JSONL record with messages referencing image_1..image_k summaries
- **WHEN** loading for GRPO training
- **THEN** the sample contains label, stage_a_summaries, and messages without images

#### Scenario: Stage-A to Stage-B conversion
- **GIVEN** Stage-A output JSONL with per_image summaries
- **WHEN** converting to Stage-B format
- **THEN** each group becomes one text-only sample with embedded summaries in user message

### Requirement: Two-line model output contract（严格两行格式）
- Line 1 SHALL be exactly "通过" or "不通过" (verdict only, no extra characters)
- Line 2 SHALL be "理由: <reasoning>" in Chinese natural language
- The output MAY reference image_i in reasoning; no fixed vocabulary required
- Extra lines are forbidden; trailing whitespace is allowed; empty reasoning is penalized by format/length rewards

#### Scenario: Valid two-line output
- **GIVEN** completion: `通过\n理由: 基于image_1的安装方式符合规范`
- **THEN** format reward is high; label reward checks line 1 against label

#### Scenario: Invalid format (extra words in verdict)
- **GIVEN** completion: `通过了\n理由: ...`
- **THEN** format reward is 0; label reward ignores malformed verdict

#### Scenario: Missing line 2
- **GIVEN** completion: `通过`
- **THEN** format reward is 0

### Requirement: Reward function passthrough
- The dataset loader MUST pass `stage_a_summaries`, `label`, and `mission` to reward functions via kwargs.
- Rewards SHALL have access to these fields for label matching and potential consistency checks.

#### Scenario: Reward function receives context
- **GIVEN** a completion generated by GRPO
- **WHEN** computing rewards
- **THEN** reward functions receive stage_a_summaries and label for scoring

<!-- Phase 3: GRPO Training Integration -->

### Requirement: Programmatic launcher with ms-swift GRPO trainer
- The system SHALL provide a Python launcher (no CLI) exposing `run_grpo(config)` that constructs trainer, datasets, and rewards.
- The launcher SHALL set `num_generations >= 2` and satisfy batch divisibility requirements.
- Rewards SHALL be loaded via Python modules; initial v1 uses label reward and format reward only.

#### Scenario: Launcher configuration
- **GIVEN** a config dict with model path, dataset path, and reward weights
- **WHEN** calling `run_grpo(config)`
- **THEN** the launcher initializes GRPO trainer with correct settings

#### Scenario: Reward weight mismatch validation
- **GIVEN** 2 reward functions but 3 weights in config
- **WHEN** initializing trainer
- **THEN** raise ValueError at startup before training begins

### Requirement: LLM-only LoRA on last-K transformer blocks
- The system SHALL freeze ViT and Aligner components.
- LoRA SHALL be applied only to the last K transformer blocks of the LLM (K is configurable; default K=4).
- Verification logs SHALL confirm `freeze_vit=true`, `freeze_aligner=true`, `freeze_llm=false`, and proper Peft/SwiftModel wrapping.

#### Scenario: Vision-frozen configuration sanity check
- **GIVEN** GRPO training initialization
- **WHEN** preparing the model
- **THEN** logs show `freeze_vit=true`, `freeze_aligner=true`, `lora_last_k=4`, and `modules_to_save` reflects LLM-only tuning

#### Scenario: Last-K block targeting
- **GIVEN** a model with 32 transformer blocks and K=4
- **WHEN** applying LoRA
- **THEN** only blocks 28-31 have LoRA adapters

### Requirement: Minimal reward set (v1)
- Initial version SHALL enable only:
  - **Label reward**: Matches "通过"/"不通过" in line 1 against `label`
  - **Format reward**: Validates strict two-line structure
- Consistency rewards (e.g., summary alignment) are deferred to v2.
- Reward weights SHALL be configurable (default: label=1.0, format=0.2).

#### Scenario: Label reward computation
- **GIVEN** completion line 1 is "通过" and label is "pass"
- **WHEN** computing label reward
- **THEN** reward is positive (e.g., 1.0)

#### Scenario: Format reward for valid structure
- **GIVEN** completion has exactly 2 lines with correct prefixes
- **WHEN** computing format reward
- **THEN** reward is positive (e.g., 0.2)

#### Scenario: Format reward for invalid structure
- **GIVEN** completion has 3 lines or missing line 2
- **WHEN** computing format reward
- **THEN** reward is 0

### Requirement: Dry-run validation
- The system SHALL provide a minimal runnable example that loads base model and optional adapters, dataset path, registers rewards, and runs a short training dry-run.
- Dry-run SHALL complete 1-2 steps with a tiny dataset; trainer produces reward logs and writes completions.jsonl.

#### Scenario: Quick smoke test
- **GIVEN** a tiny dataset with 4 samples
- **WHEN** running dry-run with 2 steps
- **THEN** training completes without errors, logs show reward values, and completions.jsonl is written

<!-- Phase 4: Integration & Deployment (Future) -->

### Requirement: End-to-end pipeline orchestration
- The system SHALL support running Stage-A inference → Stage-B dataset conversion → GRPO training → Inference with trained adapter in a single workflow.

#### Scenario: Full pipeline execution
- **GIVEN** raw images in input directory
- **WHEN** running the full pipeline
- **THEN** Stage-A produces summaries, Stage-B converts to GRPO format, training runs, and final adapter is saved

### Requirement: Model deployment and inference
- The system SHALL support loading GRPO-trained LoRA adapters for inference on new groups.
- Inference SHALL use the same two-line output contract.

#### Scenario: Production inference with trained adapter
- **GIVEN** a trained GRPO adapter and new Stage-A summaries
- **WHEN** running Stage-B inference
- **THEN** model generates two-line verdicts using the adapter
