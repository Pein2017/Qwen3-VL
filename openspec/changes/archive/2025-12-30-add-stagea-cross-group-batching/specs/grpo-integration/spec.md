# grpo-integration — Stage-A Cross-Group Image Batching

## MODIFIED Requirements

### Requirement: Per-image inference and grouped aggregation
- The Stage-A engine SHALL accept an input directory containing multiple groups with arbitrarily many images (N ≥ 1).
- Supported extensions: {jpg, jpeg, png} (case-insensitive); files MUST be discovered deterministically with natural sort.
- The engine SHALL produce a Chinese single-line summary per image; implementations MAY batch multiple images per forward pass for throughput.
- A group aggregator SHALL construct a single JSONL record per group with `per_image` keys `image_{i}` aligned to the deterministic image order.
- Group ID SHALL be derived from filenames using the pattern `^(QC-[A-Za-z]+-[0-9]{8}-[0-9]+)` when present; otherwise the immediate subdirectory name is used.
- Output JSONL fields: `group_id`, `mission`, `label`, `images` (list), and `per_image` (dict by `image_{i}`).

#### Scenario: Basic directory with 3 images in one group
- **GIVEN** files: QC-TEMP-20250118-0015956-001.jpeg, -002.jpeg, -010.jpeg
- **WHEN** running the Stage-A engine
- **THEN** the JSONL has one record with `images` length 3 and `per_image` keys: `image_1`, `image_2`, `image_3`

#### Scenario: Multi-group directory processing
- **GIVEN** subdirectories with different group IDs
- **WHEN** processing recursively
- **THEN** each group produces one JSONL record with correctly sorted `image_{i}` keys

#### Scenario: Batched per-image inference preserves per-image semantics
- **GIVEN** a group containing N images in deterministic order
- **WHEN** Stage-A runs with `batch_size>1`
- **THEN** Stage-A produces exactly N per-image summaries aligned to the input order
- **AND** the output record contains one JSONL entry for the group with `image_{1..N}` coverage

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

### Requirement: One group per sample (text-only, no images)
- Each Stage-B sample SHALL represent one group; required fields:
  - `group_id`: string
  - `task_type`: string in {"BBU安装方式检查（正装）","BBU接地线检查","BBU线缆布放","挡风板安装检查"}
  - `group_label`: "通过" | "不通过"
  - `stage_a_summaries`: dict {`image_i`: 中文单行摘要}
  - `messages`: list[dict] (system in Chinese; user includes task focus and Stage-A summaries as plain text)
- Stage-B inputs SHALL NOT include any image data (text-only reasoning).

#### Scenario: Minimal valid sample
- **GIVEN** a JSONL record with messages referencing `image_1..image_k` summaries
- **WHEN** loading for GRPO training
- **THEN** the sample contains group_label, stage_a_summaries, and messages without images

#### Scenario: Stage-A to Stage-B conversion
- **GIVEN** a Stage-A output JSONL record with `per_image` summaries
- **WHEN** converting to Stage-B format
- **THEN** each group becomes one text-only sample with embedded summaries in the user message

## RENAMED Requirements
- 图片_{i} alignment and coverage (strict validation) → image_{i} alignment and coverage (strict validation)

## ADDED Requirements

### Requirement: Stage-A SHALL expose a batching mode switch for group-local vs cross-group batching.
Stage-A SHALL provide a CLI option to select batching strategy, defaulting to group-local batching for backward compatibility.
#### Scenario: Default behavior remains group-local
- **WHEN** Stage-A is run without an explicit batching mode flag
- **THEN** batching is applied only within a group (no cross-group accumulation)

### Requirement: Stage-A SHALL support an optional cross-group batching mode while preserving grouped outputs.
Stage-A SHALL provide an opt-in mode to batch images across multiple groups to better utilize GPU capacity, while preserving the group-level output contract (one record per group; per-image alignment and strict coverage).
#### Scenario: Cross-group batching fills GPU batch without changing group aggregation
- **GIVEN** two groups A and B, each with 2 images, and `batch_size=4`
- **WHEN** Stage-A runs in cross-group batching mode
- **THEN** Stage-A MAY execute one forward pass containing images from both groups
- **AND** Stage-A outputs two JSONL records (one for A, one for B)
- **AND** each record’s per_image mapping contains only summaries for that group’s images in deterministic order

#### Scenario: Cross-group batching preserves strict coverage validation
- **GIVEN** a group with N images and Stage-A runs in cross-group batching mode
- **WHEN** any per-image output is missing for that group
- **THEN** the engine SHALL raise ValueError and not emit a partial record for that group

### Requirement: Stage-A cross-group batching SHALL preserve streaming/error semantics and output order per rank.
Cross-group batching SHALL preserve Stage-A’s per-group emission contract: each group is written at most once, failures do not emit partial records, and later groups are not blocked indefinitely by earlier failures.
#### Scenario: Failure does not deadlock buffered output
- **GIVEN** groups A, B, C in discovery order
- **AND** group A fails (e.g., image decode error or empty summary)
- **WHEN** Stage-A runs in cross-group batching mode
- **THEN** Stage-A writes no record for A
- **AND** Stage-A still writes records for B and C (in that order) without waiting for A

#### Scenario: Mixed-size groups preserve output order
- **GIVEN** groups A (many images) then B (few images) in discovery order
- **WHEN** Stage-A runs in cross-group batching mode and batches contain images from both A and B
- **THEN** the JSONL record for A appears before the JSONL record for B in the per-rank output stream

### Requirement: Stage-A cross-group batching SHALL be rank-local and memory-bounded.
In distributed mode (`torchrun`), cross-group batching SHALL not mix work across ranks, and implementations SHALL keep at most `batch_size` images in-flight per rank.
#### Scenario: Distributed runs do not cross-rank mix groups
- **GIVEN** Stage-A runs under `torchrun` with `WORLD_SIZE>1`
- **WHEN** a rank executes cross-group batching
- **THEN** it batches only images from the groups assigned to that rank’s shard

#### Scenario: In-flight images are bounded
- **WHEN** Stage-A runs in cross-group batching mode with `batch_size=B`
- **THEN** the implementation keeps at most B images in-flight at a time per rank (no unbounded buffering of decoded images/tensors)
