# grpo-integration — Stage-A Sharding Modes (Per-Group vs Per-Image)

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

### Requirement: image_{i} alignment and coverage (strict validation)
- The aggregator MUST assign `image_{1..N}` to match the sorted input order exactly.
- If any `image_{i}` index is missing or extra relative to N, the engine SHALL raise a ValueError and abort the current group.
- All discovered images MUST yield a non-empty cleaned summary; otherwise the engine SHALL raise ValueError and abort the group.

#### Scenario: Empty summary validation
- **GIVEN** an image that returns an empty string after cleaning
- **THEN** the engine raises ValueError for that group and writes no record

## ADDED Requirements

### Requirement: Stage-A SHALL expose a sharding mode switch (per_group vs per_image).
Stage-A SHALL provide a CLI option `--sharding_mode` with values:
- `per_group` (default): shard work at group granularity; a group’s images are processed by a single rank.
- `per_image`: shard work at image granularity; a group’s images MAY be processed by multiple ranks.

#### Scenario: Default behavior remains per_group
- **WHEN** Stage-A is run without an explicit `--sharding_mode`
- **THEN** Stage-A uses `per_group`

### Requirement: Stage-A SHALL NOT provide backward-compatible aliases for removed batching modes.
Stage-A SHALL expose only `--sharding_mode` for controlling sharding/execution strategy and SHALL NOT accept legacy batching-mode flags for compatibility.

#### Scenario: Legacy batching flag is rejected
- **WHEN** Stage-A is invoked with a removed `--batching_mode` flag
- **THEN** Stage-A fails fast with a clear CLI validation error

### Requirement: Group sampling SHALL occur before image job flattening.
When `pass_group_number`/`fail_group_number` sampling is enabled, Stage-A SHALL sample at the group level before creating image jobs.

#### Scenario: Sampling preserves group counts in per_image mode
- **GIVEN** `pass_group_number=P` and `fail_group_number=F`
- **WHEN** Stage-A runs with `--sharding_mode per_image`
- **THEN** at most P pass groups and at most F fail groups are selected
- **AND** only images belonging to those selected groups are processed

### Requirement: Sharding modes define alignment/coverage equivalence, not text identity.
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

### Requirement: per_group mode SHALL avoid cross-group image batches.
In `per_group` mode, Stage-A SHALL batch only within a group and SHALL NOT form a forward-pass batch containing images from multiple groups.

#### Scenario: per_group batching is group-local
- **GIVEN** two groups A and B
- **WHEN** Stage-A runs with `--sharding_mode per_group` and `batch_size>1`
- **THEN** no forward pass contains both A and B images

### Requirement: per_image mode SHALL shard images across ranks and merge on rank 0.
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

### Requirement: Final Stage-A output ordering MAY vary.
Stage-A outputs (including merged outputs under distributed runs) MAY be written in any order; consumers SHALL NOT depend on JSONL record ordering for correctness.

#### Scenario: Merge order is not a correctness criterion
- **GIVEN** a fixed input directory and fixed sampling seed
- **WHEN** Stage-A runs with `--sharding_mode per_group` and `--sharding_mode per_image`
- **THEN** the set of output groups is identical
- **AND** record ordering differences are permitted

### Requirement: per_image merge SHALL mark incomplete groups as failed and continue.
In `per_image` mode, rank 0 merge SHALL treat any group with missing/failed image summaries as a group failure:
- the group SHALL NOT emit a partial JSONL record
- merge SHALL continue for other groups

#### Scenario: Missing image result causes group failure only
- **GIVEN** group A has 3 images but only image_1 and image_2 summaries are present at merge time
- **WHEN** Stage-A merges results
- **THEN** group A emits no record
- **AND** other complete groups are still written

### Requirement: per_image intermediate outputs SHALL be deleted by default and optionally retained.
In `per_image` mode, Stage-A SHALL delete intermediate per-rank per-image output files after a successful merge by default, and SHALL provide an option to retain them for debugging.

#### Scenario: Default cleanup of intermediate files
- **WHEN** Stage-A completes a `per_image` run successfully without a keep-intermediate option
- **THEN** intermediate per-image per-rank files are removed

#### Scenario: Keep intermediate files for debugging
- **WHEN** Stage-A runs with an explicit keep-intermediate option
- **THEN** intermediate per-image per-rank files are preserved after merge
