# data-augmentation Specification Delta — add-multi-object-line-copy-paste

## ADDED Requirements

### Requirement: Background-aware small-object zoom and copy-paste
The augmentation system SHALL support an extended small-object zoom-paste operator that can place multiple copies of small objects into low-coverage background regions while preserving existing behavior by default.

#### Scenario: Default small_object_zoom_paste behavior remains unchanged
- **WHEN** the YAML config uses `small_object_zoom_paste` without new parameters (no background grid, max_copies_per_target=1, source_mode="local")
- **THEN** the operator selects at most `max_targets` small objects from the current image, crops a local context patch for each, applies a single zoom and uniform random placement per target, and appends duplicate geometries without changing existing output ordering or counts beyond the documented duplicates.

#### Scenario: Background-aware placement into low-coverage cells
- **WHEN** `small_object_zoom_paste` is configured with `placement_mode="background"` and a non-zero `grid_rows`/`grid_cols`
- **AND** the image contains at least one small-object candidate and at least one grid cell whose object coverage is below `background_max_coverage`
- **THEN** the operator computes a coarse occupancy grid using existing geometry coverage utilities
- **AND** chooses paste locations such that patches land preferentially in low-coverage cells while respecting IoU/overlap thresholds with existing geometries
- **AND** the resulting duplicates remain fully inside the image bounds with valid geometries.

#### Scenario: Multiple copies per small object with per-image cap
- **WHEN** `max_copies_per_target > 1` and `max_total_pastes` is configured
- **THEN** for each selected small object the operator MAY paste up to `max_copies_per_target` duplicates at distinct locations that pass overlap checks
- **AND** the operator SHALL not create more than `max_total_pastes` duplicates in total for the image
- **AND** originals are always preserved and duplicates are appended after originals in deterministic selection order.

### Requirement: Cross-image patch bank for small objects, clusters, and lines
The augmentation system SHALL provide an optional in-memory patch bank per dataloader worker so PatchOps can reuse small-object, cluster, and line patches across images under deterministic control.

#### Scenario: Bank is opt-in and capacity-limited
- **WHEN** a copy-paste PatchOp is configured with `source_mode="local"` or `bank_add_prob=0.0`
- **THEN** it SHALL ignore the patch bank and operate solely on patches derived from the current image
- **AND** the system SHALL behave identically to the bank-free implementation.

#### Scenario: Bank accumulation under worker-local RNG
- **WHEN** a PatchOp is configured with `source_mode` set to `"bank"` or `"mixed"` and `bank_add_prob > 0`
- **THEN** the operator MAY push patches derived from the current sample into a worker-local bank up to `bank_capacity`
- **AND** selections of which patches to add and later sample SHALL depend only on the RNG passed into the augmentation pipeline for that worker so that runs with the same seed are deterministic.

#### Scenario: Bank reuse for cross-image copy-paste
- **WHEN** the bank contains patches of a given kind (e.g., small objects, clusters, or line segments)
- **AND** a PatchOp is configured with `source_mode="bank"` or `"mixed"`
- **THEN** the operator MAY select some candidates from the bank instead of or in addition to local candidates
- **AND** pasted patches MUST still obey all IoU, coverage, and in-bounds constraints defined for the operator.

### Requirement: Object cluster copy-paste augmentation
The augmentation system SHALL provide a cluster-level PatchOp that copies small multi-object clusters and pastes them into background regions while keeping geometries consistent.

#### Scenario: Cluster construction around seed objects
- **WHEN** `object_cluster_copy_paste` is enabled in YAML with configured `min_objects_in_cluster`, `max_objects_in_cluster`, and `cluster_radius_px`
- **AND** the image contains at least one seed object that satisfies the cluster construction rules
- **THEN** the operator SHALL build one or more clusters by expanding a seed AABB by `cluster_radius_px` and collecting nearby objects whose AABBs intersect this region above a configured IoU threshold
- **AND** it SHALL discard clusters that exceed the configured max size or object-count limits.

#### Scenario: Cluster patch copy-paste into background
- **WHEN** a valid cluster is constructed and `placement_mode="background"` is configured for the operator
- **THEN** the operator SHALL crop a patch covering the cluster plus `cluster_context` pixels of padding, paste this patch into low-coverage grid cells chosen via the occupancy helpers, and apply a shared affine (scale+translate) to all cluster geometries via the existing geometry transform utilities
- **AND** originals are retained, duplicates are appended after originals, and geometries for both remain valid and in-bounds.

### Requirement: Line segment copy-paste augmentation
The augmentation system SHALL provide a PatchOp that duplicates cable-like line geometries as visually recognizable segments, with mild transforms that preserve curvature and completeness.

#### Scenario: Line segment extraction within length bounds
- **WHEN** `line_segment_copy_paste` is enabled with `min_line_length`, `max_line_length`, `min_segment_length`, and `max_segment_length`
- **AND** an input object has a `line` geometry whose total length falls within `[min_line_length, max_line_length]`
- **THEN** the operator SHALL select either the full line or a contiguous subsegment of length within `[min_segment_length, max_segment_length]` by walking along the polyline and interpolating between vertices as needed
- **AND** the chosen segment SHALL be used to define the patch AABB before adding `line_context` padding.

#### Scenario: Line patch copy-paste with preserved curvature
- **WHEN** a line or segment patch is extracted and a scale range near 1.0 (e.g., [0.9, 1.1]) is configured
- **THEN** the operator SHALL construct a uniform scale-about-center plus translation affine and apply it to the line geometry using existing transform utilities
- **AND** the resulting pasted line SHALL remain fully inside the canvas after clipping, preserve its qualitative curvature, and remain visually recognizable as a cable-like object.

#### Scenario: Controlled line overlap with existing objects
- **WHEN** a new line segment placement would cause the line's buffered AABB to exceed a configured IoU threshold with existing bbox/poly geometries or existing lines
- **THEN** the operator SHALL reject that placement and retry up to a configured number of attempts
- **AND** if no valid placement is found, the operator SHALL leave the input images/geometries unchanged for that candidate line.

## MODIFIED Requirements

### Requirement: PatchOp determinism, ordering, and telemetry invariants
PatchOps SHALL preserve deterministic behavior and ordering even when using background-aware placement and patch banks.

#### Scenario: Deterministic multi-copy small_object_zoom_paste
- **WHEN** `small_object_zoom_paste` is configured with `placement_mode="background"`, `max_copies_per_target > 1`, and a fixed RNG seed
- **THEN** repeated runs on the same record within the same worker SHALL select the same source objects, produce the same number of duplicates, and place them at the same locations in the same order, regardless of internal occupancy-grid or bank state.

#### Scenario: Deterministic cluster and line copy-paste
- **WHEN** `object_cluster_copy_paste` or `line_segment_copy_paste` is applied twice to the same record with the same RNG seed and identical patch-bank contents
- **THEN** the same clusters or line segments SHALL be selected and pasted at the same locations, originals are all retained, and duplicates appear after originals in consistent order, with crop telemetry fields left unset/empty.

