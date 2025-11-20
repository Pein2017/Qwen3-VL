# data-augmentation Specification Delta

## ADDED Requirements
### Requirement: Augmentation Curriculum Scheduling
The system SHALL support a config-driven curriculum that adjusts augmentation bypass probability and numeric operator parameters over training progress without rebuilding the dataset.

#### Scenario: Progress-based phase selection with linear ramps
- **WHEN** the YAML defines `custom.augmentation.curriculum` with ordered phase boundaries keyed by `until_percent` (preferred, 0–1 or 0–100) or `until_step`
- **THEN** effective `bypass_prob` and each overridden numeric field (e.g., op `prob`, scalar, or numeric range bounds) are linearly interpolated from the previous phase target to the current phase target over that interval (percentages resolved using trainer-reported total steps), and held at the final targets after the last phase

#### Scenario: Consistent application across ranks and workers
- **WHEN** training runs with multiple ranks and dataloader workers
- **THEN** all workers apply identical effective curriculum parameters for a given step/epoch via shared state, avoiding drift from per-worker RNG or local phase computation

#### Scenario: Fail-fast validation
- **WHEN** the curriculum config is invalid (non-monotonic boundaries, unknown op names/fields, negative probabilities, inverted ranges)
- **THEN** training fails before start-up with a clear error message and does not proceed
