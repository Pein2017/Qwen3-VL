# multi-dataset-fusion Specification (Delta)

## ADDED Requirements

### Requirement: Per-dataset summary label grouping override
The system SHALL allow fusion dataset entries to override summary label grouping behavior on a per-dataset basis.

#### Scenario: Fusion config overrides grouping for summary datasets
- **GIVEN** a fusion config entry in summary mode that sets `summary_label_grouping` to `true` or `false`
- **WHEN** the fused dataset is built for training
- **THEN** the summary label normalizer is applied (or skipped) for that dataset according to the override
- **AND** datasets without an override continue to follow `custom.summary_label_grouping`.
