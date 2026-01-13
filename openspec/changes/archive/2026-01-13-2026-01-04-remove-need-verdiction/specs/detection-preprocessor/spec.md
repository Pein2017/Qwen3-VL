## ADDED Requirements
### Requirement: Converters SHALL reject review placeholders instead of emitting them
Converters SHALL not introduce or propagate review-state placeholders when handling detection data; uncertain remarks stay as free text, and any explicit review marker is treated as invalid input.
#### Scenario: Review markers are treated as invalid input
- **WHEN** a detection converter or summary builder encounters any review-state wording (legacy third-state markers) in source annotations or intermediate desc/summary strings
- **THEN** it MUST raise a validation error and stop the conversion
- **AND** it MUST NOT rewrite desc to include a review placeholder; remarks remain free-text but keep original positive/negative values
- **AND** downstream JSONL/output summaries MUST stay binary/observational only, with no review marker keys or values
