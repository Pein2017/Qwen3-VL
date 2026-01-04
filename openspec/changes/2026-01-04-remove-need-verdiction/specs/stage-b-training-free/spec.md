## ADDED Requirements
### Requirement: Stage-B SHALL fail fast on review-state inputs
Stage-B SHALL treat any third-state wording as invalid input rather than sanitize or downgrade it; only binary verdicts remain allowed end-to-end.
#### Scenario: Reject review markers in Stage-A summaries or rollout candidates
- **WHEN** Stage-B ingestion, rollout parsing, or reflection sees any review marker (e.g., legacy third-state wording such as 待定、证据不足) in Stage-A summaries, verdict lines, or reasons
- **THEN** Stage-B MUST treat the ticket as invalid input and raise a validation error before decoding candidates
- **AND** Stage-B MUST NOT attempt to sanitize or rewrite the review marker; it simply rejects the sample
- **AND** Stage-B smoketests and unit tests MUST cover this rejection path to ensure no third-state plumbing remains
