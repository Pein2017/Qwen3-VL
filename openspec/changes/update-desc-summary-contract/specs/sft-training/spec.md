# sft-training Spec Delta (update-desc-summary-contract)

## ADDED Requirements

### Requirement: Geometry JSON spacing stability
The conversation builder SHALL serialize assistant JSON with space-separated separators to preserve tokenizer distribution for geometry arrays.

#### Scenario: Geometry arrays retain spaces
- **WHEN** the JSONLinesBuilder renders assistant JSON for geometry payloads
- **THEN** JSON separators use `", "` and `": "` (spaces preserved)
- **AND** geometry arrays (bbox_2d/poly/line) are serialized with spaces between numeric elements

