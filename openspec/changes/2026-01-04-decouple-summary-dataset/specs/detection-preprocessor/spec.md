# detection-preprocessor Delta

## MODIFIED Requirements

### Requirement: Summary JSON schema keys
When a BBU or RRU converter writes summaries, the JSON string SHALL include `统计` and SHALL NOT include `dataset`. BBU summaries MAY include `备注` (list of strings); RRU summaries MAY include `分组统计` (group id → count).

#### Scenario: Summary JSON omits dataset
- **GIVEN** a BBU summary emitted by the converter
- **WHEN** the summary JSON is written
- **THEN** the JSON contains `统计` and optional `备注`
- **AND** the JSON does not contain `dataset`

#### Scenario: RRU group stats allowed
- **GIVEN** an RRU summary with group membership
- **WHEN** the summary JSON is written
- **THEN** the JSON may include `分组统计` with group counts
- **AND** the JSON does not contain `dataset`
