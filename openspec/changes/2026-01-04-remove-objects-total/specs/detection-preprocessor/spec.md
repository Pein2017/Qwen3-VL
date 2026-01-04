# detection-preprocessor Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical converter contract (all datasets)
BBU/RRU converters SHALL follow the key=value description contract and JSON-string summary contract.

#### Scenario: Summary JSON schema keys
- **WHEN** a BBU or RRU converter writes summaries
- **THEN** the JSON string includes `dataset` and `统计`
- **AND** `统计` is a list of per-category objects each containing `类别` plus any observed attribute counts
- **AND** BBU summaries include an optional `备注` list of strings (may be empty or absent)
- **AND** RRU summaries MAY include `分组统计` (group id → count) and per-category `组` counts
- **AND** the JSON string is single-line and uses standard separators (`, ` and `: `)

