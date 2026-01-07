# stage-b-training-free Delta

## ADDED Requirements

### Requirement: Stage-B SHALL drop summary headers and pass through payloads
Stage-B summary ingestion SHALL ignore any `<DOMAIN=...>, <TASK=SUMMARY>` header line and forward the remaining payload as-is. No schema validation is enforced; JSON and non-JSON payloads are both allowed.

#### Scenario: Header dropped for summary parsing
- **GIVEN** a Stage-A summary with a two-line header + JSON payload
- **WHEN** Stage-B ingests the summary
- **THEN** the header line is discarded and only the payload line is forwarded downstream unchanged

#### Scenario: Dataset key tolerated
- **GIVEN** a summary payload containing a `dataset` key
- **WHEN** Stage-B ingests the summary
- **THEN** Stage-B passes it through unchanged without error

#### Scenario: Non-JSON payload tolerated
- **GIVEN** a summary payload that is not valid JSON
- **WHEN** Stage-B ingests the summary
- **THEN** Stage-B passes it through unchanged without error
