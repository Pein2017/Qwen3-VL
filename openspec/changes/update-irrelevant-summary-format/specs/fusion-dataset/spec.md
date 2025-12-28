# fusion-dataset Specification (Delta)

## ADDED Requirements

### Requirement: Irrelevant summary stream uses single-line output via BBU prompt
Irrelevant-summary samples SHALL be identified by `metadata._fusion_source` (e.g., `irrelevant_summary`). For those samples, the summary prompt SHALL reuse the BBU summary template and SHALL instruct a single-line output exactly `无关图片` (no `<DOMAIN=...>, <TASK=...>` header). All non-irrelevant summary samples SHALL retain the two-line summary contract with the `<DOMAIN=...>, <TASK=...>` header line followed by a JSON summary line.

#### Scenario: Irrelevant summary source uses single-line output
- **GIVEN** a summary-mode sample whose `metadata._fusion_source` equals `irrelevant_summary`
- **WHEN** prompt resolution runs for that sample
- **THEN** the applied prompt is the BBU summary template
- **AND** the prompt specifies a single-line output exactly `无关图片` without a header line.

#### Scenario: Non-irrelevant summary keeps two-line contract
- **GIVEN** a summary-mode sample whose `metadata._fusion_source` is not `irrelevant_summary`
- **WHEN** prompt resolution runs for that sample
- **THEN** the prompt keeps the two-line summary output contract (`<DOMAIN=...>, <TASK=...>` + JSON line).

#### Scenario: Irrelevant pool listed under targets for full coverage
- **GIVEN** the fusion config lists `irrelevant_summary` under targets with `ratio=1.0`
- **WHEN** target quotas are computed
- **THEN** the loader schedules the full irrelevant pool (subject to rounding) like other targets
- **AND** each record still carries `metadata._fusion_source=irrelevant_summary` so downstream handling can treat it as source-like.
