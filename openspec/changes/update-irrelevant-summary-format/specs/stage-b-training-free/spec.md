# stage-b-training-free Specification (Delta)

## ADDED Requirements

### Requirement: Stage-B prompts tolerate single-line irrelevant summaries
Stage-B prompt assembly SHALL describe irrelevant summaries without imposing a line-count requirement. Any summary containing the literal `无关图片` SHALL be treated as irrelevant whether it appears as a single-line summary or as the second line of a two-line summary.

#### Scenario: Single-line irrelevant summary
- **GIVEN** Stage-A summaries include a value that is exactly `无关图片`
- **WHEN** Stage-B prompt text is assembled
- **THEN** the prompt only states that `无关图片` marks an irrelevant image and does not require a two-line format.

#### Scenario: Two-line summary with irrelevant second line
- **GIVEN** a summary rendered as two lines where line 2 is `无关图片`
- **WHEN** Stage-B prompt text is assembled
- **THEN** the prompt treats the image as irrelevant and does not imply a format violation.
