## MODIFIED Requirements

### Requirement: Stage-B SHALL generate group verdict candidates via prompt-only rollouts under a strict two-line binary contract.
Stage-B rollout prompts SHALL keep the system prompt static (two-line contract + guardrails, no domain block). Mission guidance (G0/S*/G*) and Stage-A summaries SHALL appear only in the user prompt.

#### Scenario: Guidance and summaries live in the user prompt
- **WHEN** Stage-B builds rollout messages for a ticket,
- **THEN** the system prompt SHALL include only the two-line verdict contract and guardrails (no domain block),
- **AND** the user prompt SHALL include guidance (G0/S*/G*) plus the Stage-A summary block for the ticket.
