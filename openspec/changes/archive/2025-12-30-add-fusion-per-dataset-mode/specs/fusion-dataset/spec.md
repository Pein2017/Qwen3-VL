## MODIFIED Requirements

### Requirement: Single-template unified fusion with prompt priority
The fusion loader SHALL reuse a single template instance and select user/system prompts per sample based on that sample’s dataset **and** mode using the priority `default < domain < dataset`, with system prompt injection restored after encoding.

#### Scenario: Mode-aware prompt selection
- **WHEN** encoding a fused sample
- **THEN** dense samples follow `default < domain < dataset` prompt priority, while summary samples use the configured summary prompt for their dataset/template
- **AND** the template system prompt is restored after encoding to avoid cross-sample contamination.

## ADDED Requirements

### Requirement: Per-dataset mode selection with strict validation
Fusion datasets SHALL honor per-dataset mode declarations and validate records accordingly, while falling back to the run-level default only when a dataset omits `mode`.

- **WHEN** a fusion config entry declares `mode ∈ {dense, summary}` (or `use_summary` alias)
- **THEN** the loader SHALL resolve the sample’s mode from the dataset’s declaration, falling back to the global `custom.use_summary` only when the dataset omits `mode`
- **AND** summary mode SHALL require each record to carry a non-empty `summary` string, while dense mode SHALL require at least one object with valid geometry
- **AND** mixing summary and dense datasets in the same epoch SHALL be supported without schema errors.

#### Scenario: Mixed-mode fusion run
- **GIVEN** a fusion config whose target dataset sets `mode: summary` and whose source dataset sets `mode: dense`
- **WHEN** the fusion loader builds a batch
- **THEN** target samples use summary prompts and summary validation, source samples use dense prompts and object/geometry validation
- **AND** per-sample telemetry/debug output records the dataset name, resolved mode, and prompt source.
