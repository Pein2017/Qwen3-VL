## ADDED Requirements

### Requirement: Toggleable TOON serialization
- Dense caption datasets SHALL expose a `toon_mode` configuration flag that switches assistant responses between legacy JSON and TOON table formats without altering upstream inputs.

#### Scenario: `toon_mode` disabled
- **GIVEN** a training run with `toon_mode: false`
- **WHEN** a dense caption sample is serialized
- **THEN** the assistant reply matches the existing JSON structure, including `object_{n}` keys and `line_points` for line geometries.

#### Scenario: `toon_mode` enabled
- **GIVEN** a training run with `toon_mode: true`
- **WHEN** a dense caption sample is serialized
- **THEN** the assistant reply is a TOON block `objs[N]{type,desc,xs}` where `type ∈ {0,1,2}` (bbox, quad, line), `desc` mirrors the object description, and `xs` lists normalized coordinates with no `line_points` field.

#### Scenario: TOON round-trip parsing
- **GIVEN** a TOON-formatted assistant reply produced with `toon_mode: true`
- **WHEN** the shared decoder converts it back into an object dictionary
- **THEN** the result contains the same `desc`, geometry type, and coordinates as the equivalent JSON payload (including derived `line_points = len(coords)/2` for lines).

### Requirement: TOON format compliance
- TOON outputs SHALL comply with the upstream TOON specification for headers, delimiters, and primitive rows while enforcing dense-caption geometry constraints.

#### Scenario: Header and delimiter structure
- **GIVEN** a sample serialized with `toon_mode: true`
- **WHEN** the assistant message is inspected
- **THEN** the header reads `objs[N]{type,desc,xs}` and may optionally include a delimiter suffix (`[N\t]` for tab) exactly as defined by the TOON spec, with comma as the default.

#### Scenario: Primitive row validation
- **GIVEN** a TOON row emitted by the builder
- **WHEN** the parser reads the row
- **THEN** the `type` column is an integer in `{0,1,2}`, `desc` is a quoted or unquoted string per TOON escaping rules, and every remaining token parses as a numeric primitive normalized according to `emit_norm`.

#### Scenario: Geometry constraint enforcement
- **GIVEN** an object marked with `type = 2` (line)
- **WHEN** the builder serializes coordinates
- **THEN** the number of numeric tokens after `desc` is even; otherwise serialization FAILS with a descriptive error before handing data to the template.

### Requirement: Prompt guidance for TOON mode
- Dense caption prompts SHALL provide TOON-specific instructions whenever `toon_mode` is true while preserving the JSON instructions when the flag is false.

#### Scenario: Reviewing prompt definitions
- **GIVEN** a maintainer inspects `src/config/prompts.py`
- **WHEN** they view the dense caption prompt logic
- **THEN** they find explicit guidance describing the TOON row format conditioned on `toon_mode`, and the legacy JSON guidance remains intact for the default path.

### Requirement: Documentation coverage for TOON outputs
- Project documentation SHALL explain how dense caption outputs differ between JSON and TOON modes, including geometry encoding and the absence of `line_points` in TOON.

#### Scenario: Reading dataset docs
- **GIVEN** a user reads `docs/DATA_AND_DATASETS.md`
- **WHEN** they reach the dense caption section
- **THEN** they see side-by-side examples of JSON (with `line_points`) and TOON outputs plus notes on the `toon_mode` toggle.

