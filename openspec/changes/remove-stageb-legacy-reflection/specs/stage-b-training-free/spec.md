# stage-b-training-free Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage‑B SHALL generate group verdict candidates via prompt-only rollouts under a strict two-line binary contract.
Stage‑B MUST run **rule-search mode only**. Legacy reflection mode is removed and MUST NOT be supported in configs or runtime.

#### Scenario: Rule-search is the only supported Stage‑B mode
- **WHEN** Stage‑B loads a configuration for execution.
- **THEN** the configuration MUST specify `mode: rule_search` (or omit `mode` if defaults are removed in favor of rule-search).
- **THEN** legacy_reflection-specific configuration sections or code paths MUST be rejected/absent.

### Requirement: Stage‑B SHALL select one final verdict per ticket and export deterministic artifacts.
Stage‑B MUST export artifacts from rule-search runs only. Legacy reflection artifacts and outputs MUST NOT be generated.

#### Scenario: Stage‑B artifacts are produced by rule-search only
- **WHEN** Stage‑B completes a rule-search run.
- **THEN** artifacts are written under `{output.root}/{run_name}/{mission}/` according to rule-search behavior.
- **THEN** legacy_reflection artifacts and logs MUST NOT be produced.
