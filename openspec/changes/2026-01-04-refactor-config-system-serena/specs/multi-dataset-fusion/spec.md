# multi-dataset-fusion Spec Delta

## ADDED Requirements

### Requirement: Fusion config inheritance for dataset mixes
- Fusion config YAMLs SHALL accept top-level `extends`/`inherit` keys (string or list), resolved relative to the current file.
- When merging, `targets` and `sources` entries SHALL be merged by dataset name (matching `name`, falling back to `dataset` if `name` is missing).
- An explicit empty list in an override SHALL replace the base list (e.g., to disable sources for GRPO-only configs).

#### Scenario: Resolution-specific overlay
- **GIVEN** a base fusion config with shared sources and an overlay that only changes `train_jsonl`/`val_jsonl` for `bbu` and `rru`
- **WHEN** `FusionConfig.from_file` loads the overlay
- **THEN** the resulting config retains base templates/ratios and updates only the JSONL paths.

#### Scenario: Empty sources override
- **GIVEN** a base fusion config with non-empty `sources`
- **WHEN** an overlay sets `sources: []`
- **THEN** the resulting config contains no sources.
