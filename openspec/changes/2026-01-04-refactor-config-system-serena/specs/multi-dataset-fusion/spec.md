# multi-dataset-fusion Spec Delta

## ADDED Requirements

### Requirement: Fusion config inheritance for dataset mixes
- Fusion config YAMLs SHALL accept top-level `extends` keys (string or list), resolved relative to the current file.
- When merging, `targets` and `sources` entries SHALL be merged by dataset name (matching `name`, falling back to `dataset` if `name` is missing).
- Override entries with new dataset names SHALL be appended to the merged list after base entries.
- An explicit empty list in an override SHALL replace the base list (e.g., to disable sources for GRPO-only configs).

#### Scenario: Resolution-specific overlay
- **GIVEN** a base fusion config with shared sources and an overlay that only changes `train_jsonl`/`val_jsonl` for `bbu` and `rru`
- **WHEN** `FusionConfig.from_file` loads the overlay
- **THEN** the resulting config retains base templates/ratios and updates only the JSONL paths.

#### Scenario: Overlay adds new dataset entry
- **GIVEN** a base fusion config with targets for `bbu`
- **WHEN** an overlay adds a new target entry for `rru`
- **THEN** the resulting config contains both `bbu` and `rru` targets.

#### Scenario: Empty sources override
- **GIVEN** a base fusion config with non-empty `sources`
- **WHEN** an overlay sets `sources: []`
- **THEN** the resulting config contains no sources.
