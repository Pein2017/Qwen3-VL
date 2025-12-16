# multi-dataset-fusion (Delta: add-irrelevant-summary-source)

## ADDED Requirements

### Requirement: Irrelevant summary target stream
The fusion stack SHALL support mixing a small "irrelevant image" dataset as an additional **target** stream in summary mode, using the existing summary template.

#### Scenario: Irrelevant target is referenced in fusion config
- **WHEN** a fusion config includes a target entry:
  - `{name: irrelevant_summary, dataset: bbu, template: bbu_summary, mode: summary, train_jsonl: data/irrelevant_summary/train.jsonl, val_jsonl: data/irrelevant_summary/train.jsonl, ratio: 1}`
  - and explicitly sets `{augmentation_enabled: false, curriculum_enabled: false}`
- **THEN** the fusion loader includes it as a target dataset in `mode: summary` using the `bbu_summary` prompts/template
- **AND** its per-epoch target quota is computed from its own pool size: `quota = round(len(pool) * ratio)` (so `ratio: 1` yields each record once per epoch)
- **AND** augmentation and curriculum remain disabled for this entry regardless of global settings.

### Requirement: Irrelevant JSONL records remain canonical
The irrelevant summary JSONL records SHALL conform to the canonical detection JSONL contract even though summary-mode encoding ignores `objects`.

#### Scenario: Dummy full-frame bbox keeps contract compatibility
- **WHEN** a record contains exactly one image, a dummy full-frame bbox object, and summary text, e.g.:
  - `images: ["images/0001.jpeg"]`
  - `width: W`, `height: H`
  - `objects: [{"bbox_2d": [0, 0, W, H], "desc": "irrelevant"}]`
  - `summary: "无关图片"`
- **THEN** it passes the canonical JSONL validator
- **AND** it is eligible for fusion sampling and summary-mode template encoding, where the assistant target is the summary string.

### Requirement: Helper for irrelevant JSONL generation
The system SHALL provide a helper that builds the irrelevant summary JSONL from a folder of JPEGs with a 1:1 image-to-record mapping.

#### Scenario: Operator generates irrelevant JSONL
- **WHEN** an operator runs the helper against `data/irrelevant_summary/images/*.jpeg`
- **THEN** it emits `data/irrelevant_summary/train.jsonl` where each line references exactly one image (relative path), sets `summary` to `无关图片`, fills `width/height` from EXIF-aware image dimensions, and emits a single dummy full-frame bbox object with a non-empty `desc`
- **AND** output ordering is deterministic (sorted by path)
- **AND** unreadable or missing images are reported and skipped without terminating the run.
