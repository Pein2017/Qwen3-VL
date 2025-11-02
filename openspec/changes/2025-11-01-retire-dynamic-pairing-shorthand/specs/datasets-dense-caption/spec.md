## REMOVED Requirements

### Requirement: Dynamic pairing dataset support
- The SFT data loaders SHALL NOT support combining multiple base records into a single user turn via dynamic pairing or `images_per_user_turn`.

#### Scenario: Config sets `images_per_user_turn: 2`
- **GIVEN** a training config provides `custom.images_per_user_turn: 2`
- **WHEN** the config loader validates inputs
- **THEN** validation FAILS with an error explaining that dynamic pairing is no longer supported and the field must be removed.

#### Scenario: Legacy `DynamicPairDataset` import
- **GIVEN** a module attempts to import `src.datasets.dynamic_pair.DynamicPairDataset`
- **WHEN** the import executes after this change
- **THEN** the import FAILS with a clear exception indicating the class has been removed.

## MODIFIED Requirements

### Requirement: Single-image turn assembly
- Dense caption datasets SHALL emit exactly one image per conversational turn and preserve deterministic ordering across epochs.

#### Scenario: Sampling from the training dataset
- **GIVEN** a dataset created from a JSONL file containing three records with images
- **WHEN** iteration runs for one epoch
- **THEN** each `__getitem__` call returns a single record with one image in the user message, and the dataset length equals the number of source records.

### Requirement: Minimal assistant response structure
- The assistant response payload SHALL consist of the object hierarchy only, omitting any top-level image grouping wrappers while retaining `object_{n}` keys and geometry metadata (`line_points`, coordinate arrays).

#### Scenario: Dense mode builder output
- **GIVEN** `JSONLinesBuilder` runs in dense mode for a record with two annotated objects (bbox + line)
- **WHEN** the assistant content is generated
- **THEN** the serialized JSON is `{ "object_1": {...}, "object_2": {...} }` with `line_points` retained and **no** `图片_{n}` keys present.

#### Scenario: Summary mode builder output
- **GIVEN** the dataset operates in summary mode with a `summary` string in the record
- **WHEN** the assistant message is produced
- **THEN** the content is the summary string (or an equivalent minimal JSON with a single summary field), and no `图片_{n}` placeholders appear in the output.

### Requirement: Prompt and template guidance
- Prompts, templates, and documentation SHALL instruct the model to respond using the minimal structure without `图片_{n}` wrappers, while emphasizing the continued use of `object_{n}` indices.

#### Scenario: Prompt review
- **GIVEN** a maintainer reads `src/config/prompts.py`
- **WHEN** they inspect the dense caption prompt definition
- **THEN** the instructions describe the expected minimal assistant schema without referencing `图片_{n}`.

