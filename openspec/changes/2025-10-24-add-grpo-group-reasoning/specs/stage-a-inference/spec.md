## ADDED Requirements

### Requirement: Stage-A per-image inference and grouped aggregation
- The engine accepts an input directory that may contain multiple groups and arbitrarily many images (N ≥ 1).
- Supported extensions: {jpg, jpeg, png} (case-insensitive); files are discovered deterministically with natural sort.
- The engine runs inference one image at a time to produce a Chinese single-line summary per image.
- A group aggregator constructs a single JSONL record per group with 图片_{i} keys aligned to the deterministic image order.
- Group ID is derived from filenames using the pattern `^(QC-[A-Za-z]+-[0-9]{8}-[0-9]+)` when present; otherwise the immediate subdirectory name is used.
- Output JSONL fields: group_id, images (list), per_image (dict by 图片_{i}), raw_texts (list), clean_texts (list), timestamp.

#### Scenario: Basic directory with 3 images in one group
- Given files: QC-TEMP-20250118-0015956-001.jpeg, -002.jpeg, -010.jpeg
- When running the engine
- Then the JSONL has one record with images length 3 and per_image keys: 图片_1, 图片_2, 图片_3

### Requirement: 图片_{i} alignment and coverage (strict)
- The aggregator MUST assign 图片_{1..N} to match the sorted input order exactly.
- If any 图片_{i} index is missing or extra relative to N, the engine raises a ValueError and aborts the current group.

#### Scenario: Mismatch in 图片_{i}
- Given 2 input images for a group but per_image has keys 图片_1 and 图片_3
- Then the engine raises ValueError and does not write a partial record for that group

### Requirement: Prompting and decoding (per-image)
- Use the model’s native chat_template; no manual special tokens.
- Prompts and responses are in Chinese; prompt wording is pluggable and may be tailored later.
- Save both raw and cleaned text per image; aggregator assembles per_image mapping without requiring model-emitted 图片_{i}.
- All discovered images MUST yield a non-empty cleaned summary; otherwise the engine raises ValueError and aborts the group.

#### Scenario: Empty summary line
- Given an image that returns an empty string after cleaning
- Then the engine raises ValueError for that group and writes no record
