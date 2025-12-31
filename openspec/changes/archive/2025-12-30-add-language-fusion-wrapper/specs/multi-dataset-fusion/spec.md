# multi-dataset-fusion (Delta: add-language-fusion-wrapper)

## ADDED Requirements

### Requirement: Text-only auxiliary sources
The fusion pipeline SHALL accept text-only JSONL datasets (no images/objects) as source domains without breaking detection targets.

#### Scenario: Chat source with per-epoch ratio
- **GIVEN** a fusion config whose sources include `dataset: chat` with a valid `train_jsonl` containing `messages` turns only
- **WHEN** FusionCaptionDataset builds the epoch schedule
- **THEN** the chat source participates using its declared `ratio` (sampled with replacement) even though it has no images or geometry fields.

#### Scenario: Prompt selection for chat sources
- **GIVEN** a chat source wrapper that declares `template: chatml` (or equivalent) and provides chat-style prompts
- **WHEN** a chat record is encoded
- **THEN** the loader applies the chat template’s system/user prompts (not the detection prompts) and does **not** inject image placeholders or geometry instructions.

#### Scenario: Pass-through pre-authored conversations
- **WHEN** a record in any source contains a `messages` array and no `images`/`objects`
- **THEN** the builder reuses those messages verbatim for encoding, keeping metadata intact, instead of synthesizing detection-style user/assistant payloads.
