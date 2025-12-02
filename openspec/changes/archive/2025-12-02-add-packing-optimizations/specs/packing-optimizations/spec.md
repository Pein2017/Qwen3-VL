# packing-optimizations (Delta)

## ADDED Requirements

### Requirement: Rank0-only pack construction with broadcast
Packing SHALL build packs on rank0 only when `torch.distributed` is initialized, broadcasting pack metadata (`indices`, `lengths`, `group`, `group_domains`) to all ranks before iteration.

#### Scenario: DDP startup
- **WHEN** `training.packing: true` runs under DDP
- **THEN** only rank0 runs the pack-building pre-pass
- **AND** all ranks receive identical packs/domains via broadcast before dataloader iteration.

### Requirement: Epoch rebuild broadcast
Packing SHALL rebuild on rank0 at each epoch transition and re-broadcast updated packs so all ranks stay in sync with the dataset schedule.

#### Scenario: Epoch change
- **WHEN** `set_epoch` is invoked
- **THEN** rank0 rebuilds packs for the new schedule
- **AND** other ranks receive the updated packs before iteration resumes.

### Requirement: Opt-in cached-length packing (exact lengths)
Packing SHALL support an opt-in cached-length mode that uses precomputed exact `length` values during pack build, skipping augmentation/image loading, while leaving training-time augmentation unchanged.

#### Scenario: Cache hit
- **WHEN** cached-length mode is enabled and a valid cache is present
- **THEN** `_build_packs` uses cached lengths without calling dataset augmentation/encode
- **AND** training-time `__getitem__` still runs full augmentation per sample.

### Requirement: Cache validation and guardrails
Cached-length mode SHALL validate a hash/version derived from augmentation config, template id/version, and dataset fingerprint; on mismatch or missing cache it SHALL either fail with guidance or fall back to standard pack build per configured policy.

#### Scenario: Cache mismatch
- **WHEN** the cache hash does not match the current config/template/dataset
- **THEN** packing does not use the cache and responds according to policy (fail or fallback) with clear guidance.

### Requirement: Cache generation tooling (exact, no approximation)
The system SHALL provide a helper/CLI to generate exact per-sample lengths by running the full augmentation + template encode (`return_length=True`), outputting lengths plus the validation hash/version.

#### Scenario: Cache generation command
- **WHEN** the cache generation tool is run against a JSONL + config
- **THEN** it writes exact lengths and the associated hash/version for packing to validate and consume.
