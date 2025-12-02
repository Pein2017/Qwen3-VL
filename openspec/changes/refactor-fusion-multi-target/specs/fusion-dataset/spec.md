# fusion-dataset (Delta)

## ADDED Requirements

### Requirement: Multi-target fusion config acceptance
The fusion loader SHALL accept a `targets` list (one or more entries) and treat legacy `target` as a one-element `targets`, enforcing unique dataset names across all targets and sources.

#### Scenario: Parse multi-target config
- **WHEN** a fusion config declares `targets: [bbu, rru]` plus sources `lvis` and `lang_chat`
- **THEN** the loader parses both targets as target-domain datasets, sources as auxiliary, and rejects any name collisions.

#### Scenario: Legacy single target
- **WHEN** a fusion config provides only `target: { ... }`
- **THEN** it is normalized to a single-element targets list without changing existing behavior.

### Requirement: Self-scaled target quotas per epoch
The fusion scheduler SHALL support per-target `ratio` values that scale each target by its own pool size: `quota_i = round(len_i * ratio_i)` with `ratio_i` defaulting to `1.0`; ratios below 1 downsample, ratios above 1 upsample with replacement, and indices are shuffled deterministically each epoch.

#### Scenario: Three targets scaled independently
- **GIVEN** target pool sizes 100, 200, 300 with ratios 0.5, 1.0, 1.5
- **WHEN** an epoch schedule is built
- **THEN** quotas are 50, 200, and 450 respectively (upsample with replacement for the third target)
- **AND** each target’s indices are shuffled deterministically using mixed seeds (global, epoch, per-target seed).

### Requirement: Source quotas keyed to total target quota
The fusion scheduler SHALL compute each source quota as `round(source.ratio * total_target_quota)` where `total_target_quota` is the sum of all target quotas for that epoch; sources are sampled with replacement using deterministic seeds.

#### Scenario: Source quota from combined targets
- **GIVEN** total target quota of 303 and a source ratio 0.1
- **WHEN** building the epoch schedule
- **THEN** 30 source samples are drawn (with replacement) using deterministic seeding, and sources remain augmentation- and curriculum-off with object caps applied when configured.

### Requirement: Multi-target evaluation coverage
The fusion eval loader SHALL iterate all target validation splits (concatenated, deterministic order), SHALL NOT include sources in eval, and SHALL NOT downsample targets unless an explicit eval limit is provided.

#### Scenario: Eval over two targets
- **WHEN** targets `bbu` and `rru` each provide `val_jsonl`
- **THEN** the eval dataloader length equals the sum of their val records, with no source samples included.

### Requirement: Compatibility and defaults
When no target ratios are provided, the scheduler SHALL default each target to `ratio=1.0` (full coverage), and legacy single-target configs SHALL remain valid without code changes to callers.

#### Scenario: No ratios given
- **WHEN** a multi-target config omits `ratio`
- **THEN** every target sample is included once per epoch (full coverage), and sources still use the total target count for their quotas.
