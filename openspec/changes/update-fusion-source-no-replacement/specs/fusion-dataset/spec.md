# fusion-dataset (Delta: update-fusion-source-no-replacement)

## MODIFIED Requirements

### Requirement: Deterministic per-epoch resampling and shuffling
The fusion loader SHALL rebuild its schedule each epoch: compute each target quota as `round(len(target) * ratio_i)` with `ratio_i` defaulting to `1.0` (ratio < 1 downsamples, ratio > 1 upsamples with replacement), then sample each source by `round(source_ratio * N_target_total)` using either without-replacement draws when enabled or with-replacement draws otherwise, and finally shuffle with a deterministic seed; when without-replacement is enabled and the quota exceeds the source pool, the loader SHALL fall back to deterministic with-replacement sampling and surface this in epoch telemetry.

#### Scenario: Source without-replacement within pool size
- **WHEN** a source sets `sample_without_replacement: true` and its quota for the epoch is less than or equal to its pool size
- **THEN** the fusion loader shuffles the source indices deterministically for that epoch and takes the first `quota` indices (no duplicates within the epoch)
- **AND** target sampling and other sources remain unchanged.

#### Scenario: Source without-replacement quota exceeds pool
- **WHEN** a source sets `sample_without_replacement: true` but its computed quota exceeds its available pool size
- **THEN** the fusion loader falls back to deterministic with-replacement sampling for that source for the epoch
- **AND** the fallback is reflected in epoch-level telemetry/plan for observability.

#### Scenario: Default replacement retained
- **WHEN** a source does not set `sample_without_replacement`
- **THEN** the fusion loader samples that source with replacement exactly as before (deterministic, seeded) while still honoring target quotas and per-epoch shuffling.
