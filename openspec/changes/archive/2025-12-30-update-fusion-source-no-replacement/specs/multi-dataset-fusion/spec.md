# multi-dataset-fusion (Delta: update-fusion-source-no-replacement)

## MODIFIED Requirements

### Requirement: Offline fusion builder (static ratios)
The offline fusion builder SHALL mix datasets according to fixed per-source ratios computed once per epoch/configuration, honoring per-source policies including optional `sample_without_replacement`; when enabled and the quota is within the source pool, the builder SHALL draw unique samples (no duplicates) using a deterministic shuffle, and when the quota exceeds the pool it SHALL fall back to deterministic with-replacement sampling and surface the fallback in its logs/telemetry. Targets continue to follow the existing ratio-based sampling rules (with replacement when ratios upsample).

#### Scenario: Offline source without-replacement within pool size
- **WHEN** a fusion config enables `sample_without_replacement: true` for a source and its offline quota is less than or equal to the pool size
- **THEN** the offline builder writes each sampled record at most once in the fused JSONL for that epoch/config, using a deterministic shuffle before taking the quota.

#### Scenario: Offline source without-replacement quota exceeds pool
- **WHEN** the same flag is set but the computed quota exceeds the source pool
- **THEN** the builder reverts to deterministic with-replacement sampling for that source while still emitting logs/telemetry that the fallback occurred.

#### Scenario: Default replacement retained (offline)
- **WHEN** a source does not enable `sample_without_replacement`
- **THEN** the offline builder samples that source with replacement (as before) while respecting per-source ratios and deterministic seeding.
