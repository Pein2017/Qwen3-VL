# Design: Optional no-replacement source sampling

## Key decisions
- **Flag granularity**: per-source boolean `sample_without_replacement`; default false to keep existing runs unchanged.
- **Online scheduler**: when flag true and `quota <= pool_size`, shuffle indices deterministically (seed mixed with epoch and optional source seed) and take the first `quota`. When `quota > pool_size`, fall back to existing with-replacement sampling to avoid depletion; log in epoch plan/telemetry for visibility.
- **Offline builder**: mirror the same rule so fused JSONL matches runtime loader for the same config/seed.
- **Determinism**: reuse existing seed-mixing helper for online; offline builder derives per-source RNG from base seed ^ source seed to keep parity.
- **Non-goals**: no change to target sampling, ratios, or augmentation/curriculum policies.
