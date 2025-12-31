# sft-training (Delta)

## ADDED Requirements

### Requirement: Deterministic cross-rank grouped metric sync
Grouped per-dataset metrics SHALL synchronize keys deterministically across ranks and rely on a single reduction at logging time (via `MeanMetric.compute()`), avoiding per-step double reduction while still exposing groups seen only on nonzero ranks.

#### Scenario: Small group only on nonzero rank
- **WHEN** a fusion group (e.g., `lvis` or `lang_chat`) appears only on rank>0 within a logging window
- **THEN** rank0 logs still contain that group's loss/accuracy metrics with correct values aggregated once

#### Scenario: No double reduction vs single GPU
- **GIVEN** the same tiny fusion dataset is run on 1 GPU and on 2 GPUs
- **WHEN** reading per-group loss/accuracy after one logging interval
- **THEN** the multi-GPU values match the single-GPU run within floating tolerance, showing that state/count were reduced exactly once

#### Scenario: Deterministic key ordering across ranks
- **WHEN** different ranks observe metric keys in different insertion orders
- **THEN** the sync step gathers the union and instantiates missing metrics in sorted order so collective calls remain aligned and cannot bleed values across metric names
