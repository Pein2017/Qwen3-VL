# multi-dataset-fusion Delta (update-fusion-target-only-policies)

## ADDED Requirements
### Requirement: Provenance metadata in fused outputs
The offline fusion builder SHALL emit fused JSONL records annotated with `_fusion_domain`, `_fusion_source`, and `_fusion_template` fields consistent with the online fusion loader so downstream packers and debuggers can distinguish target vs source samples without extra configuration.

#### Scenario: Fused JSONL carries provenance
- **WHEN** the offline fusion builder materializes a fused train JSONL from a fusion config with target and auxiliary sources
- **THEN** every written record contains `_fusion_domain`, `_fusion_source`, and `_fusion_template` metadata matching its originating dataset
- **AND** consumers can group or filter fused records by these fields without inferring provenance from file paths or dataset order.
