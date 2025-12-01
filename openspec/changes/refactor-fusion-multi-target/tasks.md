[x] Inspect existing fusion specs (`openspec/specs/fusion-dataset`, `multi-dataset-fusion`) and current fusion code paths to confirm assumptions and conflicts.
[x] Extend `FusionConfig` parsing to accept `targets` list (legacy `target` alias), optional per-target `ratio`, and name-uniqueness validation.
[x] Update fusion scheduler to support multiple targets with ratio-balanced quotas per epoch; keep deterministic seeding and per-dataset policies.
[x] Keep source sampling tied to total target quota; preserve source-only aug/curriculum off and object-cap behavior.
[x] Adjust eval loader to concatenate all target val splits and exclude sources; ensure telemetry reflects multi-target counts.
[x] Refresh example config `configs/fusion/bbu_rru_lvis_coig.yaml` to the new `targets` schema with ratio illustration.
[x] Add tests: config parsing (legacy + new), quota math for multiple targets/ratios, source quota linkage, and a smoke loader iteration.
[x] Update docs (`UNIFIED_FUSION_DATASET.md`, `DATA_AND_DATASETS.md`) for schema, scheduling, and ratio balancing; run `openspec validate refactor-fusion-multi-target --strict`.
