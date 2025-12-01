# Design: Target-only fusion policies

## Principles
- **Domain separation:** target domain drives augmentation, curriculum, and evaluation; sources remain clean regularizers.
- **Determinism:** caps apply only in training on sources; eval stays stable and uncapped.
- **Provenance everywhere:** both online loader and offline fused JSONLs carry `_fusion_domain` / `_fusion_source` / `_fusion_template`.
- **Packing isolation:** packers must group by domain (or stricter by source) to avoid mixing target+source sequences.

## Scope boundaries
- No change to ratio math (small ratios may round to 0).
- No new loss-weighting; CE remains uniform.

## Implementation sketch (for future coding)
1) Fusion loader: force `include_source_eval=False`; build caps only for `domain==source && split==train`; force `augmentation_enabled=False` for sources.
2) Add `pack_group_key` helper returning fusion domain; surface in telemetry.
3) Offline builder: annotate each record with `_fusion_domain/_fusion_source/_fusion_template` before writing fused JSONL.

## Test considerations
- Unit: target-only eval length; source val ignored even when provided.
- Unit: cap not applied to target; applied to source train only; deterministic sample after cap.
- Unit: packed batches contain single domain key.
- Integration: fused JSONL rows contain provenance fields matching online loader.
