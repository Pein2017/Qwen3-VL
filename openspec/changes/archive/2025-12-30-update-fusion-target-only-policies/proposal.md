# Change Proposal: update-fusion-target-only-policies

## Why
- Eval metrics should reflect the downstream (target) domain only; including auxiliary sources skews metrics and increases variance.
- Augmentation/curriculum should be confined to the target domain; sources act as regularizers and must remain clean.
- Object caps are needed only to bound long-tail source annotations during training; targets must remain uncapped, especially in eval.
- Offline fused JSONLs must carry the same provenance metadata as the online loader to keep downstream telemetry and debuggers aligned.

## What
- Update fusion-dataset spec to enforce target-only evaluation, target-only augmentation/curriculum, and train-only source caps.
- Add a domain-aware provenance requirement so records always include `_fusion_domain`/`_fusion_source` for downstream attribution (even with padding-only batching).
- Update multi-dataset-fusion spec to require offline fusion builder to emit provenance metadata identical to the online loader.

## Impact
- Behavioral change: source validation will be disallowed; augmentation/curriculum no longer honored for sources even if configured.
- Minor contract change: fused JSONL outputs must include fusion metadata for telemetry/auditing; batch construction remains padding-only.
- Compatible with existing configs; failing only where pipelines relied on source eval or source augmentation.
