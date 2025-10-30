---
title: Anchor Dense Caption SFT with ms-swift GKD
author: Qwen3-VL
created: 2025-10-29
change-id: 2025-10-29-integrate-gkd-sft
status: in-progress
---

## Why
Recent dense-caption SFT runs collapse the language tower: models hallucinate outdoor Huawei 5G scenes for indoor rack photos. LoRA-only retuning is insufficient; we need to keep the student close to the base Qwen3-VL logits while still learning downstream captions.

## What Changes
- **GKD-backed SFT mode**: reuse ms-swift's Generalized Knowledge Distillation trainer so the base checkpoint stays as a frozen teacher.
- **Config overlays**: add YAML patterns that switch between vanilla SFT and GKD without refactoring stage configs.
- **KL telemetry**: log KL/CE terms per step to flag drift and provide quick health checks (`train/loss`, `train/sft_loss`, `train/kl_loss`, `train/token_accuracy`, plus `eval/*` mirrors).
- **Docs**: capture when to pick GKD vs vanilla SFT, and how to tune `beta`, `sft_alpha`, `seq_kd`; add forward-only KD guidance (seq_kd=false, lmbda=0.0, sft_alpha≈1.0, beta≈0.1) and note that generation knobs are ignored in that mode.

## Impact
- New spec: `sft-training` covering KL-anchored dense captioning.
- Code surfaces to touch:
  - `configs/` stage 2/3 overlays (GKD variants)
  - `docs/` dense caption training guide
  - Optional thin wrappers in `scripts/train.sh` or helper scripts to call ms-swift GKD
- Zero refactor inside `src/` unless telemetry plumbing requires a tiny callback.

## Validation Plan
1. Smoke run (<=2 epochs, small JSONL slice) with new GKD config; ensure training completes and KL stays finite.
2. Compare greedy captions on 50 indoor/outdoor QA samples before/after GKD; expect hallucination incidents to drop.
3. Ensure base language fluency intact: run text-only sanity set (existing Stage-3 summary eval) and confirm no regression >5%.
4. Update docs with usage instructions and parameter table.

## Open Questions
- Do we expose a pure KL-SFT mode (student logits vs teacher logits) alongside GKD? (Default plan: stick to ms-swift GKD; evaluate follow-up once telemetry proves stable.)
- Need to confirm teacher checkpoint format (merged base vs adapter). This proposal assumes merged base weights are available.
