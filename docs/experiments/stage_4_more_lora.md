# Stage 4: LoRA Count Control and Generalization Preservation

**Date**: October 28, 2025  
**Experiment ID**: `10-28/stage_4_more_lora`  
**Base Model**: Continue from best Stage 3 (vs Stage 2) where applicable

---

## Executive Summary

Goal: Validate whether increasing the number of LoRA targets (across LLM and Vision) actually improves performance, and find the threshold beyond which language capacity degrades and vision becomes blind/hallucinatory. We will control the count of LoRA-instrumented modules, not just learning rates, and establish clear early stop criteria.

---

## Context & Observations (from Stage 2 → Stage 3)

1) Continuing from the best Stage 3 checkpoint outperforms re-training from Stage 2, indicating more training (with a good schedule) is beneficial.  
2) Learning rates among LoRA modules were relatively non-critical; different LR assignments yielded similar trends/performance.  
3) Post-Stage 2 checkpoints exhibit significant loss of general language capacity and overfit to training corpus. Fewer trained LLM blocks reduce drift but still fail to correctly describe irrelevant images.  
4) Increasing the number of Vision LoRAs leads to blindness/hallucination, especially on images outside the training distribution. Identical WeChat screenshots show strong failure cases.  
5) Next: Control LoRA quantity (not the-more-the-better). Closely monitor when general capacity starts to drop and stop before that.

---

## Hypotheses

- H1: Increasing LoRA count beyond a small, top-layer subset causes rapid loss of language generality and increases visual hallucination/blindness.  
- H2: There exists a narrow, optimal LoRA footprint per tower (LLM, Vision) that balances specialization and generalization.  
- H3: Given LR insensitivity, controlling target quantity/placement (layers) will dominate outcomes over modest LR tweaks.  
- H4: Training continuation is helpful only while general capacity metrics remain stable; once drift begins, continued training harms robustness.

---

## Experimental Design

- Factor 1 (LLM LoRA count): last-0, last-2, last-4 layers.  
- Factor 2 (Vision LoRA count): last-0, last-2, last-4, last-6 blocks.  
- Control: Keep LoRA rank/alpha/dropout fixed (Stage 3 defaults) unless noted.  
- Initialization: Prefer continuing from the best Stage 3 checkpoint for continuity; include a Stage 2 restart baseline for contrast.  
- Schedule: Fixed epochs with conservative warmup; identical augmentation as in Stage 3 winners.

We will run a compact matrix emphasizing counts rather than LRs; a small LR sweep on the winning count may be run afterward if needed.

---

## Configuration Matrix (counts, not LRs)

All use `tuner.train_type: lora`, `target_modules: [all-linear]`, same LoRA hyperparams as Stage 3 winner (rank=16, alpha=32, dropout=0.1), `optimizer: multimodal`, packing on, WD=0.1, warmup_ratio=0.2. Aligner remains frozen unless explicitly listed in `modules_to_save` (same as Stage 2/3).

- A0 (Baseline freeze): LLM last-0 + Vision last-0  
  - `freeze_llm: true`, `freeze_vit: true`  
  - Validates data/metrics pipeline; should preserve language capacity; expects weaker task fit.

- A1: LLM last-2 + Vision last-0  
  - `freeze_llm: false`, `freeze_vit: true`  
  - Regex (4B): `language_model.layers.(34|35).(self_attn.(q_proj|k_proj|v_proj|o_proj)|mlp.(gate_proj|up_proj|down_proj))`

- A2: LLM last-4 + Vision last-0  
  - Regex (4B): `language_model.layers.(32|33|34|35).(self_attn.(q_proj|k_proj|v_proj|o_proj)|mlp.(gate_proj|up_proj|down_proj))`

- B1: LLM last-0 + Vision last-2  
  - `freeze_llm: true`, `freeze_vit: false`  
  - Regex (4B): `visual.blocks.(22|23).(attn.(qkv|proj)|mlp.(linear_fc1|linear_fc2))`

- B2: LLM last-0 + Vision last-4  
  - Regex (4B): `visual.blocks.(20|21|22|23).(attn.(qkv|proj)|mlp.(linear_fc1|linear_fc2))`

- B3: LLM last-0 + Vision last-6  
  - Regex (4B): `visual.blocks.(18|19|20|21|22|23).(attn.(qkv|proj)|mlp.(linear_fc1|linear_fc2))`

- C1: LLM last-2 + Vision last-2  
- C2: LLM last-2 + Vision last-4  
- C3: LLM last-2 + Vision last-6  

- D1: LLM last-4 + Vision last-2  
- D2: LLM last-4 + Vision last-4  
- D3: LLM last-4 + Vision last-6  

Recommendation: prioritize A1, A2, B2, B3, C2, C3; defer D* unless capacity allows, since Stage 3 suggested heavier LLM LoRA exacerbates language drift.

Learning rates: keep Stage 3 defaults (`learning_rate: 1e-4`, `vit_lr: 3e-4`, `aligner_lr: 1e-4`) initially, given observed LR insensitivity. Adjust only if clear instability occurs.

---

## Metrics & Validation

We will track both task fit and general capacity to detect drift/hallucination early:

- Task Fit
  - Eval loss (primary) on validation split
  - Train/eval loss curves; generalization gap

- Language Capacity & Robustness
  - Irrelevant-image pass rate: fraction of WeChat-like or unrelated images correctly labeled as “无关图片” (from `SYSTEM_PROMPT[_SUMMARY]` prior rule)
  - Hallucination rate on out-of-distribution screenshots (identical conversation screenshots, unseen UI images)
  - Free-form Chinese caption sanity (demo prompt: “请用自然语言描述一下这些图片”)

- Vision Robustness
  - Blindness incidents: fraction of OOD images where the model outputs empty/near-empty or unrelated content
  - Consistency across crops: run a small crop jitter set and compare outputs for stability

Instrumentation
- Use `demo/demo.py` with an OOD/irrelevant mini-benchmark list (WeChat screenshots + distractors) and log pass/hallucination outcomes per checkpoint.
- Keep a fixed eval slice for quantitative eval loss tracking.

---

## Early Stop & Safety Criteria

Stop training for a run if any of the following triggers fires (checked every eval step/between epochs):

- Irrelevant-image pass rate drops >20% from baseline (A0) or below 80% absolute for two consecutive evals.  
- Hallucination rate rises >15% absolute from baseline.  
- Generalization gap (eval − train) widens >0.06 over its best historical value and continues to worsen across two eval intervals.  
- Free-form caption sanity visibly degrades (manual spot-check on 5 fixed OOD images) twice in a row.

Checkpoint the best eval loss subject to the constraint that irrelevant-image pass ≥ baseline−10% and hallucination ≤ baseline+10%.

---

## Launch Plan

- Base configs to derive from:
  - Stage 2: `/data/Qwen3-VL/configs/stage_2_llm_lora.yaml`
  - Stage 3 winner: `/data/Qwen3-VL/configs/stage_3_vision_last6_lora.yaml`

- Example command (replace with each variant’s config file):
```bash
conda run -n ms bash /data/Qwen3-VL/scripts/train.sh \
  config=/abs/path/to/your_stage_4_variant.yaml \
  gpus=0
```

- Logging: ensure `eval_strategy: steps`, frequent `save_steps`, and TB logging enabled; add a small callback to compute irrelevant-image and hallucination rates on the OOD mini-benchmark.

---

## Expected Outcomes

- A clear Pareto frontier of LoRA counts that balance task fit and generalization.  
- Confirmation that “more LoRA” is not monotonically better; likely winners cluster near LLM last-2 and Vision last-4/last-6.  
- Concrete stop thresholds to prevent language drift and visual blindness.

---

## Next Steps (after Stage 4)

- Narrow sweep around winners: vary LoRA ranks (8/16/32) and modest LR for stability.  
- Try progressive unfreezing: (Vision last-2 → last-4 → last-6) only if metrics stay within guardrails.  
- If language capacity remains fragile, explore smaller LLM LoRA (last-1) and strengthen irrelevant detection via prompt or classifier gating.
