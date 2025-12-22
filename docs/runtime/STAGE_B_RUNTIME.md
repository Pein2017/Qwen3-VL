# Stage-B Runtime Guide

Technical runbook for the Stage-B prompt-only verdict loop. Stage-A summarization is covered separately in `./STAGE_A_RUNTIME.md`.

---

## Stage-B Runtime (group verdicts)

#### Stage-B Runner (`src/stage_b/runner.py`, `scripts/stage_b.sh`)

Purpose: Training-free、prompt-only verdict loop：ingest → rollout → rule-search gating → guidance 更新。输出为两行 Verdict/Reason。

```bash
# Default debug config bundled with the repo
gpus=0 bash scripts/stage_b.sh

# Multi-GPU (single node): ticket-parallel rollout via torchrun
# Note: `runner.per_rank_rollout_batch_size` is per-rank (per-device); global effective batch = per_rank × WORLD_SIZE.
# Example: with 8 GPUs and per_rank_rollout_batch_size: 2, each global step processes up to 16 tickets (2 per rank).
gpus=0,1,2,3,4,5,6,7 bash scripts/stage_b.sh

# Use a production config (e.g., bbu_line)
config=bbu_line gpus=0 log_level=logging \
  bash scripts/stage_b.sh

# No-model audit of core logic (fast)
bash scripts/stage_b.sh smoke
```

- Stage‑B only supports `rule_search`: baseline rollout → proposer outputs candidate rules → A/B gate on train pool → accept only via metric gate.
- When `gpus` contains multiple devices, `scripts/stage_b.sh` auto-launches single-node `torchrun`; Stage-B runs **ticket-parallel rollout** across ranks while keeping rule-search proposing/gating on rank 0.
- In multi-GPU mode, the model is replicated per rank (data-parallel); `model.device_map` is overridden to force single-GPU placement per rank (avoid accidental model-parallel sharding).
- Only rank 0 writes `{output.root}/{mission_name}/{output.run_name}/...` artifacts; other ranks are rollout workers.
- `GuidanceRepository` copies the global guidance file into `{output.root}/{mission_name}/{output.run_name}/guidance.json` (mission-scoped) so edits stay isolated until you manually promote them back.
- Shared Qwen3-VL model is reused by sampler and reflection; **no CriticEngine**。
- 生产部署约束：Stage‑A 摘要和 Stage‑B 判决在同一 Qwen3‑VL checkpoint 上运行（同一套权重/LoRA），通过不同的 prompt 实现任务切换，因此任何针对摘要的 SFT/LoRA 调整都必须兼顾 Stage‑B 的 rollout 推理质量。
- Rule-search proposer uses reflection prompts for candidate rule generation; decision/ops passes are not executed in this mode.
- `stage_a_paths` must point to Stage-A JSONL files containing `mission`, `group_id`, `label` (`pass|fail`), and `per_image`; keys are normalized to `image_{n}`.
- Resubmissions are allowed: the same `group_id` may appear multiple times as long as `label` differs. Stage-B treats `(group_id, label)` as the unique ticket identity and emits `ticket_key = "{group_id}::{label}"` in outputs to avoid collisions.

##### Config Breakdown (`src/stage_b/config.py`)

- `model`: HF checkpoint path, dtype (`bfloat16` recommended), and device map.
- `rule_search`: proposer prompt + train/eval pools + metric gate (relative error reduction + bootstrap + max_changed_fraction) + lifecycle op gating (fp/acc improvement + max_fp_rate_increase). Eval-pool metrics are logged for audit (no veto).
- `reflection`: proposer prompt settings (decision/ops prompts required by schema but not executed in rule_search).
- `runner`: epochs, `per_rank_rollout_batch_size` (per-rank batch size; global effective batch = per_rank × WORLD_SIZE in ticket-parallel mode), and `logging_steps`.
- `output`: Root/run_name plus mission subdirs for artifacts.
- `guidance`: Global seed file and snapshot retention count.
- `domain_map` / `default_domain`: Mission → domain mapping for runtime domain packs (`bbu|rru`). Stage‑B appends the domain block as a read‑only system prompt section; missing/unknown domains raise errors before rollout.
- `stage_b_distillation`: optional post-convergence ChatML export (`distill_size`, `distill_seed`, `distill_temperature`).

##### Outputs & Directory Layout

Stage‑B currently expects **exactly one mission per run**. Artifacts are written under `{output.root}/{mission_name}/{output.run_name}/`:
- `guidance.json` + `snapshots/`
- `rule_candidates.jsonl` — each proposed op with train/eval metrics + gate decision + op metadata.
- `benchmarks.jsonl` — accepted-op history (train/eval baseline/after metrics + guidance step + op metadata).
- `rule_search_hard_cases.jsonl` — per-iteration hardest mismatches for manual tracing.
- `rule_search_candidate_regressions.jsonl` — per-candidate regressions (base-correct → candidate-wrong) to audit harmful rules.
- `distill_chatml.jsonl` — optional ChatML export after rule-search early stop (when distillation is enabled).

##### Offline Audit
- `scripts/postprocess_rule_search_hard_cases.py --path <run_dir>`：汇总 `rule_search_hard_cases.jsonl`，便于人工复核与规则追踪。

##### Manual Review & Failures

- 没有 CriticEngine；只有主模型生成严格两行输出（`Verdict/Reason`）。
- rule_search 不输出 need-review 队列；解析失败的样本在统计中按无效候选处理。

##### Guardrails & Label Alignment

- Deterministic fail-first guardrails are removed; rule-search guidance drives decision behavior.
- Stage-A 摘要中的“只显示部分/无法判断/需复核”等软信号仅作为提示，不作为硬触发词。
- 推理 user prompt 会提供每张图的 `ImageN(obj=...)` 统计（由摘要中的 `×N` 求和得到），用于帮助模型推断全局图/局部特写图并进行多图协同判断。
