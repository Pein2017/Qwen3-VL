# Stage-B Runtime Guide

Technical runbook for the Stage-B prompt-only verdict loop. Stage-A summarization is covered separately in `./STAGE_A_RUNTIME.md`.

---

## Stage-B Runtime (group verdicts)

#### Stage-B Runner (`src/stage_b/runner.py`, `scripts/stage_b.sh`)

Purpose: Training-free、prompt-only verdict loop：ingest → rollout → selection → optional reflection，附带 mission guidance 更新。已移除 critic 模块与置信度/自洽度信号，输出为两行 Verdict/Reason。

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

- When `gpus` contains multiple devices, `scripts/stage_b.sh` auto-launches single-node `torchrun`; Stage-B runs **ticket-parallel rollout** across ranks while keeping **selection + reflection sequential on rank 0**.
- In multi-GPU mode, the model is replicated per rank (data-parallel); `model.device_map` is overridden to force single-GPU placement per rank (avoid accidental model-parallel sharding).
- Only rank 0 writes `{output.root}/{mission_name}/{output.run_name}/...` artifacts; other ranks are rollout workers.
- `GuidanceRepository` copies the global guidance file into `{output.root}/{mission_name}/{output.run_name}/guidance.json` (mission-scoped) so edits stay isolated until you manually promote them back.
- Shared Qwen3-VL model is reused by sampler and reflection; **no CriticEngine**。
- 生产部署约束：Stage‑A 摘要和 Stage‑B 判决在同一 Qwen3‑VL checkpoint 上运行（同一套权重/LoRA），通过不同的 prompt 实现任务切换，因此任何针对摘要的 SFT/LoRA 调整都必须兼顾 Stage‑B 的 rollout 推理质量。
- Reflection only ingests **gradient candidates** (ticket_key granularity): `label_match=False`, rollout contradictions (`pass`+`fail` mixed) / `low_agreement`, or `conflict_flag/needs_manual_review` signals. Stable-correct tickets are excluded from reflection context.
- Two-pass reflection:
  - **decision pass**: outputs `no_evidence_group_ids` = stop-gradient ticket_keys (given `gt_label` still unlearnable / cannot propose any auditable hypothesis).
  - **ops pass**: runs only on learnable groups and emits strict JSON operations + hypotheses with **strict evidence** (`operations[*].evidence` / `hypotheses[*].evidence` must be non-empty and only reference learnable `ticket_key`).
  - Stage-B enforces learnability closure (`L == E ∪ H`) with bounded retries (default: `reflection.retry_budget_per_group_per_epoch=2`) and mission-level cost caps (`reflection.max_calls_per_epoch`). Uncovered learnable groups are retried in smaller deterministic batches; exhausted budgets route to `need_review_queue.jsonl` (`reason_code=budget_exhausted`).
- `need_review_queue.jsonl` is the only **stop-gradient** human-review queue (not a low-agreement queue). Stop-gradient tickets MUST NOT drive guidance ops or rule hit/miss feedback.
- `stage_a_paths` must point to Stage-A JSONL files containing `mission`, `group_id`, `label` (`pass|fail`), and `per_image`; keys are normalized to `image_{n}`.
- Resubmissions are allowed: the same `group_id` may appear multiple times as long as `label` differs. Stage-B treats `(group_id, label)` as the unique ticket identity and emits `ticket_key = "{group_id}::{label}"` in outputs to avoid collisions.

##### Config Breakdown (`src/stage_b/config.py`)

- `model`: HF checkpoint path, dtype (`bfloat16` recommended), and device map.
- `sampler`: decode grid (temperature/top_p/max_new_tokens/stop) 与 `samples_per_decode`。提示现要求“两行输出”：`Verdict: 通过|不通过` + `Reason: <单行理由>`，不再包含 `Evidence_Positive/Negative`。
- `selection`: majority vote + temperature tie-break; uses vote strength, no confidence/self-consistency.
- `manual_review`: 低一致性阈值（`min_verdict_agreement`，mission configs 默认 0.67）用于记录 `low_agreement` 警示；人工复核入口统一为 need-review（`need_review_queue.jsonl`）。
- `reflection`: two-pass prompts (`decision_prompt_path`, `ops_prompt_path`), batch size, `max_operations`, hypothesis gating thresholds, retry budget + cost caps.
- `runner`: epochs, `per_rank_rollout_batch_size` (per-rank batch size; global effective batch = per_rank × WORLD_SIZE in ticket-parallel mode), and `logging_steps` (emit step-wise telemetry every N groups).
- `output`: Root/run_name plus mission subdirs for artifacts.
- `guidance`: Global seed file and snapshot retention count.

##### Outputs & Directory Layout

Stage‑B currently expects **exactly one mission per run**. Artifacts are written under `{output.root}/{mission_name}/{output.run_name}/`:
- `guidance.json` + `snapshots/`
- `trajectories.jsonl` — One entry per candidate with decode params, parsed verdict/reason, format flag and warnings (includes `ticket_key`, `gt_label`).
- `selections.jsonl` — Final verdict per ticket with `vote_strength`, guidance/reflection metadata, warnings (includes `ticket_key`, `gt_label`, `epoch`, `epoch_step`, `global_step`).
- `metrics.jsonl` — Step-wise `logging_steps` windows + per-epoch summary rows (both include/exclude manual review: `acc/fn/fp/n`).
- `need_review_queue.jsonl` — Human review queue (**stop-gradient**): tickets that are unlearnable after seeing `gt_label`. Entries include `ticket_key`, `gt_label`, `pred_verdict`, `reason_code` (e.g., `reflection_no_evidence_after_gt`, `budget_exhausted`), `reflection_id`, `reflection_cycle`, and optional `uncertainty_note`.
- `need_review.json` — Run-end deterministic aggregate of `need_review_queue.jsonl` for quick inspection.
- `failure_malformed.jsonl` — Parser/format/selection failures for prompt debugging (e.g., `format_error` / `no_candidates` / `no_valid_candidates` / `selection_error`).
- `reflection.jsonl` — Applied/attempted guidance updates with evidence ticket_keys; `reflection.proposal` includes `hypotheses`, `uncertainty_note`, and `no_evidence_group_ids` for need-review routing.
- `hypotheses.json` — Mission-scoped hypothesis pool (signature, support cycles, evidence union, status).
- `hypothesis_events.jsonl` — Append-only hypothesis lifecycle events (proposed/promoted).
- `group_report_delta.jsonl` — Optional step-wise “delta snapshots” for the most recent `logging_steps` window (per ticket: selection + candidate summaries + flags), for monitoring drift during a run.
- `group_report.jsonl` — Optional full consolidated report generated at run end (can be rebuilt offline via `python scripts/stage_b_group_report.py --run-dir ...`).

> Metrics note: `_compute_metrics` treats `pass` as positive class (`fp` = pred `pass` & gt `fail`; `fn` = pred `fail` & gt `pass`; `model_verdict=null` is counted based on gt). `include_manual_review` / `exclude_manual_review` is driven by the internal `in_manual_review` flag (selection warnings + hard failures), and is independent from the human review queue (`need_review_*`).

> Tip: raise `sampler.samples_per_decode` when you need multiple attempts per group under the same decode hyperparameters; drop it to `1` for deterministic sweeps.
> Configure multi-epoch sweeps via `runner.epochs`. The runner shuffles tickets using `(seed + epoch)` so repeated runs with the same seed stay reproducible.

##### Offline Audit
- `scripts/stage_b_conflict_audit.py --run-dir <output/run_name> [--mission <name>]`：列出重复 `conflict_flag=true` 但从未被反思操作覆盖的样本（常见于标签/摘要噪声或指导缺口），便于人工复核或回溯 Stage-A。

##### Manual Review & Failures

- 没有 CriticEngine；只有主模型生成严格两行输出（`Verdict/Reason`）。
- 两行协议解析失败/无可用候选/selection 报错等“硬故障”仅写入 `failure_malformed.jsonl` 与日志（不进入 need-review）。
- need-review（人工复核）由 runner 在反思 flush 边界执行路由：以 **decision pass** 输出的 `no_evidence_group_ids` 作为 stop-gradient 判定来源（严格 ticket_key 粒度）。未覆盖 learnable groups 会进入 bounded retry；耗尽 retry budget / cost cap 后以 `reason_code=budget_exhausted` 路由到 need-review。

##### Guardrails & Label Alignment

- mission-scoped fail-first 仍存在：当 Stage-A 摘要出现与当前 G0 相关的明确负项触发词时，可确定性覆盖为不通过，并重写 Reason（保持两行协议与无第三状态词）。
- Stage-A 负项证据提取按“中文逗号（，）分割的对象条目”做更细粒度的 mission 相关性判断；若某条目包含“只显示部分/无法判断/需复核”等软信号，则该条目内的负项不会进入确定性 fail-first 命中（避免把不确定当确定证据）。
- 推理 user prompt 会提供每张图的 `ImageN(obj=...)` 统计（由摘要中的 `×N` 求和得到），用于帮助模型推断全局图/局部特写图并进行多图协同判断。
