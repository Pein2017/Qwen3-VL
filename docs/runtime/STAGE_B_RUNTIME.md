# Stage-B Runtime Guide

Technical runbook for the Stage-B prompt-only verdict loop. Stage-A summarization is covered separately in `./STAGE_A_RUNTIME.md`.

---

## Stage-B Runtime (group verdicts)

#### Stage-B Runner (`src/stage_b/runner.py`, `scripts/stage_b_run.sh`)

Purpose: Training-free、prompt-only verdict loop：ingest → rollout → selection → optional reflection，附带 mission guidance 更新。已移除 critic 模块与置信度/自洽度信号，输出为两行 Verdict/Reason。

```bash
# Default debug config bundled with the repo
gpus=0 bash scripts/stage_b_run.sh

# Use the production run config
config=configs/stage_b/run.yaml gpus=0 log_level=logging \
  bash scripts/stage_b_run.sh
```

- `GuidanceRepository` copies the global guidance file into `{output.root}/{output.run_name}/{mission}/guidance.json` so edits stay isolated until you manually promote them back.
- Shared Qwen3-VL model is reused by sampler and reflection; **no CriticEngine**。
- 生产部署约束：Stage‑A 摘要和 Stage‑B 判决在同一 Qwen3‑VL checkpoint 上运行（同一套权重/LoRA），通过不同的 prompt 实现任务切换，因此任何针对摘要的 SFT/LoRA 调整都必须兼顾 Stage‑B 的 rollout 推理质量。
- Reflection runs only on explainable mismatches (GT vs model) and proposes ≤3 micro-guidance edits; malformed or non-explainable samples are logged and skipped.
- Reflection prompt ops: `add|update|delete|merge|none`. Use `merge` when two guidance lines are semantically redundant—LLM picks a canonical `key` and lists `merged_from`; the engine folds sources/hit-miss stats into the keeper, removes merged keys, then exact-compacts.
- `stage_a_paths` must point to Stage-A JSONL files containing `mission`, `group_id`, `label` (`pass|fail`), and `per_image`; keys are normalized to `image_{n}`.

##### Config Breakdown (`src/stage_b/config.py`)

- `model`: HF checkpoint path, dtype (`bfloat16` recommended), and device map.
- `sampler`: decode grid (temperature/top_p/max_new_tokens/stop) 与 `samples_per_decode`。提示现要求“两行输出”：`Verdict: 通过|不通过` + `Reason: <单行理由>`，不再包含 `Evidence_Positive/Negative`。
- `selection`: majority vote + temperature tie-break; uses vote strength, no confidence/self-consistency.
- `manual_review`: 低一致性阈值（`min_verdict_agreement`）用于记录警示；大多数复核仍通过 `manual_review_queue.jsonl`。
- `reflection`: prompt-only JSON ops; batch size configurable (debug default 4), `max_operations` (<=3 recommended).
- `runner`: epochs and `rollout_batch_size` (larger batches speed up sampling).
- `output`: Root/run_name plus mission subdirs for artifacts.
- `guidance`: Global seed file and snapshot retention count.

##### Outputs & Directory Layout

Each mission writes to `{output.root}/{output.run_name}/{mission}/`:
- `guidance.json` + `snapshots/`
- `trajectories.jsonl` — One entry per candidate with decode params, parsed verdict/reason, evidence arrays, format flag.
- `selections.jsonl` — Final verdict per ticket with `vote_strength`, guidance/reflection metadata, warnings.
- `manual_review_queue.jsonl` — Tickets needing human review (`format_error` or `no_explainable_evidence`).
- `failure_malformed.jsonl` — Parser/format failures for prompt debugging.
- `reflection.jsonl` — Applied/attempted guidance updates with evidence group ids.

> Tip: raise `sampler.samples_per_decode` when you need multiple attempts per group under the same decode hyperparameters; drop it to `1` for deterministic sweeps.
> Configure multi-epoch sweeps via `runner.epochs`. The runner shuffles tickets using `(seed + epoch)` so repeated runs with the same seed stay reproducible.

##### Offline Audit
- `scripts/stage_b_conflict_audit.py --run-dir <output/run_name> [--mission <name>]`：列出重复 `conflict_flag=true` 但从未被反思操作覆盖的样本（常见于标签/摘要噪声或指导缺口），便于人工复核或回溯 Stage-A。

##### Manual Review & Failures

- 没有 CriticEngine；只有主模型生成四行输出（含 JSON evidence）。
- 解析失败或缺失 evidence 的候选会写入 `failure_malformed.jsonl`，对应 group 进入 `manual_review_queue.jsonl`（reason `format_error`).
- GT 与模型矛盾且缺少相应 evidence 时，group 进入 `manual_review_queue.jsonl`（reason `no_explainable_evidence`），不参与反思。

##### Guardrails & Label Alignment

- 不再有硬编码 `label=fail` 覆盖；未解释的 GT/模型矛盾进入 `manual_review_queue.jsonl`。
- 风险控制依赖 prompt 引导 + evidence 质量；无规则黑名单。
