# Stage-A/Stage-B Runtime Guide

Combined technical runbook for running Stage-A summarization and the Stage-B prompt-only verdict loop. Keep this file as the single engineer-facing reference for runtime commands and outputs.

---

## Stage-A Runtime (single-image summaries)

- Purpose: Generate per-image summaries (rare/long-tail coverage) that Stage-B ingests for group-level verdicts.
- Entrypoint: `scripts/stage_a_infer.sh` wraps `python -m src.stage_a.cli` with mission defaults and device selection.

```bash
mission=挡风板安装检查 gpu=0 verify_inputs=true \
  bash scripts/stage_a_infer.sh
```

**Inputs**
- Layout: `<root>/<mission>/{审核通过|审核不通过}/<group_id>/*.{jpg,jpeg,png}`; labels inferred from parent dir (`审核通过`→`pass`, `审核不通过`→`fail`).

**Outputs**
- JSONL at `<output_dir>/<mission>_stage_a.jsonl` with `group_id`, `mission`, `label`, `images`, and normalized `per_image` keys (`image_1`, `image_2`, ...). Format aligns with `docs/DATA_JSONL_CONTRACT.md`.

**Key flags / env vars**
- `mission` (required) — must match `SUPPORTED_MISSIONS`.
- `verify_inputs` — logs first-chunk hashes and grid/token counts; optional.
- `no_mission` — skip mission focus text for smoke tests.
- `gpu` / `device` — device selection (`cuda:N` or `cpu`).

**Checkpoint choices**
- Adapter-based for iteration; merged checkpoints for production (see `scripts/train.sh` exports or `swift export --merge_lora true`).

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
- Shared Qwen3-VL model is reused by sampler and reflection; **no CriticEngine**.
- Reflection runs only on explainable mismatches (GT vs model) and proposes ≤3 micro-guidance edits; malformed or non-explainable samples are logged and skipped.
- `stage_a_paths` must point to Stage-A JSONL files containing `mission`, `group_id`, `label` (`pass|fail`), and `per_image`; keys are normalized to `image_{n}`.

##### Config Breakdown (`src/stage_b/config.py`)

- `model`: HF checkpoint path, dtype (`bfloat16` recommended), and device map.
- `sampler`: decode grid (temperature/top_p/max_new_tokens/stop), `samples_per_decode`, optional format filter. Prompt现要求“两行输出”：`Verdict: 通过|不通过` + `Reason: <单行理由>`，不再包含 `Evidence_Positive/Negative`。
- `selection`: majority vote + temperature tie-break; uses vote strength, no confidence/self-consistency.
- `manual_review`: basic gating still available but most review routed via explicit queues.
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
