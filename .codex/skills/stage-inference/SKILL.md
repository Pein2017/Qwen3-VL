---
name: stage_inference
description: "Two-stage QC inference (Stage-A summarization + Stage-B group verdicts) for the Qwen3-VL project. Use for fast operator runbooks: how to run Stage-A/Stage-B, where artifacts live, and a debug checklist. Prefer docs/ as the source of truth; this skill should point to docs and scripts instead of duplicating long knowledge."
---

# Stage-A/Stage-B QC Inference (Operator Runbook)

Keep `docs/` as the source of truth. Use this skill as a fast “router”:
- Run Stage-A/Stage-B quickly via `scripts/`.
- Jump to the right docs/code paths.
- Follow a short debug checklist without duplicating long mission knowledge.

## 0) Quick commands

Run Stage-A (per-image summaries):
```bash
mission=挡风板安装检查 gpus=0 bash scripts/stage_a.sh
```
Override defaults when needed:
```bash
checkpoint=/path/to/ckpt input_dir=/path/to/groups output_dir=/path/to/out mission=挡风板安装检查 gpus=0 bash scripts/stage_a.sh
```

Run Stage-B (per-group verdicts, rule_search only):
```bash
config=bbu_line gpus=0 bash scripts/stage_b.sh
```

Baseline-only Stage-B audit (skip proposer/reflection):
```bash
jump_reflection=true config=bbu_line gpus=0 bash scripts/stage_b.sh
```

Smoke/audit Stage-B implementation (no model required):
```bash
bash scripts/stage_b.sh smoke
```

## 1) Decide what is broken (Stage-A vs Stage-B)

1) If the issue is “summaries are wrong / format is wrong / per-image text missing” → treat as **Stage-A**.
2) If summaries look fine but verdict is surprising / artifacts look inconsistent / guidance changes are odd → treat as **Stage-B**.

## 2) Where to look (docs are canonical)

Stage-A runtime + contract:
- `docs/runtime/STAGE_A_RUNTIME.md`
- `docs/runtime/STAGE_A_STAGE_B.md`

Stage-B runtime + artifacts:
- `docs/runtime/STAGE_B_RUNTIME.md`
- `docs/runtime/STAGE_A_STAGE_B.md`

Mission knowledge / rules (do NOT duplicate here):
- `docs/stage-B-knowledge-Chinese.md`

## 3) Artifact map (Stage-B)

Under `{output.root}/{mission}/{run_name}/` (see `configs/stage_b/*.yaml`):
- `rule_candidates.jsonl`: proposed candidate rules + metrics + gate decisions.
- `benchmarks.jsonl`: accepted-rule history (baseline/after metrics).
- `rule_search_hard_cases.jsonl`: hardest mismatches per iteration (manual tracing).
- `rule_search_candidate_regressions.jsonl`: base-correct → candidate-wrong audits.
- `guidance.json` + `snapshots/`: mission-specific guidance and snapshots.
- `distill_chatml.jsonl`: optional export when `stage_b_distillation` is enabled.

Baseline-only audit (`jump_reflection=true`):
- `baseline_metrics.json`, `baseline_ticket_stats.jsonl`
- `baseline_wrong_cases.jsonl`, `baseline_np_cases.jsonl`, `baseline_ng_cases.jsonl`
- `baseline_metrics_steps.jsonl`

## 4) Code map (what to edit)

Stage-A:
- Entrypoint: `src/stage_a/`
- Prompts: `src/stage_a/prompts.py`, `src/prompts/summary_profiles.py`, `src/prompts/domain_packs.py`
- Summary builder / data contracts: `data_conversion/pipeline/summary_builder.py`, `docs/data/`

Stage-B:
- Entrypoint: `src/stage_b/runner.py` + `scripts/stage_b.sh`
- Ingest Stage-A JSONL: `src/stage_b/ingest/stage_a.py`
- Prompt construction: `src/stage_b/sampling/prompts.py`, `src/prompts/domain_packs.py`
- Output parsing: `src/stage_b/rollout.py` (`_parse_two_line_response`)
- Guidance persistence: `src/stage_b/io/guidance.py`
- Export format: `src/stage_b/io/export.py`

## 5) Debug checklist (fast + deterministic)

If Stage-B output is “weird”:
1) Verify inputs parse: Stage-A JSONL has `mission/group_id/label/per_image` and `per_image` is non-empty.
2) Inspect `rule_candidates.jsonl` + `benchmarks.jsonl`: are candidates passing gate but hurting metrics?
3) Check `guidance.json` step drift: compare with `snapshots/` to see what changed.
4) If baseline-only, inspect `baseline_wrong_cases.jsonl` first.
5) If this is “mission-rule” disagreement, read `docs/stage-B-knowledge-Chinese.md` and decide whether rules or prompts need adjustment.

Search tips:
- `rg -n \"rule_candidates|benchmarks|hard_cases|regressions\" output_post/stage_b -S`
- `rg -n \"_parse_two_line_response|Verdict:\" src/stage_b -S`
