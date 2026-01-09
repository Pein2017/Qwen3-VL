---
name: qwen3vl-stageb-training-free
description: "Stage-B training-free optimization (rule_search tree-growth only). Use for config iteration, prompt changes, regression checks, Stage-A ingest compatibility, and no-model smoke audits for Stage-B verdicts."
---

# Stage-B Training-Free Optimization

Scope: prompt-only verdict loop driven by Stage-A summaries, **rule_search only**:
- rollout-heavy tree-growth with proposer + metric gate
- optional baseline-only audit via `jump_reflection=true`

## 0) Authoritative references

- OpenSpec: `openspec/AGENTS.md`, `openspec/project.md`, `openspec/specs/stage-b-training-free/spec.md`
- Runtime docs: `docs/runtime/STAGE_B_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md`
- Training/inference overview: `docs/training/REFERENCE.md`
- Mission knowledge: `docs/stage-B-knowledge-Chinese.md`

When spec/doc conflicts with current code, pause and route changes through OpenSpec before implementation.

## 1) Entry points

No-model audit (fast):
```bash
bash scripts/stage_b.sh smoke
```

Rule-search (tree-growth) run:
```bash
config=bbu_line gpus=0 bash scripts/stage_b.sh
```

Baseline-only audit (skip proposer/reflection):
```bash
jump_reflection=true config=bbu_line gpus=0 bash scripts/stage_b.sh
```

Use `conda run -n ms ...` if the environment is not activated.

## 2) Output layout (source of truth = code)

Run directory: `{output.root}/{mission_name}/{run_name}/`
Key artifacts (rule_search):
- `rule_candidates.jsonl`, `benchmarks.jsonl`
- `rule_search_hard_cases.jsonl`, `rule_search_candidate_regressions.jsonl`
- `distill_chatml.jsonl` (only when `stage_b_distillation` is enabled)
- `guidance.json` + `snapshots/`

Baseline-only audit (when `jump_reflection=true`):
- `baseline_metrics.json`, `baseline_ticket_stats.jsonl`
- `baseline_wrong_cases.jsonl`, `baseline_np_cases.jsonl`, `baseline_ng_cases.jsonl`
- `baseline_metrics_steps.jsonl`

## 3) Code map (post-refactor)

- Config & schema: `src/stage_b/config.py`
- Main runner: `src/stage_b/runner.py`
- Stage-A ingest: `src/stage_b/ingest/stage_a.py`
- Prompt assembly: `src/stage_b/sampling/prompts.py`
- Rollout + parsing: `src/stage_b/rollout.py`
- Scoring & metrics: `src/stage_b/scoring/`
- Guidance IO: `src/stage_b/io/guidance.py`
- Exports + reports: `src/stage_b/io/export.py`, `src/stage_b/io/group_report.py`
- Rule search: `src/stage_b/rule_search.py`

## 4) Behavioral invariants (verify after changes)

- Two-line protocol: `Verdict: 通过|不通过` + `Reason: <single line>`
- Rollout prompt does not expose GT labels
- Rule-search proposer uses reflection prompts, but decision/ops passes are not executed
- Multi-GPU rollout is ticket-parallel; rank 0 writes artifacts
- Stage-B expects **exactly one mission per run_name**

## 5) Debug checklist (quick isolation)

- Parse/format errors -> `src/stage_b/rollout.py`
- Metric drift -> `src/stage_b/scoring/` + `src/stage_b/signals.py`
- Guidance snapshots -> `{run_dir}/guidance.json` + `snapshots/`
- Rule-search gating -> `src/stage_b/rule_search.py` + `src/stage_b/config.py`
- Stage-A ingest schema mismatch -> `src/stage_b/ingest/stage_a.py` + `docs/runtime/STAGE_A_RUNTIME.md`

## 6) Minimal validations

- No-model smoke: `bash scripts/stage_b.sh smoke`
- Small real run: use a tiny Stage-A JSONL with a single mission
- If prompts changed: inspect `rule_candidates.jsonl` + `benchmarks.jsonl` first
