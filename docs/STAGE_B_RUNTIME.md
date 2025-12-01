# Stage-B Runtime Guide

#### Stage-B Runner (`src/stage_b/runner.py`, `scripts/stage_b_run.sh`)

Purpose: Training-free verdict loop that performs ingest → rollout → selection → reflection with mission-specific guidance updates, returning canonical `pass|fail` verdicts per ticket.

```bash
# Default debug config bundled with the repo
gpus=0 bash scripts/stage_b_run.sh

# Use the production run config
config=configs/stage_b/run.yaml gpus=0 log_level=logging \
  bash scripts/stage_b_run.sh
```

- `GuidanceRepository` copies the global guidance file into `{output.root}/{output.run_name}/{mission}/guidance.json` so edits stay isolated until you manually promote them back.
- The shared Qwen3-VL model (defined under `model.*`) is reused by the sampler, critic, and reflection engine to minimize VRAM footprint.
- `stage_a_paths` must point to Stage-A JSONL files that contain `mission`, `group_id`, `label` (`pass|fail`), and a `per_image` map; the runner normalizes `image_{n}` keys automatically. Ensure upstream Stage-A outputs adhere to `docs/DATA_JSONL_CONTRACT.md` for geometry/object structure.

##### Config Breakdown (`src/stage_b/config.py`)

- `model`: HF checkpoint path, dtype (`bfloat16` recommended), and device map (`auto` for single GPU, or explicit slices).
- `sampler`: Array of decode configs (temperature/top_p/max tokens/stop tokens) plus `samples_per_decode` and optional format filter. `RolloutSampler` iterates this grid deterministically per batch.
- `signals`: Enables deterministic metrics — `store_confidence` propagates candidate confidence; `enable_consistency` computes consensus ratios for manual review gating。附带轻量“不确定词面”解析（按 mission 关键词过滤），将模糊/无法判断等表述标记为 `needs_manual_review` 与 `uncertainty_notes`，并降置信但不直接判 fail。
- `critic`: Controls `CriticEngine` (prompt template, decode knobs, per-candidate caps, char budgets). Disable this block to skip LLM critiques.
- `selection`: Policy + tie-breaker for `select_for_group` (e.g., `label_match_then_confidence`).
- `manual_review`: Thresholds for deferring to humans when the majority verdict is low-confidence despite agreement or when label match flips happen.
- `reflection`: Prompt path, batch size, eligibility policy, and change budgets. `rapid_mode` removes guardrails for quick dry runs.
- `runner`: Epoch count (per mission). Combined with `seed` and `_shuffle_indices`, this determines ticket order per epoch.
- `output`: Root/run_name plus legacy paths for consumers expecting `trajectories.jsonl` / `selections.jsonl`.
- `guidance`: Global guidance file and snapshot retention count.

##### Outputs & Directory Layout

Each mission writes to `{output.root}/{output.run_name}/{mission}/`:
- `guidance.json` + `snapshots/` — Mission-specific guidance plus retained history.
- `trajectories.jsonl` — One entry per candidate with signals, critic output, reflection metadata, epoch number.
- `selections.jsonl` — Final verdict per ticket，含 `conflict_flag`、`needs_manual_review`，以及选中的候选索引、反思轮次、manual-review 标记和 eligibility 备注。
- `reflection.jsonl` — Chronological log of reflection bundles, proposals, applied status, and the evidence groups they cite.

> Tip: raise `sampler.samples_per_decode` when you need multiple attempts per group under the same decode hyperparameters; drop it to `1` for deterministic sweeps.
> Configure multi-epoch sweeps via `runner.epochs`. The runner shuffles tickets using `(seed + epoch)` so repeated runs with the same seed stay reproducible.

##### Offline Audit
- `scripts/stage_b_conflict_audit.py --run-dir <output/run_name> [--mission <name>]`：列出重复 `conflict_flag=true` 但从未被反思操作覆盖的样本（常见于标签/摘要噪声或指导缺口），便于人工复核或回溯 Stage-A。

##### Critic & Manual Review

- `CriticEngine` consumes parsed candidates + deterministic signals and emits JSON critiques with `summary`, `critique`, `issues`, and optional `uncertainty_note`. Configure prompts under `configs/stage_b/prompts/critic*.md`.
- Manual review signals分两层：  
  - `manual_review_recommended=true` 来自高一致性但与标签矛盾的池子（原有阈值逻辑），这类样本会跳过反思直接人工复核。  
  - `needs_manual_review`/`uncertainty_notes` 记录“无法判断/模糊”等表述或 critic 给出的 `needs_recheck`/`evidence_sufficiency=false`/`recommended_action=人工复核`，供审计和反思触发，不直接做硬规则 fail。
- Deterministic signals now carry `conflict_flag` (label_mismatch 或守护逻辑触发)；反思 eligibility 会优先吸纳 conflict/uncertainty 样本，同时保持 change_cap/batch_size 守护。

##### Guardrails & Label Alignment

- 历史标签为 `fail` 时，最终 verdict 仍固定为 `fail`（可在 reason 中标注“怀疑噪声”），避免误放行。
- 没有硬编码 rule_fail 列表；风险控制依赖 prompt 引导 + 标签兜底 + 不确定/冲突信号。
