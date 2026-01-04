# Design — Guidance Scaffold (S*) + Hypothesis Gating

## Overview

目标：让 Stage‑B 在不牺牲可扩展性的前提下，最大化 “模型挖掘潜在规律” 的能力，同时避免把结构性脚手架与噪声规则直接写进 `guidance`。

核心思想：
- **结构不变量（scaffold）**：mission‑wise、只读、始终进入 rollout prompt，确保推理框架稳定；
- **候选假设（hypothesis）**：reflection 输出“可证伪、可泛化”的候选规则，先入池，跨批次积累支持证据，再晋升为 `G*`。

## Layered Guidance Model

### Key namespaces

- `G0`：mission checklist（可被 reflection 更新，但不可被删除）
- `S1..Sn`：mission‑wise scaffold（只读；结构不变量）
- `G1+`：mutable guidance（可增删改合并；由 reflection 学习/清理）

### Prompt rendering (conceptual)

Rollout prompt SHOULD render two clearly separated blocks:
1) `S*` as **“结构不变量（必须遵守）”**
2) `G0+` as **“可学习规则（可能被更新）”**

Rationale: 提高模型对 scaffold 的服从度，降低把 scaffold 当作“可改写规则”的概率。

## Hypothesis Workflow

### High-level flow

1) **Rollout**: produce multiple candidates (3 temps) under strict binary 2-line protocol.
2) **Selection**: majority vote → final verdict; compute signals (label_match, contradiction, vote_strength, low_agreement).
3) **Decision pass (stop-gradient)**: given `gt_label`, decide `no_evidence_group_ids` (ticket_key-level stop-gradient).
4) **Ops pass (learnable-only)**:
   - propose hypotheses for learnable groups (NOT direct guidance edits by default).
   - each hypothesis MUST be falsifiable and evidence-backed (ticket_key list).
5) **HypothesisGate (deterministic)**:
   - accumulate support evidence across reflection cycles/batches.
   - promote only when thresholds are met.
6) **Promotion**:
   - convert promoted hypothesis into `G*` experience (upsert key=None).
   - obey per-epoch cap and lifecycle cleanup.

### Hypothesis definition

A hypothesis is a candidate rule with three properties:
- **Generalizable**: “条件→结论（通过/不通过）”，不包含样本细节；
- **Testable**: 能在后续 batch 上被支持/反驳（至少能产生 hit/miss）；
- **Falsifiable**: 明确写出一个“反例条件”使其可被淘汰。

### Minimal gating (simplicity-first)

不引入额外的“对候选规则投票”的 LLM pass；门控完全 deterministic：

- Normalize each hypothesis into a stable signature:
  - simplify Chinese, normalize spaces, trim punctuation, stable template.
- For each signature, maintain:
  - `support_cycles`: number of distinct reflection cycles where it was proposed
  - `support_ticket_keys`: union of evidence ticket_keys
  - `first_seen`, `last_seen`
  - `status`: `candidate|promoted|rejected`

Promotion rule (default):
- Promote if:
  - `support_cycles >= 2`
  - `len(unique(support_ticket_keys)) >= 6`
  - AND within the current epoch promotion cap.
  - Thresholds are config-overridable; no special-case dimensions.

Falsification:
- Primary: after promotion, existing hit/miss + confidence lifecycle provides automatic淘汰（miss 累积）。
- Secondary (optional future): include “counterexample” reporting and reject pre-promotion.

### Closure with hypotheses

We keep learnability closure but extend “coverage source”:
- Let `L` be learnable ticket_keys in ops pass input.
- Let `H` be union of validated `hypotheses[*].evidence` (ticket_keys).
- Let `E` be union of validated `operations[*].evidence` (ticket_keys).

The system MUST enforce closure `L == (H ∪ E)` via bounded retries.  
This ensures every learnable case is either:
- stop-gradient (decision pass), OR
- contributes to a hypothesis/op evidence set (gradient).

## Prompt Changes (requirements-level intent)

Decision prompt:
- strictly outputs only `no_evidence_group_ids` + `decision_analysis`.
- explicitly instructs that third-state logic → stop-gradient, not rules.

Ops prompt:
- outputs hypotheses (and optionally immediate operations).
- enforces binary rule text; forbids “复核/不应直接/佐证/证据不足/待定”等第三态词面及常见变体。
- provides a stable phrasing template to reduce de-dup friction in signature normalization.

## Observability / Artifacts

New mission-scoped artifacts under `{output.root}/{mission}/{run_name}/`:
- `hypotheses.json` (or JSONL): aggregated pool (candidate/promoted/rejected + counters)
- `hypothesis_events.jsonl`: append-only events (proposed/promoted/rejected) with cycle metadata and evidence ticket_keys

Notes:
- Stop-gradient queue artifacts are consolidated into a single queue file; no manual-review queue artifacts.
- `dimension` is optional and must not be used for brand-specific gating.

These artifacts are purely for optimization traceability and do not affect rollout protocol.
