# Stage-A & Stage-B: Group Quality Control Pipeline

Business briefing on the two-stage quality control pipeline that underpins production-grade group verdicts.

**Stage‑1 (Stage‑A)** focuses on **single-image basic object recognition** (including rare/long-tail items) and produces structured evidence per image.

**Stage‑2 (Stage‑B)** consumes that evidence plus labels to **verify installation criteria at the group/ticket level** and issues a binary `Pass`/`Fail` verdict with rationale.

**Audience**: Quality operations leads, production planners, mission owners, compliance partners.

**Further technical reading**: `./STAGE_A_RUNTIME.md`, `./STAGE_B_RUNTIME.md`, and `openspec/changes/2025-11-03-adopt-training-free-stage-b/`.

---

## 1. Why This Pipeline Matters

- Maintains consistent pass/fail decisions for mission-based image inspections without training new models.
- Reduces manual review backlog by converting raw imagery into structured evidence and auto-generated verdicts.
- Creates an auditable trail of how guidance evolves, enabling governance teams to approve or roll back policy shifts.
- Provides a shared decision ledger for downstream systems (MES, supplier scorecards, compliance dashboards).

The pipeline is intentionally **training-free**: instead of fine-tuning a checkpoint for each mission, Stage-B refines written guidance that the model reads before it answers. This keeps iteration fast, approvals lightweight, and costs predictable.

---

## 2. Business Backdrop

### Mission Workflow
- Each mission corresponds to a customer-facing inspection program (e.g., 挡风板 quality, enclosure sealing, electrical harness labeling).
- Field teams supply labeled photo groups drawn from factory or partner submissions.
- Operations must deliver verdicts within SLA while preserving justification texts for later audits.

### Pain Points Addressed
- **Volume**: Daily submissions per mission can exceed what manual reviewers can process.
- **Variance**: Verdict language and evidence historically differed between reviewers, complicating traceability.
- **Playback**: Leadership needs to know *why* decisions changed, not just the outcome.

The Stage-A/Stage-B stack addresses these by standardizing summaries, orchestrating verdict exploration, and codifying learnings into portable guidance.

---

## 3. Stage-A in Business Terms

| What | Business Interpretation |
| ---- | ---------------------- |
| Input | Mission folders with “审核通过/审核不通过” splits and photo groups (`QC-{mission}-{date}-{id}`) curated by labeling ops. |
| Process | Frozen vision-language model drafts neutral, per-image observations with explicit coverage for rare/long-tail objects. |
| Output | JSONL feed of observation packets—each group has structured per-image descriptions, label provenance, timestamp, and completion flag. |

Note: The same `group_id` may appear in both label folders for resubmitted batches. Stage-A emits one record per occurrence, preserving the provided label; Stage-B ingests them as distinct tickets keyed by `(group_id, label)`.

**Why it matters**
- Creates an explainable narrative for each image, so Stage-B and human auditors can reason about defects.
- Normalizes vocabulary and enforces required metadata (mission, group id, label source).
- Acts as the single source of truth for Stage-B and any downstream analytics (e.g., defect heatmaps).

**Operational guardrails**
- Image catalog must stay synchronized with mission focus; onboarding checklists ensure a representative sample per verdict.
- Data Ops spot-checks Stage-A JSONL outputs before releasing batches to Stage-B.
  - Throughput note: Stage-A supports a `sharding_mode=per_image` runtime mode to improve load balancing and batch utilization across GPUs; see `./STAGE_A_RUNTIME.md` for details and determinism notes.
- Prompt profile note: Stage‑A runtime composes the **summary_runtime** profile (base summary prompt + concise domain glossary for `bbu|rru`). Summary SFT training stays on the **summary_train_min** profile (format + task criterion only).

---

## 4. Stage-B in Business Terms

| What | Business Interpretation |
| ---- | ---------------------- |
| Input | Stage-A summaries + ground-truth labels + current mission guidance. |
| Process | Prompt-only rollouts（提示=guidance+Stage-A 摘要，不含 GT；**领域提示以只读块追加在 system prompt**，由 Stage‑B config 的 `domain_map/default_domain` 决定；S* 为只读结构不变量，G0+ 为可学习规则）；推理输出必须严格两行二分类：`Verdict: 通过|不通过` + `Reason: ...`，且最终输出禁止任何第三状态词面。多数表决 selection + **mission-scoped fail-first** 确定性护栏：仅当负项与当前 mission 的 `G0` 相关时才触发整组不通过（含 pattern-first `不符合要求/<issue>`）；若护栏覆盖采样 verdict，则必须重写 `Reason` 以与最终 `Verdict` 一致。Stage‑B 仅对**梯度候选**触发反思（错例、rollout 矛盾/低一致性、冲突/需复核信号）；反思为 two-pass：decision pass 在看到 `gt_label` 后判定 stop-gradient（`no_evidence_group_ids`），ops pass 仅基于 learnable groups 产出严格 JSON ops + hypotheses（含严格 evidence）。系统对未覆盖 learnable groups 做 bounded retry（默认 2 次）并设置成本上界；耗尽预算者进入 `need_review_queue.jsonl`（stop-gradient, `reason_code=budget_exhausted`）。两行协议解析失败/无可用候选/selection 报错等硬故障仅写入 `failure_malformed.jsonl`。重跑同一 run_name 时重建 per-run artifacts 与 reflection_cache，指导沿用上次快照（除非显式 reset）。 |
| Output | Final binary verdicts (`pass` / `fail`) in JSONL, trajectories for audit, step-wise + epoch summary metrics (`metrics.jsonl`), optional step-wise group snapshot deltas (`group_report_delta.jsonl`), reflection log, hypothesis pool (`hypotheses.json` + `hypothesis_events.jsonl`), `need_review_queue.jsonl` + `need_review.json`（人工复核）, `failure_malformed.jsonl`（硬故障调试）, and updated mission-specific guidance repository (full `group_report.jsonl` generated at run end). |

**Experiences = living policy**
- Guidance entries (`[G0]`, `[G1]`, …) are policy snippets the model reads before making a decision.
- Reflection analyzes recent wins/losses and requests incremental edits (add, revise, retire rules).
- Each applied change records rationale, evidence ticket_keys, and reflection id for traceability; `need_review_queue.jsonl` is reserved for **stop-gradient** tickets: after seeing `gt_label`, decision pass still cannot propose any auditable hypothesis for that group (`no_evidence_group_ids`), so the sample is excluded from learning and queued for human investigation. Hard failures stay in `failure_malformed.jsonl`.
- Guidance updates are applied to mission-specific files; promotion to global guidance requires manual review and deployment.

**Business value**
- Enables weekly policy refresh without model retraining.
- Ensures decisions remain aligned with latest defect definitions and customer commitments.
- Produces a clear chain of responsibility: which batch triggered the change, who approved deployment, and the rationale for each guidance update.

---

## 5. End-to-End Operating Model

1. **Data Intake & Preprocessing (Field Ops + Data Ops)**
   - Capture mission-tagged photo groups, confirm label quality, drop into intake bucket.
- Optional: normalize new annotation exports via `data_conversion/convert_dataset.sh` (see `../data/DATA_PREPROCESSING_PIPELINE.md`) before training or refreshing Stage‑A models.
2. **Stage-A Run (ML Ops)**
   - Schedule nightly or on-demand summarization; publish Stage-A JSONL with completion report.
3. **Pre-Flight Checks (Quality PM)**
   - Review Stage-A quality spot-check, confirm guidance file is populated, sign off on batch release.
4. **Stage-B Loop (Reflection Steward)**
   - Start run, monitor per-batch dashboards (label match, semantic consistency).
   - 在 legacy LLM 模式下，可对“uncertain” 反思提案进行人工审批；确定性模式无需审批（自动保守处理）。
5. **Verdict Handoff (Business Owner)**
   - Consume selections feed; annotate exceptions needing human escalation.
   - Update downstream systems (MES, supplier notifications, billing adjustments).
6. **Guidance Governance (Compliance)**
   - Archive reflection logs, compare against policy baseline, prepare weekly review deck.

**Cadence suggestions**
- High-volume missions: run Stage-A continuously, Stage-B hourly with 32-record reflection batches.
- Low-volume missions: Stage-A weekly, Stage-B on demand with manual approval of guidance edits.

> Run commands and config details live in `./STAGE_B_RUNTIME.md` to keep this page focused on business operations.

---

## 6. Decision Metrics & Business KPIs

- **Label Match Rate**: Primary accuracy KPI; tracked overall and by mission, with pre/post reflection uplift.
- **Escalation Rate**: Percentage of groups routed to manual review; target <5% for stable missions.
- **Time to Verdict**: Intake to final verdict SLA (hours). Stage-B automation keeps this <4h on average.
- **Guidance Churn**: Count of applied reflections per week; spikes trigger policy review.
- **Verdict/Reason Drift**: Monitor verdict分布与 Reason 关键要素是否异常；持续偏移时检查 prompt 和数据。
- **Supplier Impact**: Number of adverse decisions per supplier; feed into commercial scorecards.

Each KPI is reported via `reflection.jsonl`, `selections.jsonl`, and step-wise telemetry in `metrics.jsonl`（包含/排除人工复核两套 acc/fn/fp/n，同时包含 `logging_steps` 窗口与 epoch 汇总）. For “why did verdict/reason change”, use `group_report_delta.jsonl` (windowed deltas) + the end-of-run `group_report.jsonl`. Analytics teams tie these into Tableau/Looker dashboards.

---

## 7. Governance & Control Points

### Mission Onboarding Checklist
- Business owner defines mission scope, success metrics, and escalation contacts.
- Compliance verifies any personally identifiable information (PII) handling.
- Guidance author drafts initial experiences (`G0`–`G2`) and documents rationale.
- Dry run on historical data; review 50-sample audit before go-live.

### Change Management
- Reflection proposals tagged with `uncertainty_note` require human approval before merging.
- Guidance repository snapshots retained per retention policy (default 10) for rollback audits.
- Weekly governance meeting reviews: (1) applied operations, (2) KPI deltas, (3) outstanding risks.

### Risk & Mitigation
- **Label Noise**: Address via trust signals and supplier feedback loops.
- **Model Drift**: Monitor semantic advantage and confidence; escalate to ML team if sustained decline.
- **Policy Misalignment**: Maintain crosswalk between experiences and official SOP paragraphs.

---

## 8. Stakeholders & Responsibilities

- **Quality Operations Lead**: Owns pipeline schedule, ensures mission coverage, signs off on verdict release.
- **Reflection Steward**: Monitors Stage-B runbooks, approves/refuses uncertain reflections, documents rationale.
- **Mission Owner**: Defines success metrics, receives weekly KPI bundle, requests guidance tweaks.
- **Compliance Partner**: Reviews reflection logs, checks experience wording for regulatory alignment.
- **Data Engineering**: Maintains intake directories, storage quotas, and access controls.

**RACI Snapshot**
- *Run Stage-A/Stage-B*: R = ML Ops, A = Quality Ops Lead.
- *Approve Guidance Changes*: R = Reflection Steward, A = Mission Owner, C = Compliance.
- *Metric Reporting*: R = Analytics, A = Mission Owner, I = Leadership.

---

## 9. Integration Touchpoints

- **MES / Production Systems**: Consume `selections.jsonl` for automated stop/go decisions.
- **Supplier Portals**: Render Stage-A summaries and final verdict reasons in dispute workflows.
- **Audit Archive**: Store guidance snapshots + trajectories for quarterly compliance testing.
- **Analytics Warehouse**: Load JSONL outputs into structured tables for KPI dashboards.

Ensure downstream teams understand schema stability commitments; breaking changes require OpenSpec review.

---

## 10. Support & Further Reading

- Technical operations guide: `./STAGE_B_RUNTIME.md`
- Prompt templates & schema: `configs/prompts/`, `src/prompts/summary_profiles.py`, `src/prompts/domain_packs.py`, `src/stage_b/sampling/prompts.py`
- Reflection change log & rationale: `openspec/changes/2025-11-03-adopt-training-free-stage-b/`
- Visualization utilities for QA spot checks: `vis_tools/`

For hands-on runbooks (CLI flags, YAML examples), refer to the technical documents above. This page intentionally focuses on the *why* and *who* behind the pipeline.

---

## 11. Glossary

- **Mission**: A production-quality inspection program with clearly defined pass/fail policy.
- **Experience**: A numbered guidance entry read by the model; represents codified business rules.
- **Reflection**: Automated review that proposes experience edits based on recent performance.
- **Verdict**: Final pass/fail decision plus supporting rationale text.
- **Guidance Repository**: Mission-specific policy file that evolves through reflection cycles; changes are auditable and reversible via snapshots.

---

## 12. Revision Log

- **2025-11-06** — Refocused document on business context, governance, and KPI ownership (Quality Ops request).
- **2025-01-XX** — Initial technical deep-dive (superseded, see technical runbooks for details).
