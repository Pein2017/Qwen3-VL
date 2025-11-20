# Stage-A & Stage-B: Group Quality Control Pipeline

Business briefing on the two-stage quality control pipeline that underpins production-grade group verdicts.

**Audience**: Quality operations leads, production planners, mission owners, compliance partners.

**Further technical reading**: `docs/STAGE_A_TECH_GUIDE.md`, `docs/STAGE_B_RUNTIME.md`, `openspec/changes/2025-11-03-adopt-training-free-stage-b/`.

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
| Process | Frozen vision-language model drafts neutral, per-image observations. |
| Output | JSONL feed of observation packets—each group has structured descriptions, label provenance, timestamp, and completion flag. |

**Why it matters**
- Creates an explainable narrative for each image, so Stage-B and human auditors can reason about defects.
- Normalizes vocabulary and enforces required metadata (mission, group id, label source).
- Acts as the single source of truth for Stage-B and any downstream analytics (e.g., defect heatmaps).

**Operational guardrails**
- Image catalog must stay synchronized with mission focus; onboarding checklists ensure a representative sample per verdict.
- Data Ops verifies `stage_a_complete=true` before releasing batches to Stage-B.

---

## 4. Stage-B in Business Terms

| What | Business Interpretation |
| ---- | ---------------------- |
| Input | Stage-A summaries + ground-truth labels + current mission guidance. |
| Process | Samples multiple verdict drafts, scores them with deterministic signals, convenes an LLM “reflection” to propose guidance edits, and applies approved changes. |
| Output | Final verdicts (JSONL), trajectories for audit, reflection log summarizing guidance shifts, updated mission-specific guidance repository. |

**Experiences = living policy**
- Guidance entries (`[G0]`, `[G1]`, …) are policy snippets the model reads before making a decision.
- Reflection analyzes recent wins/losses and requests incremental edits (add, revise, retire rules).
- Each applied change records rationale, evidence group ids, and reflection id for traceability.
- Guidance updates are applied to mission-specific files; promotion to global guidance requires manual review and deployment.

**Business value**
- Enables weekly policy refresh without model retraining.
- Ensures decisions remain aligned with latest defect definitions and customer commitments.
- Produces a clear chain of responsibility: which batch triggered the change, who approved deployment, and the rationale for each guidance update.

---

## 5. End-to-End Operating Model

1. **Data Intake (Field Ops)**
   - Capture mission-tagged photo groups, confirm label quality, drop into intake bucket.
2. **Stage-A Run (ML Ops)**
   - Schedule nightly or on-demand summarization; publish Stage-A JSONL with completion report.
3. **Pre-Flight Checks (Quality PM)**
   - Review Stage-A quality spot-check, confirm guidance file is populated, sign off on batch release.
4. **Stage-B Loop (Reflection Steward)**
   - Start run, monitor per-batch dashboards (label match, semantic consistency).
   - Approve or veto reflection proposals flagged as “uncertain” before they are applied (configurable).
5. **Verdict Handoff (Business Owner)**
   - Consume selections feed; annotate exceptions needing human escalation.
   - Update downstream systems (MES, supplier notifications, billing adjustments).
6. **Guidance Governance (Compliance)**
   - Archive reflection logs, compare against policy baseline, prepare weekly review deck.

**Cadence suggestions**
- High-volume missions: run Stage-A continuously, Stage-B hourly with 32-record reflection batches.
- Low-volume missions: Stage-A weekly, Stage-B on demand with manual approval of guidance edits.

### Runbook: CLI Reference

- **Stage-A summaries** — `scripts/stage_a_infer.sh` wraps `python -m src.stage_a.cli`, enforces the canonical mission focus prompts, and verifies that every intake folder carries `stage_a_complete=true` before emitting summaries into `output_post/stage_a/`. Override mission/device via env vars (`mission=... gpu=0`).
- **Stage-B reflection loop** — `scripts/stage_b_run.sh` launches `python -m src.stage_b.runner` with `configs/stage_b/run.yaml` (or a mission-specific override). It writes `guidance.json`, `trajectories.jsonl`, `selections.jsonl`, and `reflection.jsonl` under `{output.root}/{output.run_name}/{mission}/`. Add `log_level=debug` when ops needs to inspect critic output.
- **Guidance hygiene** — The runner copies the global guidance file into each mission directory. Promote or roll back mission edits by syncing those files back into the shared guidance repo after review.

---

## 6. Decision Metrics & Business KPIs

- **Label Match Rate**: Primary accuracy KPI; tracked overall and by mission, with pre/post reflection uplift.
- **Escalation Rate**: Percentage of groups routed to manual review; target <5% for stable missions.
- **Time to Verdict**: Intake to final verdict SLA (hours). Stage-B automation keeps this <4h on average.
- **Guidance Churn**: Count of applied reflections per week; spikes trigger policy review.
- **Confidence Distribution**: Monitor for drift—sustained low confidence may indicate prompt or data issues.
- **Supplier Impact**: Number of adverse decisions per supplier; feed into commercial scorecards.

Each KPI is reported via `reflection.jsonl` and `selections.jsonl` exports. Analytics teams tie these into Tableau/Looker dashboards.

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

- Technical operations guide: `docs/STAGE_B_RUNTIME.md`
- Prompt templates & schema: `configs/prompts/`, `src/stage_b/prompts.py`
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
