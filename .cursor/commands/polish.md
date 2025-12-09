---
name: /polish-prompt
id: polish-prompt
category: Prompting
description: Rewrite a raw user request into a clear, complete, context-aware prompt for this repo, without limiting task types.
---
$ARGUMENTS

**Purpose**  
Turn vague or partial input into a crisp prompt that another agent can execute safely. Stay general: support any task (code, configs, docs, data, Stage-A/B, training, ops).

**Mindset**
- Clarity first; output is a polished prompt, not the solution.
- Use real repo context when it materially disambiguates; don’t invent details.
- Don’t constrain scope unless the user already set it.

**When to run**
- The request is underspecified, ambiguous, or context-dependent.
- Multiple parts of the system could be involved (docs, configs, pipelines, models, agents).
- Before drafting plans/edits/audits to ensure shared understanding.

**Quickstart (5 steps)**
1) Extract intent: restate the core ask; if stage/config/artifact paths are unclear, ask one precise question only if truly blocking.
2) Context discovery: skim relevant repo anchors when helpful: `docs/README.md` (index), `docs/training/REFERENCE.md` (arch map), `scripts/README.md`, `configs/`, `openspec/AGENTS.md` for governance, `src/stage_a/` & `scripts/stage_a_infer.sh`, `src/stage_b/` & `scripts/stage_b_run.sh`, `src/config/schema.py` for validation, `data_conversion/`, `docs/AUGMENTATION.md`.
3) Gap filling: add only essentials the user omitted (paths, configs, expected artifacts, validation/seed/logging norms); note pass/fail normalization and geometry/taxonomy rules if relevant.
4) Prompt construction: write a self-contained instruction with goal, scope, inputs/outputs, required files/configs, expected validation/tests, and any doc/update expectations.
5) Safety check: ensure no fictional files/APIs; keep dependencies justified; avoid broad try/except; fail fast with remediation hints.

**Output**  
Return a single polished prompt in plain language that includes:
- Clear goal and scope (don’t force a stage; use user’s intent).
- Minimal but essential repo context you found relevant.
- Explicit inputs/outputs (paths/configs/artifacts), validation or test steps if applicable.
- Constraints: config-first, deterministic seeds, use `src/utils/logger.get_logger`, no silent defaults.
If info is missing and blocking, ask one concise clarifying question; otherwise proceed.

**Rules**
- Do not modify files.
- Stay general-purpose; do not narrow task categories.
- Cite real paths/configs when referenced; no inventions.