---
name: code-review
description: "Automated Python code review skill: run ruff (format + lint) and pyright-compatible type checking (pyright or basedpyright), normalize machine-readable findings, audit compliance with docs/reference/SCHEMA_CONSTITUTION.md, and produce a structured report (Summary, Static Analysis, Constitution Violations, Design & Architecture Review, Actionable Fix Plan). Use when asked to review Python code/PRs, lint/type-check, enforce Schema Constitution rules, or assess architecture quality."
---

# Python Code Review (static + schema + architecture)

This skill produces a structured, technically grounded review report for a target Python scope:
- **Static analysis**: `ruff` (format + lint) + `pyright`-compatible type checker (`pyright` if installed, otherwise `basedpyright`).
- **Schema Constitution compliance**: audit against `docs/reference/SCHEMA_CONSTITUTION.md` with clause-mapped findings.
- **Architecture & design**: coupling/cycles/leaky abstractions with concrete evidence.

## 0) Clarify review scope (avoid accidental full-repo scans)

Before running tools, resolve *what is under review*:
- **Explicit paths** (preferred): specific files/dirs (example: `src/stage_b`, `scripts/stage_b.sh` does not apply; this skill targets Python).
- **Git change set**: staged vs unstaged vs a commit range.

If scope is not provided, default to reviewing `src/` only.

## 1) Run the automated collector (machine-readable outputs; auto-fix simple lint)

The helper script prints a **machine-readable JSON report to stdout** (fast introspection).

Defaults are optimized for “auto code reviewer” behavior:
- Apply `ruff` formatting (`ruff format`)
- Apply safe `ruff` auto-fixes (`ruff check --fix`)
- Run schema/architecture heuristics (`--mode full`)

Optionally (when `--write-artifacts` is enabled), it also writes:
- raw tool outputs (`ruff_check.json`, `ruff_format.json`, `pyright.json`)
- unified normalized issues (`issues.json`)
- the JSON report (`summary.json`)

Command (recommended; uses the project’s `ms` conda env):
```bash
conda run -n ms python .codex/skills/code-review/scripts/code_review.py --paths src
```

To review multiple roots:
```bash
conda run -n ms python .codex/skills/code-review/scripts/code_review.py --paths src scripts tests
```

If `pyright` is not installed, the script automatically falls back to `basedpyright` (pyright-compatible engine) when available.

Tool availability notes:
- Prefer running via `conda run -n ms ...` (project default).
- If neither `pyright` nor `basedpyright` is available on `PATH`, install one of them (example: `conda run -n ms pip install basedpyright`).

To keep runs lighter/faster (skip schema/architecture heuristics):
```bash
conda run -n ms python .codex/skills/code-review/scripts/code_review.py --mode fast --paths src
```

To avoid writing code changes (disable auto-fix/format) while still collecting results:
```bash
conda run -n ms python .codex/skills/code-review/scripts/code_review.py --no-fix --no-format --format-check --paths src
```

To write JSON artifacts:
```bash
conda run -n ms python .codex/skills/code-review/scripts/code_review.py --write-artifacts --output-dir tmp/code-review --paths src
```

## 2) Normalize findings into a unified issue format

When `--write-artifacts` is enabled, the script writes `issues.json` with records shaped like:
```json
{
  "tool": "ruff | pyright | basedpyright | constitution | architecture",
  "kind": "lint | format | type | constitution | architecture",
  "severity": "error | warning | info",
  "code": "F401 | reportGeneralTypeIssues | SCHEMA-NONTRIVIAL-MAPPING | IMPORT-CYCLE | null",
  "message": "Human-readable explanation",
  "path": "repo/relative/path.py",
  "line": 123,
  "column": 4,
  "end_line": 123,
  "end_column": 20,
  "category": "Derived grouping label",
  "clause": "For constitution findings: section title reference",
  "remediation": "Minimal-diff fix guidance"
}
```

## 3) Constitution compliance audit (clause-mapped)

Open and apply: `docs/reference/SCHEMA_CONSTITUTION.md`.

Workflow:
1) Treat the script’s `constitution` findings as **suspects**, not final verdicts.
2) For each confirmed violation, include:
   - **Clause reference**: cite the relevant section title (example: “Function signatures and returns”).
   - **What**: the concrete code construct (signature/return/attribute) and where it lives.
   - **Why**: the rule being violated (non-trivial mapping, missing validation at boundary, etc.).
   - **Impact**: how this affects correctness, maintainability, or downstream coupling.
   - **Remediation**: smallest structured-type refactor that satisfies the rule.

## 4) Architecture & design review (evidence-based)

Use the script’s `architecture` findings and extend with manual checks:
- **Tight coupling**: many direct imports / fan-out / cross-layer dependency direction.
- **Leaky abstractions**: public interfaces exposing transport-shaped dicts/lists or internal file layout.
- **Anti-patterns**: global mutable state, hidden I/O, mixed responsibilities, “god” modules.
- **Testability**: ability to construct components without filesystem/network/GPU side effects.

Every finding must be grounded in at least one concrete artifact:
- module/file path(s)
- import-cycle chain(s)
- API signature(s)
- configuration/boundary location(s)

## 5) Emit the structured report (required sections)

Do not expose tool commands, raw logs, or raw JSON in the user-facing response.
Only show results and the minimal next actions.

If `constitution` / `architecture` findings are present, do **not** refactor automatically.
Ask whether to:
- **Skip** (record as known debt), or
- **Refactor/update immediately** (proceed with minimal-diff changes).

User-facing response sections (keep brief; expand only on request):
1) **Summary**: Pass / Conditional Pass / Fail + risk
2) **Auto-Fixes Applied**: what was auto-fixed (format/lint), and whether lint is now clean
3) **Static Analysis Findings**: remaining Ruff + Pyright issues (top items only)
4) **Non-Trivial Findings**: schema/architecture concerns + decision gate (skip vs refactor)
5) **Next Steps**: ordered, minimal-diff
