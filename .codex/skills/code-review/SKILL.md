---
name: code-review
description: "Fail-fast Python code review: strict ruff (format + lint) + pyright-compatible type checking, Schema Constitution audit, and architecture review. Never silently ignores critical issues - fails fast on violations with explicit error codes. Use for pre-commit hooks, CI/CD gates, or ensuring code quality standards."
---

# Python Code Review (fail-fast static + schema + architecture)

**Fail-fast principle**: This skill stops immediately on critical issues and never silently continues. Every scenario is explicitly handled with clear error codes and remediation paths.

- **Static analysis**: `ruff` (format + lint) + `pyright`-compatible type checker with zero tolerance for errors
- **Schema Constitution compliance**: strict audit against `docs/reference/SCHEMA_CONSTITUTION.md`
- **Architecture & design**: validates coupling/cycles/leaky abstractions with concrete evidence

## 0) Define review scope (mandatory for control)

**Fail-fast requirement**: Scope must be explicitly defined to prevent uncontrolled execution.

- **Explicit paths required**: Target specific Python files/dirs only (e.g., `src/core`, `tests/`)
- **Git changeset**: Use `--git-diff HEAD~1` for targeted reviews
- **No default scope**: Fails if no `--paths` or `--git-diff` specified

## 1) Execute automated collector (fail-fast on critical issues)

**Exit codes**: 0 (clean), 1 (lint/format violations), 2 (type errors), 3 (constitution violations), 4 (architecture violations)

Script behavior:
- **Format enforcement**: Applies `ruff format` (fails on unformattable code)
- **Safe auto-fixes**: Applies `ruff check --fix` (fails on unfixable violations)
- **Strict mode**: `--mode strict` enables zero-tolerance for all issue types

Artifacts (when `--write-artifacts` enabled):
- Raw outputs: `ruff_check.json`, `ruff_format.json`, `pyright.json`
- Normalized issues: `issues.json`
- Summary report: `summary.json`

**Primary command (strict mode)**:
```bash
conda run -n ms python .codex_config/else/skills/code-review/scripts/code_review.py --mode strict --paths src
```

**Multi-target review**:
```bash
conda run -n ms python .codex_config/else/skills/code-review/scripts/code_review.py --mode strict --paths src tests
```

**Tool requirements**: `pyright` or `basedpyright` must be installed in `ms` env. Fails immediately if neither available.

**CI/CD usage** (fail-fast):
```bash
conda run -n ms python .codex_config/else/skills/code-review/scripts/code_review.py --mode strict --paths src --exit-on-any-issue
```

**Validation-only** (no fixes, fail on issues):
```bash
conda run -n ms python .codex_config/else/skills/code-review/scripts/code_review.py --mode strict --no-fix --no-format --format-check --paths src
```

**With artifacts** (for debugging):
```bash
conda run -n ms python .codex_config/else/skills/code-review/scripts/code_review.py --mode strict --write-artifacts --output-dir tmp/review --paths src
```

**Issue limits**: `--max-issues 0` shows all issues (default: 200). Artifacts always contain full issue list.

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

## 3) Constitution compliance audit (strict clause enforcement)

**Fail-fast**: Constitution violations are critical errors that must be fixed immediately.

Required actions:
1) **Verify findings**: Cross-reference `docs/reference/SCHEMA_CONSTITUTION.md` for each violation
2) **Document violations** with:
   - **Clause reference**: Exact section title (e.g., "Function signatures and returns")
   - **Location**: File path and line number
   - **Violation**: Specific rule broken (non-trivial mapping, missing validation, etc.)
   - **Impact**: How it affects correctness/maintainability
   - **Fix**: Minimal structured-type refactor required

## 4) Architecture & design validation (evidence-based enforcement)

**Fail-fast**: Architecture violations block progression and require immediate remediation.

Validates against:
- **Tight coupling**: Excessive direct imports, wrong dependency direction
- **Leaky abstractions**: Public APIs exposing internal dicts/lists or file layouts
- **Anti-patterns**: Global mutable state, hidden I/O, mixed responsibilities, monolithic modules
- **Testability**: Components must be constructable without external dependencies

**Evidence requirement**: Every violation must include concrete artifacts:
- Import cycle chains
- API signatures with problematic parameters
- Boundary locations missing validation
- File paths showing architectural violations

## 5) Generate fail-fast report (structured enforcement)

**Exit behavior**: Non-zero exit code on any critical findings. Never silently continues.

Report structure (concise, actionable):
1) **Status**: Clean / Requires Fixes / Blocked (constitution/architecture violations)
2) **Auto-Fixes**: Applied formatting/linting changes with success confirmation
3) **Critical Issues**: Schema/architecture violations requiring immediate action
4) **Static Issues**: Remaining ruff/pyright problems with remediation steps
5) **Next Actions**: Ordered, minimal-diff fixes (no optional steps)

**Enforcement**: Constitution and architecture violations always require immediate fixes. No "skip" options for critical issues.
