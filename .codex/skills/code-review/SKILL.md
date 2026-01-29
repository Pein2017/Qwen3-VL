---
name: code-review
description: "Fail-fast Python code review: ruff (format + lint) + pyright-compatible type checking, plus Schema Constitution + architecture audit checklist. Uses standard CLI tools (no helper scripts) for portability across machines."
---

# Python Code Review (fail-fast static + schema + architecture)

This skill is intentionally script-free: it relies on standard, portable CLI tools (`ruff`, `pyright`/`basedpyright`) plus a lightweight manual audit checklist for Schema Constitution and architecture.

- **Static analysis**: `ruff` (format + lint)
- **Type checking**: `pyright` or `basedpyright`
- **Schema Constitution compliance**: audit against `docs/reference/SCHEMA_CONSTITUTION.md`
- **Architecture & design**: validate coupling/cycles/leaky abstractions with concrete evidence

## 0) Define review scope (mandatory for control)

- **Explicit paths required**: target specific Python files/dirs only (e.g., `src/`, `tests/`)
- **Prefer small scopes**: start with the files you touched; expand only if needed

## 1) Run ruff (format + lint)

**Format check (no changes)**:
```bash
conda run -n ms ruff format --check --diff src tests
```

**Lint check (no changes)**:
```bash
conda run -n ms ruff check src tests
```

**Auto-fix (optional)**:
```bash
conda run -n ms ruff check --fix src tests
conda run -n ms ruff format src tests
```

## 2) Run a type checker (pyright-compatible)

Use whichever is available in the `ms` env:
```bash
conda run -n ms pyright src tests
```

Or:
```bash
conda run -n ms basedpyright src tests
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

## 5) Generate a review report (concise + actionable)

Suggested structure:

1) **Status**: Clean / Requires Fixes / Blocked (constitution/architecture violations)
2) **Auto-Fixes**: Formatting/lint changes applied (if any)
3) **Critical Issues**: Schema/architecture violations requiring immediate action
4) **Static Issues**: Remaining ruff/pyright problems with remediation steps
5) **Next Actions**: Ordered, minimal-diff fixes (no optional steps)

**Enforcement**: Constitution and architecture violations always require immediate fixes. No "skip" options for critical issues.
