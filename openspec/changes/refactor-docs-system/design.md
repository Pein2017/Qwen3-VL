# Design: Documentation System Redesign

## Goals
- Single, coherent information architecture with predictable navigation.
- Consistent document formatting and metadata for faster upkeep.
- Clear mapping between code directories and documentation owners.

## Information Architecture (Target)
Proposed top-level layout (subject to audit findings):

```
docs/
  README.md                  # global index + directory map
  overview/                  # project overview, architecture, glossary
  data/                      # data pipeline, contracts, datasets, augmentation
  training/                  # training playbooks + reference
  runtime/                   # stage A/B runtime + diagnostics
  ops/                       # setup, deployment, platform dependencies
  reference/                 # prompts, business knowledge, long-form references
```

Notes:
- Existing `docs/data/`, `docs/training/`, and `docs/runtime/` remain but are normalized.
- `docs/setup/` and `docs/platform/` are merged under `docs/ops/`.
- `docs/stage_b/` and `docs/reference/stage-B-knowledge-Chinese.md` are consolidated under `docs/reference/`.
- `docs/draft/` is removed; content is either upgraded to a target section or deleted if obsolete.

## Standard Document Template
Each document should follow a lightweight, consistent header:

- `# Title`
- `Status:` (Active/Deprecated/Draft)
- `Scope:` (one sentence)
- `Owners:` (team or function)
- `Last updated:` (YYYY-MM-DD)
- `Related:` (key cross-links)

Section ordering should favor:
1) Purpose / Context
2) Main content (procedures, contracts, tables)
3) References / related docs

## Navigation Rules
- `docs/README.md` is the canonical index and must list every doc.
- Each top-level section includes a short index (`README.md`) that links to its children.
- All links are relative and kept up to date during moves.

## Migration Strategy
- Build a file-by-file audit log to ensure full coverage.
- Map each existing file to its new location and update links.
- Remove stale sections in place; use Git history for legacy traceability.
