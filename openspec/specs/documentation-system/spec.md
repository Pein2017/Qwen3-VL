# documentation-system Specification

## Purpose
Define the documentation system standards for structure, metadata, link integrity, and audit traceability.
## Requirements
### Requirement: Unified Documentation Architecture
The documentation set under `docs/` MUST follow a single, defined information architecture with a global index and section indices.

#### Scenario: Global and section indices exist
- WHEN the documentation base is reorganized
- THEN `docs/README.md` lists all documentation files and top-level sections
- AND each top-level section includes an index that links to its documents

### Requirement: Standard Document Metadata
Every documentation file MUST include a standard metadata header to improve consistency and maintenance.

#### Scenario: Metadata header present
- WHEN a documentation file is opened
- THEN it includes `Status`, `Scope`, `Owners`, and `Last updated` fields near the top

#### Scenario: Metadata values are standardized
- WHEN a metadata header is authored
- THEN `Status` MUST be one of `Active`, `Deprecated`, or `Draft`
- AND `Owners` MUST name a team, function, or system role (not a personal name)

### Requirement: Accuracy Against Current Code
Documentation content MUST reflect the current codebase and configuration surface under `src/`, `scripts/`, `configs/`, `data_conversion/`, `public_data/`, and related directories; outdated or deprecated guidance MUST be removed or explicitly marked as deprecated.

#### Scenario: Stale guidance removed
- WHEN a module, script, or configuration option no longer exists
- THEN references to it are removed or replaced with current equivalents
- AND any remaining legacy references are labeled as deprecated with a reason

#### Scenario: Codebase alignment verified
- WHEN a documentation file is updated
- THEN its statements about commands, file paths, and configuration options are validated against the current codebase

### Requirement: Link Integrity
All internal documentation links MUST be relative and remain valid after reorganization.

#### Scenario: Links remain functional after moves
- WHEN documentation files are moved or renamed
- THEN every internal link is updated to the new relative path
- AND no internal links point to missing files

### Requirement: Repository-Wide Reference Updates
All Markdown files in the workspace MUST be audited for documentation references, and any reference to a moved or renamed documentation file MUST be updated in-place (no compatibility stubs).

#### Scenario: Cross-repo markdown references are updated
- WHEN a documentation file path changes
- THEN every Markdown file in the repository that references it is updated to the new path
- AND no compatibility or redirect stub is left at the old path

### Requirement: In-Repository Path References
Documentation MUST NOT hard-code absolute filesystem paths outside the repository. For in-repo references, paths MUST be repo-relative; for external dependencies, use module-qualified identifiers or placeholders instead of absolute filesystem locations.

#### Scenario: Paths are repo-relative or placeholder-based
- WHEN a documentation file references a file path
- THEN in-repo paths are expressed relative to the repository root
- AND external dependencies are referenced via module identifiers or placeholders (not absolute filesystem paths)
- AND no absolute filesystem paths outside the repository are used

### Requirement: No Duplicate or Orphaned Content
Documentation MUST avoid duplicated content and MUST ensure every documentation file is indexed in its section README and the global index.

#### Scenario: Duplicated topics are consolidated
- WHEN two or more documentation files cover the same topic
- THEN content is consolidated into a single canonical document
- AND references are updated to the canonical document

#### Scenario: Orphaned docs are eliminated
- WHEN a documentation file exists under `docs/`
- THEN it is linked from its section README and from `docs/README.md`

### Requirement: Audit Log Location and Format
The audit log MUST be stored at `docs/overview/AUDIT_LOG.md` and MUST include each documentation file path, the review outcome, and the review date.

#### Scenario: Audit log is complete and consistent
- WHEN the documentation audit is completed
- THEN `docs/overview/AUDIT_LOG.md` lists every file under `docs/` with a status and date

### Requirement: Audit Traceability
An audit record MUST exist that shows each documentation file was reviewed against the current codebase.

#### Scenario: Per-file audit log
- WHEN the documentation audit is completed
- THEN an audit log lists each file under `docs/` with its review outcome (updated, moved, removed, or unchanged)

