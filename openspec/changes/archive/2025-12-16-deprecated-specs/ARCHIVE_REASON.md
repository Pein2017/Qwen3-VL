# Archive Reason: Deprecated/Not Implemented Specs (2025-12-16)

## Summary
`add-multi-object-line-copy-paste` was archived because none of its proposed augmentation ops or helpers exist in the codebase and no tasks were completed. The repo already ships padding-only augmentation paths without patch-bank copy/paste, so the speculative design is stale.

## Archived Specs

### 1. `add-multi-object-line-copy-paste`
**Status**: Not Implemented (never started)

**Evidence**
- All tasks in `tasks.md` remain unchecked; no accompanying tests or configs were added.
- Code search shows no implementations of the proposed ops (`ObjectClusterCopyPaste`, `LineSegmentCopyPaste`, patch-bank helpers) in `src/datasets/augmentation/` (`rg "ObjectClusterCopyPaste"` returns no results).
- No config entries or docs mention the new ops; the active augmentation registry only includes existing PatchOps (zoom/paste, smart resize, etc.).

**Decision**
- Marked deprecated to avoid confusion with the current augmentation roadmap and padding-only runtime. Can be resurrected later with a fresh proposal if multi-object/line copy-paste is reprioritized.
