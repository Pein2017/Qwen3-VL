# Proposal: Refactor GRPO Rewards + Config Surface

## Why
- GRPO summary reward implementation is highly repetitive (loop/parsing/gating blocks repeated across many reward classes), increasing maintenance cost and risk of inconsistencies.
- GRPO-specific configuration (notably CHORD) currently lives in `custom.extra` and is validated inline in `src/sft.py`, which fragments config validation and obscures the public config surface.
- Reward registration is triggered in multiple places, creating implicit side effects and making dependency ordering harder to reason about.

## Scope
- Refactor `src/rlhf/` GRPO reward code into shared parsing/context utilities and simplified reward classes.
- Standardize GRPO configuration validation (including CHORD) in a dedicated validator and promote typed config fields.
- Rename reward identifiers to a namespaced dot form and update `configs/grpo/*` to match (no backward compatibility).
- Update documentation that describes GRPO configuration and reward identifiers.

## Non-Goals
- Altering GRPO algorithmic behavior or reward semantics.
- Changing ms-swift interfaces or trainer behavior.
- Adding new reward logic or new GRPO training features beyond refactor and configuration cleanup.

## Compatibility
- **Breaking change**: legacy reward identifiers (e.g., `summary_format`) and legacy config path `custom.grpo_chord` will be removed.
- All GRPO configs must be updated to the new reward identifiers and `custom.grpo.chord` structure.

## Risks
- Misalignment between new reward identifier names and ms-swift reward registry if registration is incomplete.
- Configuration validation regressions for GRPO-only settings (CHORD + reward list/weights) if the new validator is not exhaustive.

## Success Criteria
- All GRPO configs in `configs/grpo/` validate under the new schema.
- `pytest tests/test_summary_grpo_rewards.py` passes.
- Duplicated GRPO reward parsing and gating logic is consolidated into shared utilities with per-reward implementations reduced to scoring logic only.
