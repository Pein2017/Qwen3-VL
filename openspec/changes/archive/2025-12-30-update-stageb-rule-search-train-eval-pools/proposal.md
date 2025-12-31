# Proposal: Update Stage-B Rule-Search to Train/Eval Pools

## Why
Current rule_search uses a validation subset for both proposal generation and gate evaluation, and the naming (`eval_sampler`, `validate_size`) is ambiguous with respect to training vs evaluation. This makes it hard to reason about sample usage and to implement a workflow that reuses all tickets efficiently while still keeping a small holdout for veto/monitoring.

## What
Introduce **train/eval pool** semantics and rename rule_search fields accordingly:
- A **train pool** is used for rollout, proposal generation, and gate evaluation (same batch).
- An **eval pool** (holdout, fixed for the run) is used only for monitoring and veto, with a configurable tolerance for degradation.

Key additions:
- Train pool size (default 512) and rotation across the dataset.
- Eval pool fraction (default 0.2), fixed for the run.
- Rename sampler fields to `train_sampler` and `eval_sampler` to reflect usage.
- Add an eval-pool veto rule: accept candidate only if eval metric does not degrade beyond a threshold (default 0.01 absolute acc drop).
- Remove legacy/compat parsing; only new train/eval naming is supported.

## Impact
- Configuration breaking changes; legacy keys are removed without compatibility shims.
- Rule-search behavior changes: proposal + gate are computed on the same train pool; eval pool only vetoes/regresses.
- Docs and tests must be updated to reflect train/eval semantics.

## Success Criteria
- Rule-search can iterate pools of size 512, propose top-32 hard cases, and gate on the same pool.
- Eval pool remains fixed and can veto candidate rules when degradation exceeds threshold.
- Configuration and outputs clearly distinguish train vs eval usage.

## Risks / Trade-offs
- Train-pool gate may overestimate gains (information leakage). Eval pool only partially mitigates.
- If eval pool is too small or uses too few samples, veto becomes noisy.
