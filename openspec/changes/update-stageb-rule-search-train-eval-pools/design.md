# Design: Train/Eval Pools for Rule-Search

## Overview
Rule-search operates on **train pools** (proposal + gate) and a fixed **eval pool** (monitor/veto).
Naming is unified to make sampling intent explicit (`train_*` vs `eval_*`).

## Pooling
1. **Eval pool (fixed)**:
   - Sample once at run start from full dataset.
   - Size controlled by `rule_search.eval_pool_size` (default 128).
   - Excluded from train pools.
2. **Train pool (rolling)**:
   - Remaining tickets are shuffled once using seed.
   - Iteration uses the next `train_pool_size` tickets.
   - If end is reached, wrap and reshuffle (optional) and continue; early-stop is the only termination.

## Iteration Flow
For each iteration:
1. Rollout on **train pool** using `train_sampler`.
2. Compute per-ticket stats; select top‑K hard cases (`reflect_size`, default 32).
3. Propose candidate rules.
4. Gate candidates on the **same train pool** (A/B).
5. Evaluate accepted candidate on **eval pool** using `eval_sampler`.
6. If eval degradation exceeds threshold, veto candidate.

## Acceptance Criteria
Candidate accepted iff:
- Train gate passes (existing RER/changed_fraction/bootstrap thresholds)
- Eval pool metric does **not** degrade beyond `eval_veto.max_acc_drop` (default 0.01).

## Naming Changes (config)
- `rule_search.train_pool_size` (replaces `validate_size`)
- `rule_search.train_pool_fraction` (replaces `validate_fraction`)
- `rule_search.train_with_replacement` (replaces `validate_with_replacement`)
- `rule_search.train_sampler` (replaces `eval_sampler` for train/gate sampling)
- `rule_search.eval_pool_size` (new)
- `rule_search.eval_sampler` (new; used only for holdout evaluation)

## Compatibility
Legacy keys are removed; no backward-compat parsing is provided.
