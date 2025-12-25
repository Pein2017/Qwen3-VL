# Proposal: Enforce AND Semantics and Drop Overlength Stage-B Prompts

## Why
Stage-B rollout in bbu_cable shows elevated false positives when guidance rules are treated as loosely optional instead of conjunctive. In addition, prompt truncation during rollout/proposer/reflection can silently remove evidence, producing inconsistent decisions and undermining rule-search quality.

## What
- Make rollout prompt explicitly treat guidance rules as an AND relationship by default; OR behavior is only allowed when a rule text explicitly states it.
- Enforce strict prompt-length handling: when a prompt exceeds the configured budget, drop the entire ticket/example instead of truncating.
- Standardize rollout prompt budgets across all Stage-B missions: rollout max prompt tokens = 4096; proposer/reflection max length = 12000.
- Add structured logging for overlength drops to aid monitoring.

## Impact
- Stage-B behavior changes for all missions: rollout decisions become stricter and more consistent with conjunctive rules.
- Overlength prompts no longer get truncated, which may reduce usable samples but improves correctness and auditability.
- Config updates are required across `configs/stage_b/*.yaml`.

## Success Criteria
- Rollout prompt explicitly encodes AND semantics for guidance rules and requires evidence coverage.
- Overlength prompts are dropped (not truncated) in rollout and proposer/reflection paths.
- All Stage-B configs set rollout max prompt tokens to 4096 while keeping proposer/reflection max length at 12000.
- Logs surface dropped ticket counts and token budgets per run.

## Risks / Trade-offs
- Higher drop rates could reduce effective sample sizes and slow rule-search convergence.
- Stricter AND semantics may increase false negatives until guidance rules adapt.
- Additional logging may slightly increase IO overhead.
