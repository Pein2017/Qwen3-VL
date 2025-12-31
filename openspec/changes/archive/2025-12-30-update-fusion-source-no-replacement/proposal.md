# Proposal: update-fusion-source-no-replacement

## Problem
- Current fusion spec and implementation sample all source domains **with replacement** each epoch (online loader) and in offline fused JSONL builder. 
- For very large source pools relative to targets, replacement causes avoidable duplicate sampling within an epoch, wasting diversity without any upside, while the spec disallows alternative behavior.

## Goal
- Add an opt-in, per-source "sample without replacement" mode for fusion sources, keeping the default behavior (with replacement) intact for backward compatibility. 
- Apply the same policy to both online scheduler and offline fused JSONL builder so behavior stays aligned.
- Deterministically fall back to with-replacement only when `quota > pool_size`.

## Scope & Non-goals
- Scope: scheduling policy only; no changes to ratios, target sampling, or augmentation/cap policies. 
- Non-goals: changing defaults, altering target sampling, or introducing adaptive/epoch-varying ratios.

## Risks / Mitigations
- Risk: silent change to existing runs if default flipped — avoided by keeping default = with-replacement.
- Risk: edge case when quota exceeds pool size — handle by deterministic fallback to replacement and log/telemetry.

## Success Criteria
- New flag available per source in fusion config; default preserves existing behavior.
- Online and offline fusion honor the flag identically and remain deterministic (seeded per epoch/source).
- Specs/docs/tests updated and `openspec validate update-fusion-source-no-replacement --strict` passes.
