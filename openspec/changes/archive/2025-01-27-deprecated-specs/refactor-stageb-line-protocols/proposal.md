# Refactor Stage-B LLM Outputs To Line Protocols

## Context
- High-temperature decoding frequently yields malformed JSON from the Critic and Reflection prompts, causing parser failures or fallback coercion.
- Stage-B already stores artifacts as JSONL after parsing; only the LLM-facing format needs simplification.
- We want deterministic, auditable outputs while reducing generation fragility.

## Problem
- Strict “LLM must emit JSON” in critic/reflection leads to high failure rates under exploration grids.
- Current fallbacks (_fallback_coerce, semi-structured JSON parsing) obscure format drift and complicate monitoring.

## Goals
- Replace Critic and Reflection generation formats with explicit line-based protocols (key/value lines, operation lines).
- Parse line protocols into structured objects inside Stage-B, then persist as JSONL unchanged for audit/replay.
- Remove JSON-generation requirement from prompts; treat JSON parsing as optional legacy (off by default).
- Surface format violations as errors/metrics instead of silent coercion.

## Non-Goals
- No change to Stage-A JSONL contract or ingest.
- No change to stored JSONL schemas (`trajectories.jsonl`, `selections.jsonl`, `reflection.jsonl`).
- No changes to selection policy or reflection eligibility logic beyond input format handling.

## Impact / Risks
- Requires prompt updates and new parsers; format drift must be monitored (add violation counters).
- Backward compatibility: optional legacy JSON parsing switch may be kept for safety during rollout.

## Timeline / Rollout
- Implement parsers + prompts behind a gated config, run `openspec validate` before sharing.
