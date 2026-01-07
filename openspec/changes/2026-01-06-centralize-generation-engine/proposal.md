# Proposal: Centralize generation/inference engine (HF + vLLM colocate)

## Why
- Generation/inference logic is currently duplicated across:
  - Stage-A inference (`src/stage_a/inference.py`) — Qwen3-VL load + processor setup + `generate()` for image summaries
  - Stage-B tooling (`src/stage_b/runner.py`, `src/stage_b/reflection/engine.py`, `src/stage_b/rollout.py`) — model/tokenizer load + chat-template rendering + `generate()` for rule-search/reflection
  - Visualization/eval scripts (`vis_tools/vis_qwen3.py`) — Qwen3-VL load + processor setup + `generate()` for ad-hoc inference and dumps
- Duplication increases drift risk (padding-side, attention impl, dtype/device placement, prompt rendering, decoding knobs).
- Post-training evaluation and deployment workflows benefit from a single, testable “engine” surface that can be reused by:
  - offline eval scripts (dense dumps, spot checks)
  - Stage-A/B runtime entrypoints
  - visualization utilities

## Clarified scope (this proposal)
- The centralized engine SHALL focus exclusively on:
  - model/tokenizer/processor loading
  - prompt rendering / chat-template application
  - generation invocation and decoding (batch-first)
- Output parsing and downstream post-processing (dense JSON extraction, summary JSON extraction, Stage‑B verdict protocols) SHALL remain in existing, task-specific modules.
- vLLM integration SHALL be **colocate only** (no server mode in this change).
- The engine SHALL provide:
  - a “safe” Hugging Face backend (transformers `generate`)
  - a “fast” vLLM backend (colocate), when available

## Relationship to ms-swift GRPO rollout
- ms-swift already implements a GRPO rollout engine with vLLM integration (including weight/adapter sync and training-time concerns).
- This change SHALL NOT re-implement or override ms-swift’s GRPO rollout engine.
- The centralized engine is intended for evaluation and deployment entrypoints in this repository; training-time rollout remains owned by ms-swift.

## What changes
- Add a centralized **generation/inference engine** under `src/` that provides:
  - shared model/tokenizer/processor loading helpers (HF + vLLM)
  - shared chat-template prompt rendering (tokenizer vs processor ownership, safe “no-thinking” fallback)
  - shared batch `generate()` wrappers for:
    - text-only generation (Stage-B reflection / rule-search)
    - vision-language generation (Qwen3-VL image+text inference)
  - structured configuration objects (Schema Constitution compliant) to reduce “dict soup” at call sites
- Migrate existing call sites to the centralized engine, minimizing behavior changes:
  - Stage-A: loading + per-image/batch inference delegates to the engine
  - Stage-B: loading + generation blocks delegate to the engine
  - vis_tools: inference path delegates to the engine (keep CLI UX stable)

## Non-goals
- Changing model quality/decoding behavior intentionally.
- Consolidating output parsing or evaluation logic.
- Adding vLLM server mode.
- Rewriting Stage-A/B distributed orchestration (only generation blocks are centralized).
- Replacing ms-swift training-time rollout code.

## Impact / breaking changes
- Intended to be behavior-preserving (refactor-only intent for callers).
- Minor risk of incidental behavior changes if defaults diverge; mitigations:
  - keep generation knobs caller-owned
  - add smoke checks for prompt rendering + decoding
  - ensure padding-side/pad-token and chat-template behavior remains consistent

## Success criteria
- A single “engine” module becomes the source of truth for:
  - HF/vLLM model loading and configuration normalization
  - chat-template prompt rendering
  - batch generation invocation + decode
- `vis_tools/vis_qwen3.py`, `src/stage_a/inference.py`, `src/stage_b/runner.py`, `src/stage_b/reflection/engine.py`,
  and `src/stage_b/rollout.py` no longer duplicate model-loading and prompt-rendering logic.

## Risks
- vLLM multimodal parity varies by model + vLLM version; the vLLM backend MUST fail fast with a clear error when unsupported,
  and optionally support explicit fallback to the HF backend when configured.
- Hidden coupling: some scripts may rely on implicit defaults (`trust_remote_code`, attention impl, pad-token fallback).

