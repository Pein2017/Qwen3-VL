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
- The engine SHALL render prompts with model “thinking” disabled (e.g., `enable_thinking=False`) for **all** call sites
  (Stage-A, Stage-B, and `vis_tools`) to keep outputs deterministic and parseable.
- Output parsing and task-specific post-processing (dense JSON extraction, summary JSON extraction, Stage‑B verdict protocols) SHALL remain in existing, task-specific modules.
- Optional, reusable generation-time features (e.g., HF `StoppingCriteria` / `LogitsProcessor` hooks and stop-policy normalization) MAY be implemented as engine “plugins” so that utility scripts (e.g., `vis_tools`) can share behavior without forcing it on Stage-A/B.
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
  - unified stop-policy handling for common generation termination patterns (stop strings and stop token ids), with backend-appropriate wiring and deterministic post-truncation
  - a stable decoded-text result shape with two slots:
    - `text`: cleaned decoded assistant text (default consumer path)
    - `raw_text`: decoded text including special tokens (debugging and optional plugin consumers)
  - structured configuration objects (Schema Constitution compliant) to reduce “dict soup” at call sites
  - optional plugin hooks to reuse advanced HF-only controls (e.g., stopping criteria / logits processors) in `vis_tools` without coupling them into Stage-A/B default behavior
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
- Disabling “thinking” everywhere may change output shape for some checkpoints; this change treats “no-thinking” as an explicit,
  intended behavior and should be validated on representative Stage-A and `vis_tools` workflows.
