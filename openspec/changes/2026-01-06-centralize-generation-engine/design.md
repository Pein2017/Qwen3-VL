# Design: Centralize generation/inference engine (HF + vLLM colocate)

## Constraints
- Must preserve existing call-site behavior where possible (refactor-only intent).
- Must support batch generation for:
  - text-only generation (Stage-B reflection / rule-search)
  - vision-language generation (Stage-A and `vis_tools`)
- Must remain compatible with existing distributed execution; the engine SHALL NOT own orchestration.
- Must comply with Schema Constitution for non-trivial structured inputs/outputs and configs.
- vLLM support is colocate-only (no server mode) in this change.

## Background: ms-swift vLLM/GRPO rollout
ms-swift already provides:
- a colocate vLLM engine abstraction
- a GRPO-specific vLLM rollout engine with training-time weight/adapter sync

This repository’s centralized engine is NOT a training rollout replacement.
Instead, it is an evaluation/deployment generator that can optionally reuse vLLM integration patterns learned from ms-swift:
- engine kwargs normalization and version feature-detection
- robust stop handling (`stop` and token-id stops)
- multimodal request wiring for Qwen3-VL

## Proposed module layout
- `src/generation/contracts.py`
  - dataclasses for structured configs and request/response types:
    - `ModelLoadConfig`, `GenerationOptions`
    - `TextGenerationRequest`, `VlmGenerationRequest`
    - `GenerationResult`
- `src/generation/backends/base.py`
  - `GenerationBackend` protocol:
    - `generate_text_batch(...)`
    - `generate_vlm_batch(...)`
    - capability flags (`supports_vlm`, `supports_token_ids`, etc.)
- `src/generation/backends/hf_backend.py`
  - transformers-based backend (safe default)
  - supports text-only + image+text batch generation
- `src/generation/backends/vllm_backend.py`
  - vLLM colocate backend (fast path when available)
  - supports text-only batch generation
  - supports image+text batch generation when vLLM+model supports multimodal (best-effort with explicit capability checks)
- `src/generation/chat_template.py`
  - chat-template rendering helpers with safe feature detection:
    - tokenizer-owned `apply_chat_template(..., enable_thinking=False)` when supported
    - fallback paths when `enable_thinking` is unsupported
    - processor-owned chat-template rendering for VL models
- `src/generation/engine.py`
  - single façade that wires backend + chat template + decoding:
    - `generate_text_batch(...)`
    - `generate_vlm_batch(...)`

## Backend selection and fallback policy
The engine SHALL support explicit backend selection:
- `backend="hf"` for correctness-first behavior
- `backend="vllm"` for speed-first behavior

Because vLLM multimodal support may be model/version-dependent, the design SHOULD include an explicit fallback mechanism:
- either fail-fast with a clear `RuntimeError` (default)
- or allow a configured fallback backend (`fallback_backend="hf"`) for unsupported requests

Silent fallback is forbidden (would hide perf regressions and complicate reproducibility).

## Batch-first API sketch
- `GenerationEngine.generate_text_batch(requests: Sequence[TextGenerationRequest], options: GenerationOptions) -> list[GenerationResult]`
- `GenerationEngine.generate_vlm_batch(requests: Sequence[VlmGenerationRequest], options: GenerationOptions) -> list[GenerationResult]`

Where each request contains:
- chat messages (system/user/assistant turns)
- for VLM: exactly one image per request in v1 (multi-image can be added later)

## Migration approach
- Start with Stage-B (text-only) to validate engine surface and batching.
- Migrate Stage-A next (VLM batch); preserve pixel-budget and image preprocessing behavior.
- Migrate `vis_tools` last to avoid breaking operator workflows; keep any custom `StoppingCriteria`/`LogitsProcessor` local.

