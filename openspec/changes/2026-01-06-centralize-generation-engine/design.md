# Design: Centralize generation/inference engine (HF + vLLM colocate)

## Constraints
- Must preserve existing call-site behavior where possible (refactor-only intent).
- Must support batch generation for:
  - text-only generation (Stage-B reflection / rule-search)
  - vision-language generation (Stage-A and `vis_tools`)
- Must remain compatible with existing distributed execution; the engine SHALL NOT own orchestration.
- Must comply with Schema Constitution for non-trivial structured inputs/outputs and configs.
- vLLM support is colocate-only (no server mode) in this change.
- Prompt rendering MUST disable model “thinking” blocks for all call sites (e.g., always attempt `enable_thinking=False` with
  a compatibility fallback when unsupported).

## Target environment (observed)
- Conda env: `ms`
- `transformers==4.57.1` and `vllm==0.11.0` are available in the target runtime image.

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
- `src/generation/stop_policy.py`
  - stop-policy normalization and deterministic post-truncation helpers shared across backends.
- `src/generation/plugins/base.py`
  - optional plugin protocol(s) for reusable generation-time behavior, with backend-specific hooks:
    - HF-only hooks: `StoppingCriteria` / `LogitsProcessor` injection
    - post-decode hooks: deterministic text trimming/cleanup (no task-specific parsing)
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
    - tokenizer-owned `apply_chat_template(..., enable_thinking=False)` (primary path)
    - processor-owned `apply_chat_template(..., enable_thinking=False)` for VLM models
    - compatibility fallback when `enable_thinking` is rejected by a particular implementation
    - processor-owned chat-template rendering for VL models
- `src/generation/preprocess.py`
  - processor/tokenizer normalization shared across call sites (padding-side, pad-token fallback, and VLM pixel budget knobs).
- `src/generation/engine.py`
  - single façade that wires backend + chat template + decoding:
    - `generate_text_batch(...)`
    - `generate_vlm_batch(...)`

## Result shape: two decoded text slots
`GenerationResult` SHOULD carry two decoded slots for maximum reuse across Stage-A/B and utility scripts:
- `text`: cleaned decoded assistant text (primary consumer path)
- `raw_text`: decoded text including special tokens (debugging, troubleshooting, and optional plugin consumers)

Both slots are plain decoded text; neither slot performs task-specific parsing.

## Stop handling: unified stop policy with deterministic truncation
Stop handling is a frequent drift source across call sites (Stage-B rollout, reflection/proposer, utility scripts).

The design SHOULD centralize stop policy in `GenerationOptions`, including:
- `stop`: stop strings (e.g., common chat terminators)
- `stop_token_ids`: optional explicit token-id stops

Backend wiring:
- HF backend:
  - treat single-token stop strings as EOS where possible (append to `eos_token_id` list)
  - treat explicit `stop_token_ids` as EOS (append to `eos_token_id` list)
  - apply deterministic final post-truncation on decoded text for **all** stop strings (multi-token and single-token)
- vLLM backend (0.11.0):
  - use `SamplingParams.stop_token_ids` as the canonical hard-stop mechanism (mirrors ms-swift’s “stop may be ineffective” note)
  - still pass `stop` strings when possible
  - apply the same deterministic post-truncation pass for reproducibility and parity

Silent fallback between backends remains forbidden; stop-policy behavior is caller-driven via options.

## Backend selection and fallback policy
The engine SHALL support explicit backend selection:
- `backend="hf"` for correctness-first behavior
- `backend="vllm"` for speed-first behavior

Because vLLM multimodal support may be model/version-dependent, the design SHOULD include an explicit fallback mechanism:
- either fail-fast with a clear `RuntimeError` (default)
- or allow a configured fallback backend (`fallback_backend="hf"`) for unsupported requests

Silent fallback is forbidden (would hide perf regressions and complicate reproducibility).

## Batch-first API sketch
- `GenerationEngine.generate_text_batch(requests: Sequence[TextGenerationRequest], options: GenerationOptions, plugins: Sequence[GenerationPlugin] | None = None) -> list[GenerationResult]`
- `GenerationEngine.generate_vlm_batch(requests: Sequence[VlmGenerationRequest], options: GenerationOptions, plugins: Sequence[GenerationPlugin] | None = None) -> list[GenerationResult]`

Where each request contains:
- chat messages (system/user/assistant turns)
- for VLM: exactly one image per request in v1 (multi-image can be added later)

## VLM preprocessing and processor normalization (must not regress Stage-A)
Stage-A currently relies on processor-level normalization that must remain stable after refactor:
- left-padding and left-truncation for decoder-only batching
- pad-token fallback (`pad_token := eos_token` when missing)
- image pixel budget normalization (Stage-A mutates `image_processor.size` / `min_pixels` / `max_pixels`)

This proposal recommends centralizing these behaviors in the engine loader/preprocess layer:
- `ModelLoadConfig` and/or a nested `VlmPreprocessOptions` should carry `max_pixels` and optional overrides such as `do_resize`.
- HF backend SHOULD pass pixel budgets via `images_kwargs` (as Stage-A currently does).
- vLLM backend SHOULD pass pixel budgets via `mm_processor_kwargs` (vLLM `TextPrompt` supports `mm_processor_kwargs` in 0.11.0).

## Plugins: optional, reusable generation-time behavior
Utility scripts (e.g., `vis_tools/vis_qwen3.py`) may require advanced generation-time controls such as:
- balanced-JSON stopping criteria
- duplicate-object suppression via logits processors

These behaviors SHOULD be supported via an optional plugin mechanism:
- absence of plugins implies baseline behavior (no extra stopping criteria/logits processors)
- plugins are explicitly provided by the caller (no implicit defaults)
- plugins MUST NOT perform task-specific parsing; only generation-time controls and deterministic text trimming/cleanup

### Plugin compatibility and early rejection
Some plugin capabilities are backend-specific (e.g., HF `StoppingCriteria` / `LogitsProcessor` hooks have no direct vLLM
equivalent). The design SHOULD reject incompatible plugin usage as early as possible:
- At plugin registration time (when the selected backend is known), or
- Immediately before generation begins (preflight validation), rather than during token generation.

The plugin protocol SHOULD expose a small, declarative compatibility surface (e.g., supported backends and required
capabilities). The engine SHOULD validate plugins against:
- the selected backend (`hf` vs `vllm`)
- request type (text-only vs VLM)
- backend capability flags (e.g., `supports_vlm`)

When incompatible, the engine MUST raise a clear error before any generation work is started (fail-fast, no partial output).

## Migration approach
- Start with Stage-B (text-only) to validate engine surface and batching.
- Migrate Stage-A next (VLM batch); preserve pixel-budget and image preprocessing behavior.
- Migrate `vis_tools` last to avoid breaking operator workflows; keep any custom `StoppingCriteria`/`LogitsProcessor` local.
