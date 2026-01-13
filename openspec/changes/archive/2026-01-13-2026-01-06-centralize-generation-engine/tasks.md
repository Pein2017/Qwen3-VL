# Tasks: Centralize generation/inference engine (HF + vLLM colocate)

- [x] Pre-implementation audit
  - [x] Inventory current generation entrypoints and duplicated blocks:
    - [x] `src/stage_a/inference.py` (load + encode + generate + decode)
    - [x] `src/stage_b/runner.py` (load + tokenizer settings)
    - [x] `src/stage_b/reflection/engine.py` (apply_chat_template + generate + decode)
    - [x] `src/stage_b/rollout.py` (batched generate + EOS/stop handling)
    - [x] `vis_tools/vis_qwen3.py` (load + apply_chat_template + generate + decode)
  - [x] Confirm supported model variants:
    - [x] VL Qwen3-VL checkpoints (Stage-A / vis_tools)
    - [x] text-only checkpoints (Stage-B)
  - [x] Confirm vLLM availability + multimodal support constraints in the target environment.
    - [x] Confirm `conda run -n ms python -c "import vllm; print(vllm.__version__)"` succeeds (expected `0.11.0` in the current runtime image).
    - [x] Confirm vLLM prompt schema supports `TextPrompt` with `multi_modal_data` and `mm_processor_kwargs` (vLLM `TextPrompt` TypedDict).
  - [x] Confirm ms-swift GRPO rollout remains out-of-scope (no reimplementation).

- [x] Engine scaffolding (`src/generation/`)
  - [x] Add Schema-Constitution-compliant config + contract types:
    - [x] `ModelLoadConfig`
    - [x] `GenerationOptions`
    - [x] `StopOptions` (or equivalent fields in `GenerationOptions`) for `stop` strings and `stop_token_ids`
    - [x] `ChatTemplateOptions` (include “no-thinking” default; caller override optional)
    - [x] `VlmPreprocessOptions` (pixel budget + optional resize knobs) to preserve Stage-A behavior
    - [x] request/response dataclasses for batch APIs
    - [x] Ensure `GenerationResult` has two decoded slots: `text` and `raw_text`
  - [x] Add backend protocol + façade `GenerationEngine`.
  - [x] Implement shared chat-template rendering helpers (tokenizer vs processor ownership).
    - [x] Ensure prompt rendering always attempts `enable_thinking=False` (no-thinking for all calls) with a compatibility fallback.
  - [x] Implement stop policy helpers (single-token stop -> token-id stop + deterministic post-truncation on decoded text).
  - [x] Add optional plugin protocol for reusable generation-time behavior (HF stopping criteria / logits processors + post-decode trimming).
  - [x] Implement plugin compatibility preflight validation (fail fast before generation begins on incompatible backend/request type).

- [x] Implement HF backend (safe)
  - [x] Text-only batch generation (encode via tokenizer, `model.generate`, decode).
  - [x] VLM batch generation (encode via processor, `model.generate`, decode).
  - [x] Implement unified stop-policy behavior (stop strings + stop token ids) with deterministic post-truncation.
  - [x] Wire optional plugin hooks (HF-only `StoppingCriteria` / `LogitsProcessor`) without changing baseline behavior when no plugins are provided.
  - [x] Return both cleaned decoded text and minimal token accounting (best-effort).

- [x] Implement vLLM colocate backend (fast)
  - [x] Text-only batch generation via vLLM engine.
  - [x] VLM batch generation when supported:
    - [x] Implement capability checks for model+vLLM version
    - [x] Fail fast with actionable errors when unsupported
  - [x] Implement unified stop-policy behavior (stop strings + stop token ids) consistent with HF backend semantics.
  - [x] Add explicit fallback policy wiring (optional `fallback_backend="hf"`).

- [x] Migrate call sites (behavior-preserving)
  - [x] Stage-B:
    - [x] Replace `src/stage_b/reflection/engine.py` generation block with engine usage.
    - [x] Centralize `src/stage_b/rollout.py` stop/EOS handling via engine stop options (keep sampling semantics unchanged).
  - [x] Stage-A:
    - [x] Replace model/processor loading with engine usage.
    - [x] Replace shared encode/generate/decode logic with engine batch helpers.
  - [x] vis_tools:
    - [x] Update `vis_tools/vis_qwen3.py` to use engine load + generate helpers.
    - [x] Implement custom `StoppingCriteria` / `LogitsProcessor` logic as optional engine plugins (off by default; enabled explicitly by flags/config).

- [x] Validation
  - [x] Run ruff auto-fix + pyright under conda env `ms` (ruff ok; pyright unavailable in env).
  - [x] Run unit tests (pytest suites for `tests/stage_a` and `tests/stage_b` pass under conda env `ms`).
  - [x] Run minimal smoke command set (help/arg-parse paths):
    - [x] `conda run -n ms python -m src.stage_a.cli --help`
    - [x] `conda run -n ms python vis_tools/vis_qwen3.py --help`
  - [x] Update runbooks and docs impacted by Stage-A/B inference changes:
    - [x] `docs/training/REFERENCE.md`
    - [x] `docs/runtime/STAGE_A_STAGE_B.md`
