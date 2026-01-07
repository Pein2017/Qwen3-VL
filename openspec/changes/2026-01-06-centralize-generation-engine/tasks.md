# Tasks: Centralize generation/inference engine (HF + vLLM colocate)

- [ ] Pre-implementation audit
  - [ ] Inventory current generation entrypoints and duplicated blocks:
    - [ ] `src/stage_a/inference.py` (load + encode + generate + decode)
    - [ ] `src/stage_b/runner.py` (load + tokenizer settings)
    - [ ] `src/stage_b/reflection/engine.py` (apply_chat_template + generate + decode)
    - [ ] `src/stage_b/rollout.py` (batched generate + EOS/stop handling)
    - [ ] `vis_tools/vis_qwen3.py` (load + apply_chat_template + generate + decode)
  - [ ] Confirm supported model variants:
    - [ ] VL Qwen3-VL checkpoints (Stage-A / vis_tools)
    - [ ] text-only checkpoints (Stage-B)
  - [ ] Confirm vLLM availability + multimodal support constraints in the target environment.
  - [ ] Confirm ms-swift GRPO rollout remains out-of-scope (no reimplementation).

- [ ] Engine scaffolding (`src/generation/`)
  - [ ] Add Schema-Constitution-compliant config + contract types:
    - [ ] `ModelLoadConfig`
    - [ ] `GenerationOptions`
    - [ ] request/response dataclasses for batch APIs
  - [ ] Add backend protocol + façade `GenerationEngine`.
  - [ ] Implement shared chat-template rendering helpers (tokenizer vs processor ownership).

- [ ] Implement HF backend (safe)
  - [ ] Text-only batch generation (encode via tokenizer, `model.generate`, decode).
  - [ ] VLM batch generation (encode via processor, `model.generate`, decode).
  - [ ] Return both cleaned decoded text and minimal token accounting (best-effort).

- [ ] Implement vLLM colocate backend (fast)
  - [ ] Text-only batch generation via vLLM engine.
  - [ ] VLM batch generation when supported:
    - [ ] Implement capability checks for model+vLLM version
    - [ ] Fail fast with actionable errors when unsupported
  - [ ] Add explicit fallback policy wiring (optional `fallback_backend="hf"`).

- [ ] Migrate call sites (behavior-preserving)
  - [ ] Stage-B:
    - [ ] Replace `src/stage_b/reflection/engine.py` generation block with engine usage.
    - [ ] Optionally centralize `src/stage_b/rollout.py` stop/EOS handling (keep sampling semantics unchanged).
  - [ ] Stage-A:
    - [ ] Replace model/processor loading with engine usage.
    - [ ] Replace shared encode/generate/decode logic with engine batch helpers.
  - [ ] vis_tools:
    - [ ] Update `vis_tools/vis_qwen3.py` to use engine load + generate helpers.
    - [ ] Keep custom `StoppingCriteria` / `LogitsProcessor` logic local.

- [ ] Validation
  - [ ] Run ruff auto-fix + pyright under conda env `ms`.
  - [ ] Run unit tests.
  - [ ] Run minimal smoke command set (help/arg-parse paths):
    - [ ] `conda run -n ms python -m src.stage_a.cli --help`
    - [ ] `conda run -n ms python vis_tools/vis_qwen3.py --help`

