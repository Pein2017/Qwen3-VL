# generation-engine Specification

## Purpose
TBD - created by archiving change 2026-01-06-centralize-generation-engine. Update Purpose after archive.
## Requirements
### Requirement: A centralized generation engine provides batch-first APIs for text and VLM
The repository SHALL expose a centralized engine under `src/generation/` that provides batch-first generation APIs:
- text-only batch generation
- vision-language (text+image) batch generation

The engine SHALL return decoded text only; structured parsing SHALL remain outside the engine.

Each generation result SHALL expose two decoded text slots:
- `text`: cleaned decoded assistant text (primary consumer path)
- `raw_text`: decoded text including special tokens (debugging and optional plugin consumers)

#### Scenario: Text-only batch request returns decoded strings
- **GIVEN** a batch of chat message requests without images
- **WHEN** the generation engine runs text-only batch generation
- **THEN** it returns one decoded assistant text result per request
- **AND** each result includes both `text` and `raw_text`
- **AND** no task-specific output parsing is performed inside the engine

#### Scenario: VLM batch request returns decoded strings
- **GIVEN** a batch of chat message requests each containing exactly one image
- **WHEN** the generation engine runs VLM batch generation
- **THEN** it returns one decoded assistant text result per request
- **AND** each result includes both `text` and `raw_text`
- **AND** no dense JSON extraction or normalization is performed inside the engine

### Requirement: The engine supports HF and vLLM colocate backends with explicit selection
The engine SHALL support the following backends:
- Hugging Face backend (transformers `generate`) as the correctness-first default
- vLLM backend in colocate mode as an optional faster backend

Backend selection SHALL be explicit (e.g., by config or CLI flag), and SHALL NOT silently change at runtime.

#### Scenario: HF backend is explicitly selected
- **GIVEN** engine configuration selecting `backend="hf"`
- **WHEN** batch generation is executed
- **THEN** the HF backend is used for loading and generation

#### Scenario: vLLM backend is explicitly selected
- **GIVEN** engine configuration selecting `backend="vllm"`
- **WHEN** batch generation is executed
- **THEN** the vLLM backend is used for loading and generation

### Requirement: Stop-policy handling is centralized and consistent across call sites and backends
The engine SHALL centralize stop-policy handling used by downstream workflows (e.g., Stage-B rollout sampling),
supporting:
- stop strings (e.g., common chat terminators)
- optional stop token ids

When stop strings are provided, the engine SHALL apply deterministic post-truncation on decoded output text to ensure
consistent behavior across backends.

#### Scenario: Stage-B-style stop strings truncate the decoded output deterministically
- **GIVEN** a text-only batch request with stop strings configured
- **WHEN** batch generation is executed via the HF backend
- **THEN** each decoded `text` result is truncated at the earliest stop string occurrence (if present)
- **AND** no task-specific parsing is performed inside the engine

### Requirement: vLLM multimodal support is best-effort with fail-fast errors and optional explicit fallback
Because vLLM multimodal support may depend on vLLM version and model architecture, the vLLM backend SHALL treat VLM
generation as a best-effort capability with explicit capability reporting and deterministic failure behavior.

Concretely, the vLLM backend:
- SHALL expose a capability flag for VLM support
- SHALL fail fast with a clear error when VLM generation is requested but unsupported
- MAY support an explicit fallback backend configuration (e.g., `fallback_backend="hf"`) for unsupported requests

Silent fallback is forbidden.

#### Scenario: vLLM VLM request fails fast when unsupported
- **GIVEN** engine configuration selecting `backend="vllm"` with no fallback configured
- **AND** the vLLM backend reports `supports_vlm == false`
- **WHEN** VLM batch generation is requested
- **THEN** the engine raises a clear error indicating that vLLM multimodal generation is unsupported

#### Scenario: vLLM VLM request uses explicit fallback when configured
- **GIVEN** engine configuration selecting `backend="vllm"` and `fallback_backend="hf"`
- **AND** the vLLM backend reports `supports_vlm == false`
- **WHEN** VLM batch generation is requested
- **THEN** the engine executes the request via the HF backend
- **AND** the engine records (via logs or structured metadata) that fallback occurred

### Requirement: Chat-template rendering is centralized and consistent across call sites
The engine SHALL centralize chat-template rendering and SHALL support:
- tokenizer-owned `apply_chat_template(...)` paths for text-only models
- processor-owned `apply_chat_template(...)` paths for VLM models

The engine SHALL disable model “thinking” blocks (e.g., `enable_thinking=False`) for deterministic and parseable outputs across **all** call sites (Stage-A, Stage-B, and `vis_tools`).
When the parameter is unsupported, the engine SHALL use a compatibility fallback without changing message semantics.

#### Scenario: enable_thinking fallback does not crash
- **GIVEN** a tokenizer whose `apply_chat_template` does not accept `enable_thinking`
- **WHEN** the engine renders a prompt with thinking disabled
- **THEN** rendering succeeds via a compatibility fallback

### Requirement: VLM processor/tokenizer normalization is centralized (padding + pixel budget)
The engine SHALL centralize common processor/tokenizer normalization required for stable batching and reproducible VLM inference, including:
- left-padding for decoder-only batching (`padding_side="left"`)
- pad-token fallback when missing (e.g., `pad_token := eos_token` when safe)
- VLM pixel budget controls (e.g., `max_pixels`, and optionally `min_pixels` / `do_resize`) that preserve Stage-A runtime behavior

#### Scenario: Stage-A VLM pixel budget is preserved through engine loading
- **GIVEN** Stage-A configuration specifying a VLM pixel budget (e.g., `max_pixels=786432`)
- **WHEN** the engine loads the processor/model for VLM batch generation
- **THEN** the processor is configured such that the same pixel budget is applied during encoding
- **AND** downstream VLM generation behavior does not regress due to silent preprocessing drift

### Requirement: Optional plugins enable reusable generation-time customization without coupling to Stage-A/B defaults
The engine SHALL support an optional plugin mechanism for reusable generation-time behavior, including:
- HF-only generation hooks (e.g., `StoppingCriteria` and `LogitsProcessor` injection)
- deterministic post-decode text cleanup/trimming (non-parsing)

Plugins MUST be explicitly enabled by the caller. When plugins are absent, baseline engine behavior SHALL apply.

Plugins SHALL NOT perform task-specific parsing or semantic interpretation of model outputs.

The engine SHALL validate plugin compatibility before starting generation and SHALL fail fast with a clear error when:
- a plugin requires backend-specific hooks that are unsupported by the selected backend, or
- a plugin requires a request type or capability that is unsupported for the current request (e.g., VLM-only plugin on a text request).

#### Scenario: vis_tools enables a custom stopping plugin explicitly
- **GIVEN** a VLM batch request executed via the HF backend
- **AND** a plugin is explicitly provided to stop at a balanced JSON boundary
- **WHEN** batch generation is executed
- **THEN** generation uses the plugin-provided stopping behavior
- **AND** the engine still returns only decoded text slots (`text` and `raw_text`)

#### Scenario: Plugin incompatible with backend is rejected before generation begins
- **GIVEN** engine configuration selecting `backend="vllm"`
- **AND** a plugin is configured that requires HF-only hooks (e.g., `StoppingCriteria`)
- **WHEN** batch generation is requested with that plugin enabled
- **THEN** the engine raises a clear error before starting generation

### Requirement: Model loading is centralized and normalizes common inference knobs
The engine SHALL centralize model loading for both backends and SHALL normalize:
- dtype selection
- device placement and device selection
- attention implementation selection (when applicable)
- pad-token and padding-side defaults (when applicable)

#### Scenario: Pad token fallback is applied consistently
- **GIVEN** a tokenizer without an explicit pad token
- **WHEN** the engine loads the tokenizer for batch generation
- **THEN** a consistent pad-token fallback strategy is applied

