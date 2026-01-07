# generation-engine Specification (Change Proposal)

## Purpose
Provide a centralized generation/inference engine for evaluation and deployment workflows in this repository.

The engine consolidates model loading, chat-template rendering, and batch generation + decoding behind a small, stable API.

## Non-goals (this change)
- Output parsing (dense JSON extraction, summary JSON extraction, Stage‑B verdict parsing).
- Stage-A/B orchestration changes (group discovery, sharding, distributed merge logic).
- vLLM server mode.
- Re-implementing or overriding ms-swift’s training-time GRPO rollout engine.

## ADDED Requirements

### Requirement: A centralized generation engine provides batch-first APIs for text and VLM
The repository SHALL expose a centralized engine under `src/generation/` that provides batch-first generation APIs:
- text-only batch generation
- vision-language (text+image) batch generation

The engine SHALL return decoded text only; structured parsing SHALL remain outside the engine.

#### Scenario: Text-only batch request returns decoded strings
- **GIVEN** a batch of chat message requests without images
- **WHEN** the generation engine runs text-only batch generation
- **THEN** it returns one decoded assistant text result per request
- **AND** no task-specific output parsing is performed inside the engine

#### Scenario: VLM batch request returns decoded strings
- **GIVEN** a batch of chat message requests each containing exactly one image
- **WHEN** the generation engine runs VLM batch generation
- **THEN** it returns one decoded assistant text result per request
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

When supported, the engine SHOULD disable model “thinking” blocks (e.g., `enable_thinking=False`) for deterministic and parseable outputs in evaluation pipelines.
When the parameter is unsupported, the engine SHALL use a compatibility fallback without changing message semantics.

#### Scenario: enable_thinking fallback does not crash
- **GIVEN** a tokenizer whose `apply_chat_template` does not accept `enable_thinking`
- **WHEN** the engine renders a prompt with thinking disabled
- **THEN** rendering succeeds via a compatibility fallback

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
