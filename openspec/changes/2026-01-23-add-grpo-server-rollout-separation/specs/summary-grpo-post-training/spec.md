# summary-grpo-post-training (Delta)

## ADDED Requirements

### Requirement: Summary GRPO launch supports server-mode rollout separation
Summary GRPO post-training SHALL support **server-mode vLLM rollout separation** on a single node by splitting:
- rollout generation: a separately launched `swift rollout` server on dedicated GPUs
- learner training: the standard trainer entrypoint (`scripts/train.sh` → `src/sft.py`) on its own GPUs

When server mode is used:
- The trainer connectivity configuration SHALL remain under the `rlhf` block (ms-swift `RLHFArguments`).
- Rollout-server launch parameters (server-only vLLM knobs) SHALL be specified under `custom.extra.rollout_server`
  and consumed by a server-mode launcher script.

#### Scenario: Server-mode launch splits rollout and learner GPUs
- **GIVEN** a summary GRPO config with `rlhf.vllm_mode = "server"` and `custom.extra.rollout_server` populated
- **WHEN** launching the rollout server on 6 GPUs and the learner on 2 GPUs
- **THEN** the rollout server uses only the 6 rollout GPUs and the learner uses only the 2 training GPUs
- **AND** the learner connects to the rollout server using the `rlhf.vllm_server_host`/`rlhf.vllm_server_port` settings

#### Scenario: Rollout-server config is the source of truth for vLLM max length
- **GIVEN** server mode is used
- **WHEN** the rollout server is launched
- **THEN** `vllm_max_model_len` is taken from `custom.extra.rollout_server.vllm_max_model_len`
- **AND** the system does not rely on trainer-side `rlhf.vllm_max_model_len` for external vLLM configuration

### Requirement: Server-mode launcher validates config and fails fast
The server-mode launcher SHALL:
- Resolve training YAML configs with `extends` using the same merge rules as the training config loader.
- Require `custom.extra.rollout_server` to be present when `rlhf.vllm_mode == "server"`.
- Validate `rlhf.vllm_server_host` is a non-empty list and `rlhf.vllm_server_port` is a non-empty list.
- For single-node v1 operation, the launcher SHALL require `len(rlhf.vllm_server_host) == 1` and `len(rlhf.vllm_server_port) == 1`.
- For single-node v1 operation, the rollout server SHALL be local-only:
  - `rlhf.vllm_server_host[0]` SHALL be one of `127.0.0.1` or `localhost`
  - the launcher SHALL bind `swift rollout --host` to `rlhf.vllm_server_host[0]` (not `0.0.0.0`)
- Fail fast if `rlhf.vllm_server_port[0]` is already bound by another process.
- Validate server-only vLLM knobs:
  - `vllm_tensor_parallel_size` and `vllm_data_parallel_size` are positive integers
  - `vllm_tensor_parallel_size * vllm_data_parallel_size` equals the number of visible rollout GPUs
- Validate server-side token budget:
  - `custom.extra.rollout_server.vllm_max_model_len` SHALL be a positive integer
  - When `global_max_length` is provided in the training YAML, launcher SHALL validate:
    - `custom.extra.rollout_server.vllm_max_model_len >= global_max_length`
  - For this workflow, configs SHOULD set `custom.extra.rollout_server.vllm_max_model_len == global_max_length`
    since both represent total (input prompt + image tokens + output) budget.
- For multimodal DoRA training stability, server-mode rollout SHALL forbid vLLM LoRA:
  - `custom.extra.rollout_server.vllm_enable_lora` SHALL be false when provided
  - If `custom.extra.rollout_server.vllm_enable_lora` is true, the launcher SHALL fail fast with a clear error message

When `rlhf.vllm_mode == "colocate"`, `custom.extra.rollout_server` SHALL be treated as optional and ignored by the rollout-server launcher.

#### Scenario: Occupied port fails fast
- **GIVEN** the configured rollout port (`rlhf.vllm_server_port[0]`) is already bound by another process
- **WHEN** launching the rollout server
- **THEN** the launcher fails fast with a clear error message and does not start the server

#### Scenario: Missing rollout_server config fails fast in server mode
- **GIVEN** a config with `rlhf.vllm_mode = "server"`
- **AND** `custom.extra.rollout_server` is missing
- **WHEN** launching the rollout server
- **THEN** the launcher fails fast with a clear configuration error message

#### Scenario: Multiple server ports fail fast in single-node mode
- **GIVEN** a config with `rlhf.vllm_mode = "server"`
- **AND** `rlhf.vllm_server_port` contains more than one entry
- **WHEN** launching the rollout server
- **THEN** the launcher fails fast with a clear configuration error message

#### Scenario: Non-local host is rejected for local-only server mode
- **GIVEN** a config with `rlhf.vllm_mode = "server"`
- **AND** `rlhf.vllm_server_host = ["0.0.0.0"]`
- **WHEN** launching the rollout server
- **THEN** the launcher fails fast with a clear configuration error message

#### Scenario: vLLM LoRA is forbidden in server mode
- **GIVEN** a config with `rlhf.vllm_mode = "server"`
- **AND** `custom.extra.rollout_server.vllm_enable_lora = true`
- **WHEN** launching the rollout server
- **THEN** the launcher fails fast with a clear configuration error message

#### Scenario: vLLM max model length below global_max_length fails fast
- **GIVEN** a config with `rlhf.vllm_mode = "server"`
- **AND** `global_max_length = 12000`
- **AND** `custom.extra.rollout_server.vllm_max_model_len = 8192`
- **WHEN** launching the rollout server
- **THEN** the launcher fails fast with a clear configuration error message
- **AND** the error message indicates that `vllm_max_model_len` must be at least `global_max_length`

### Requirement: Server-mode launcher waits for rollout server health by default
The server-mode launcher SHALL:
- Use the `rlhf.vllm_server_host` / `rlhf.vllm_server_port` settings to construct the rollout server base URL.
- Poll `/health/` until `wait_timeout` (default: 120s) before starting training.
- Fail fast with a clear error if the rollout server is not healthy by `wait_timeout`.

#### Scenario: Learner health-check timeout fails fast
- **GIVEN** a config with `rlhf.vllm_mode = "server"`
- **AND** the rollout server is not running
- **WHEN** launching the server-mode launcher
- **THEN** it fails fast after `wait_timeout` with a clear error message

### Requirement: Server-mode rollout uses a unified launcher
Server-mode rollout separation SHALL be operated via a unified launcher script that:
- starts `swift rollout` on dedicated GPUs in the background
- runs `scripts/train.sh` on dedicated GPUs in the foreground (so the console shows learner logs)
- terminates the rollout server automatically when the launcher exits (normal exit, error, or SIGINT)

This change deprecates the combined launcher `scripts/grpo_server_train.sh` in favor of the unified launcher.

#### Scenario: Operator uses a single launcher and sees only learner logs
- **GIVEN** a summary GRPO config with `rlhf.vllm_mode = "server"`
- **WHEN** the operator starts the unified launcher
- **THEN** the rollout server and learner run on their respective GPU sets
- **AND** the console output is primarily learner logs (server logs are redirected to a file)

## MODIFIED Requirements

### Requirement: GRPO launch uses shared modules and `rlhf` block
Summary GRPO SHALL be launched via `scripts/train.sh` (shared training entrypoint) and SHALL configure GRPO under the
`rlhf` block (including `rlhf_type=grpo`, reward functions/weights, and rollout connectivity settings).

For vLLM integration:
- In **colocate** mode, all vLLM rollout configuration SHALL remain under `rlhf.*`.
- In **server** mode, trainer connectivity SHALL remain under `rlhf.*`, and server-only rollout launch parameters
  (e.g., `vllm_max_model_len`, `vllm_gpu_memory_utilization`, `vllm_tensor_parallel_size`, `vllm_data_parallel_size`)
  SHALL be specified under `custom.extra.rollout_server` and consumed by the rollout-server launcher.

#### Scenario: Colocate mode uses only rlhf for rollout configuration
- **GIVEN** a summary GRPO config with `rlhf.vllm_mode = "colocate"`
- **WHEN** training is launched
- **THEN** `scripts/train.sh` invokes the standard training entrypoint with `rlhf` settings
- **AND** no `custom.extra.rollout_server` configuration is required

#### Scenario: Server mode uses rlhf connectivity and rollout_server server-only knobs
- **GIVEN** a summary GRPO config with `rlhf.vllm_mode = "server"`
- **WHEN** training is launched with a separately launched rollout server
- **THEN** the learner connects using `rlhf.vllm_server_host`/`rlhf.vllm_server_port`
- **AND** the rollout server is configured using `custom.extra.rollout_server` for server-only vLLM knobs
