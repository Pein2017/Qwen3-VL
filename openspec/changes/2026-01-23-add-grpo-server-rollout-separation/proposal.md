# Proposal: Split Summary GRPO Rollout (vLLM Server) From Learner GPUs

## Why
Current summary-mode GRPO configs run vLLM in **colocate** mode (`rlhf.use_vllm=true`, `rlhf.vllm_mode=colocate`),
which forces rollout KV-cache and training activations to share the same GPUs. This creates:

- **VRAM contention**: configs compensate by lowering `vllm_gpu_memory_utilization` (e.g. 0.5), limiting rollout throughput.
- **Poor utilization**: rollout-heavy GRPO alternates between inference-heavy and backward-heavy phases on the same devices.
- **Instability pressure**: enabling vLLM LoRA sync is fragile for multimodal + DoRA + ViT/aligner training; full sync is safer but
  still wastes compute when colocated.

We want to move to ms-swift **server mode** rollouts so we can allocate:
- **6 GPUs for rollout** (`swift rollout` vLLM server)
- **2 GPUs for training** (`torchrun` learner)

This is a single-node (8-GPU) deployment and prioritizes **efficiency + stability** over strict determinism.

## What Changes
1) **Server-rollout separation for summary GRPO**
   - Migrate only:
     - `configs/train/grpo/summary_1024.yaml`
     - `configs/train/grpo/summary_1024_attr_key_recall.yaml`
   - Rollout server topology:
     - `TP=1`, `DP=6` (6 GPUs)
     - `vllm_max_model_len=12000` (matches `global_max_length`; total input+output token budget)
     - start with `vllm_gpu_memory_utilization=0.6` and allow increasing once stable.
     - forbid vLLM LoRA (`vllm_enable_lora=false`) for multimodal DoRA stability.
   - NOTE: Other summary GRPO presets (e.g., 2048 presets or existing server-mode presets) are **out of scope** for this
     change and are not migrated to the new `custom.extra.rollout_server` contract yet.

2) **Clean separation of trainer vs rollout-server settings**
   - Trainer connectivity stays in `rlhf.*` (ms-swift `RLHFArguments`):
     - `rlhf.vllm_mode=server`
     - `rlhf.vllm_server_host=[...]`
     - `rlhf.vllm_server_port=[8080]`
     - `rlhf.vllm_server_timeout=...`
     - bind/connect host is local-only (`127.0.0.1`).
   - Rollout server launch settings move under a Qwen3‑VL-owned namespace:
     - `custom.extra.rollout_server.*` (server-only vLLM knobs forwarded to `swift rollout`).
   - This avoids ms-swift warnings where some `rlhf.vllm_*` fields are ignored for external vLLM.

3) **New launch scripts**
   - Add a rollout-server launcher (server only): starts `swift rollout` on dedicated GPUs and fails fast if the configured
     port (8080 by default for this workflow) is not free.
   - Add a training launcher (learner only): runs `scripts/train.sh` on 2 GPUs and optionally health-checks the server.

4) **Deprecation of the existing combined launcher**
   - The repo already contains a combined launcher (`scripts/grpo_server_train.sh`) that starts both server and training.
   - This change replaces it with separate server-only + learner-only launchers, and removes the combined launcher
     to avoid ambiguous operator paths.

## Lifecycle Contract (Server-Mode Rollout)
This change intentionally removes the combined launcher and defines a clear operator contract for server-mode runs:

- **Start server first** (server-only):
  - Operator runs `bash scripts/grpo_rollout_server.sh config=<path> gpus=<rollout_gpus>` in a dedicated terminal (or tmux).
  - The server launcher resolves YAML `extends` and prints the resolved `host`, `port`, `TP`, `DP`, `vllm_max_model_len`,
    and `model.model` before starting.
  - The server launcher MUST fail fast if `rlhf.vllm_server_port[0]` is already in use to prevent accidentally connecting
    to a stale server.
  - The server binds to loopback only (`127.0.0.1` / `localhost`), not `0.0.0.0`.

- **Start learner second** (training-only):
  - Operator runs `bash scripts/grpo_train_server_mode.sh config=<path> gpus=<train_gpus>`.
  - By default, the learner launcher polls `http://<host>:<port>/health/` until a timeout, and fails fast if the server
    is not ready.

- **Stop behavior**:
  - The rollout server is an explicitly managed process. The learner launcher does not automatically stop the server
    when training exits.

- **Concurrency constraint (single node)**:
  - This workflow supports **one server-mode GRPO job per node at a time**. The ms-swift server-mode weight-sync uses an
    NCCL communicator with a default group port and is not explicitly isolated by this change; running multiple concurrent
    jobs on one node may lead to port collisions or hangs.

## Acceptance Criteria
- [ ] Migrated configs (`summary_1024*.yaml`) use server mode:
  - [ ] `rlhf.use_vllm=true` and `rlhf.vllm_mode=server`
  - [ ] `rlhf.vllm_server_host=["127.0.0.1"]` and `rlhf.vllm_server_port=[8080]` (single-node)
  - [ ] `global_max_length=12000`
  - [ ] `custom.extra.rollout_server.vllm_max_model_len=12000` (same input+output token budget)
- [ ] Separate launch scripts exist and are documented:
  - [ ] `scripts/grpo_rollout_server.sh` (server-only)
  - [ ] `scripts/grpo_train_server_mode.sh` (training-only)
  - [ ] `scripts/grpo_server_train.sh` is removed (expected) to avoid ambiguous operator paths.
- [ ] Server launcher validation is implemented:
  - [ ] host/port list validation + local-only enforcement
  - [ ] port availability check (fail fast)
  - [ ] `TP*DP == len(visible_server_gpus)` check
  - [ ] `vllm_max_model_len` is a positive int and `vllm_max_model_len >= global_max_length` when `global_max_length` is set
  - [ ] `vllm_enable_lora=true` is rejected in server mode for this workflow
- [ ] Minimal smoke run is possible:
  - [ ] server starts and `/health/` returns 200
  - [ ] learner run starts after health-check and connects successfully (no immediate timeouts)
- [ ] Docs updated per tasks (`scripts/README.md`, training GRPO runbook docs, and references in `technical_report.md`).
- [ ] `openspec validate 2026-01-23-add-grpo-server-rollout-separation --strict` passes.

## Rollout Plan (Single Node, 8 GPUs)
1. Choose a single 8-GPU node and select a **6+2 split**:
   - rollout server GPUs: `0,1,2,3,4,5`
   - learner GPUs: `6,7`
2. Start the rollout server:
   - `bash scripts/grpo_rollout_server.sh config=configs/train/grpo/summary_1024.yaml gpus=0,1,2,3,4,5`
   - Confirm `/health/` is ready and TP/DP/max_len match the config.
3. Start training:
   - `bash scripts/grpo_train_server_mode.sh config=configs/train/grpo/summary_1024.yaml gpus=6,7`
4. Observe early training stability (first N steps) and verify no repeated vLLM disconnects/timeouts.

## Rollback Plan
1. Stop the learner process.
2. Stop the rollout server process (Ctrl-C / kill from its terminal).
3. Revert the migrated configs back to colocate mode:
   - set `rlhf.vllm_mode=colocate`
   - remove (or ignore) `custom.extra.rollout_server`
4. Launch training via `scripts/train.sh` (colocate rollout).

## Test Plan
- **Validation tests** (fast, no GPUs):
  - Config merge/extends resolution works for the new launcher.
  - Missing required fields fail fast with clear, field-path errors.
  - `vllm_max_model_len < global_max_length` fails fast with a clear error.
  - Occupied-port detection fails fast and does not start the server.
- **Integration smoke** (single node):
  - Start server, confirm `/health/`.
  - Start learner and confirm it connects and runs initial steps without vLLM timeout spam.

## Monitoring / Operator Signals
- vLLM server:
  - `/health/` availability and response time
  - server log errors (OOM, engine init failures, weight-sync failures)
  - GPU memory and utilization on rollout GPUs
- Learner:
  - step time and any `vllm_server_timeout` failures
  - training loss/reward signals vs baseline colocate runs

## Impact
- Higher rollout throughput and better GPU utilization (6 rollout GPUs run continuously).
- Reduced VRAM contention on training GPUs, enabling more stable DoRA + multimodal training (ViT/aligner/LLM).
- Cleaner config semantics: trainer YAML stays the single source of truth, while server-only params are explicitly scoped.

## Out of Scope
- Multi-node rollout server deployment.
- Extending this migration to other GRPO configs (dense or 2048 summary presets).
- Guaranteeing bitwise-deterministic rollout sampling in server mode.
- Modifying upstream ms-swift behavior (unless explicitly required later).
- Running multiple concurrent server-mode GRPO jobs on the same node without explicit port isolation.
