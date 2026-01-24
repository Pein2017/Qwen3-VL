# Design: Summary GRPO Server-Mode Rollout Separation (6 Rollout GPUs + 2 Train GPUs)

## Goals
- Use ms-swift **vLLM server mode** for GRPO rollouts to separate rollout inference from learner training GPUs.
- Support a single-node 8-GPU setup with a **6+2 split**:
  - Rollout server: 6 GPUs (TP=1, DP=6)
  - Learner: 2 GPUs (torchrun world_size=2)
- Keep configs **YAML-first** (same entrypoint `src/sft.py`) and preserve ms-swift integration.
- Prioritize **stability + efficiency** for multimodal DoRA training (ViT + aligner + LLM), avoiding vLLM LoRA sync.

## Non-Goals
- Multi-node rollout clusters.
- Bitwise-deterministic rollouts.
- Extending migration to dense GRPO configs (explicitly out of scope for this change).

## Background (ms-swift + vLLM server mode)
In ms-swift, server mode is activated when training args include `vllm_server_host`/`vllm_server_port` (or base_url).
The trainer connects via HTTP for `/infer/` and uses an NCCL communicator for weight sync.

Important implications:
- Some `rlhf.vllm_*` knobs (e.g. `vllm_max_model_len`) are **ignored** on the trainer side for external vLLM and must be set
  when launching `swift rollout`.
- Adapter-only weight sync depends on server-side vLLM LoRA support and is fragile for multimodal + ViT/aligner training.
  This design uses **full merged-weight sync** (server launched with `--vllm_enable_lora false`).

## Configuration Layout

### Trainer-side connectivity (ms-swift `RLHFArguments`)
These fields live under `rlhf.*` in the training YAML:
- `rlhf.use_vllm: true`
- `rlhf.vllm_mode: server`
- `rlhf.vllm_server_host: ["127.0.0.1"]`
- `rlhf.vllm_server_port: [8080]`
- `rlhf.vllm_server_timeout: 240`

Notes:
- Host/port are list-typed to match ms-swift args schema.
- We keep a single host/port in this change (single node).

### Rollout-server launch config (Qwen3‑VL-owned)
These fields live under `custom.extra.rollout_server` and are consumed by the new launcher:

```yaml
custom:
  extra:
    rollout_server:
      # 6 rollout GPUs = TP * DP
      vllm_tensor_parallel_size: 1
      vllm_data_parallel_size: 6

      # Start conservative; user can increase.
      vllm_gpu_memory_utilization: 0.6
      # In this repo, `global_max_length` represents total (input prompt + image tokens + output).
      # Keep rollout max length aligned to avoid truncation or server-side errors.
      vllm_max_model_len: 12000

      # Robust default for multimodal + DoRA (no vLLM LoRA).
      vllm_enable_lora: false

      # Optional passthroughs (kept minimal; launcher may support a curated subset):
      # vllm_enable_prefix_caching: true
      # vllm_disable_custom_all_reduce: false
      # vllm_engine_kwargs: {...}
```

Validation:
- Launcher MUST validate `rlhf.vllm_server_port` is a non-empty list (single-node v1 uses `len == 1`).
- Launcher MUST validate TP/DP are positive integers and TP*DP == number of visible server GPUs.
- Launcher MUST validate `vllm_max_model_len` is a positive int.
- When `global_max_length` is present in the training YAML, launcher MUST validate:
  - `custom.extra.rollout_server.vllm_max_model_len >= global_max_length`
  - (recommended) migrated configs set them equal since both represent total input+output budget.
- Launcher MUST forbid `vllm_enable_lora: true` in server mode for this workflow (stability).

## New Scripts / Entrypoints

### Unified launcher (server + learner, single entrypoint)
Responsibilities:
- Resolve YAML with `extends` (using the same config merge logic as `ConfigLoader.load_yaml_with_extends`).
- Extract `model.model` path, `rlhf` server connectivity, and `custom.extra.rollout_server`.
- Fail fast if the configured rollout port (`rlhf.vllm_server_port[0]`; 8080 by default for this workflow) is already in use.
- Launch `swift rollout` on dedicated rollout GPUs in the background and redirect server logs to a file.
- Poll `/health/` until `wait_timeout` before starting training (fail fast on timeout).
- Launch learner training via `scripts/train.sh` on dedicated training GPUs.
- Record the rollout server PID/PGID and terminate the server automatically when the launcher exits (normal exit, error, or SIGINT).

CLI interface:
- `bash scripts/grpo_server_mode.sh config=<path> server_gpus=0,1,2,3,4,5 train_gpus=6,7 [wait_timeout=...] [CONDA_ENV=ms]`

### Deprecation strategy for `scripts/grpo_server_train.sh`
The current repo has a combined launcher (`scripts/grpo_server_train.sh`) that:
- parses `rlhf.*` and starts `swift rollout` + training in one process wrapper.

This change replaces it with a single unified launcher and removes the combined launcher to avoid ambiguous operator paths.

## Stability Notes (DoRA + ViT/aligner/LLM training)
- Server mode isolates rollout VRAM from training VRAM, helping when training ViT and aligner.
- We keep vLLM LoRA disabled and rely on full merged-weight sync to avoid known multimodal instability.
- Start with conservative `vllm_gpu_memory_utilization=0.6` and increase if stable.

## Known Constraints / Risks
- ms-swift rollout deployment may auto-select a free port if the requested port is occupied; we mitigate this by
  checking port availability before launch and failing fast.
- The NCCL group port used for weight sync is not explicitly configured in this change; running multiple
  concurrent jobs on the same node may still cause collisions. For now, treat this workflow as
  **one server-mode GRPO job per node** unless explicit port isolation is added later.
