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
     - `vllm_max_model_len=8192`
     - start with `vllm_gpu_memory_utilization=0.6` and allow increasing once stable.
     - forbid vLLM LoRA (`vllm_enable_lora=false`) for multimodal DoRA stability.

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

## Impact
- Higher rollout throughput and better GPU utilization (6 rollout GPUs run continuously).
- Reduced VRAM contention on training GPUs, enabling more stable DoRA + multimodal training (ViT/aligner/LLM).
- Cleaner config semantics: trainer YAML stays the single source of truth, while server-only params are explicitly scoped.

## Out of Scope
- Multi-node rollout server deployment.
- Extending this migration to other GRPO configs (dense or 2048 summary presets).
- Guaranteeing bitwise-deterministic rollout sampling in server mode.
- Modifying upstream ms-swift behavior (unless explicitly required later).
