# GRPO Server-Mode NCCL Error (Tracking Draft)

Status: Unresolved. Logged for future resolution.

## Summary
In ms-swift GRPO server mode, training fails while initializing the vLLM weight‑sync communicator. The error occurs during NCCL all‑reduce in the client communicator after the rollout server is confirmed up.

## Observed Date
- 2025-12-30

## Context
- Launch path: `bash scripts/grpo_server_train.sh`
- Rollout server: `swift rollout` running on GPUs `0,1`
- Training: `python -m src.sft --config configs/grpo/summary_grpo_server.yaml`
- Mode: GRPO + external vLLM server (`rlhf.vllm_mode: server`)

## Stack Trace (condensed)
```
... src/config/loader.py -> ms-swift RLHFArguments.__post_init__ -> _init_rollout
... _init_external_vllm -> VLLMClient.init_communicator -> PyNcclCommunicator
... ncclAllReduce -> RuntimeError: NCCL error: <truncated>
```

## Current Notes
- A temporary edit was attempted in `/data/ms-swift/swift/trainers/rlhf_trainer/vllm_client.py`
  to change NCCL device selection; it was reverted and the error persisted.
- Rollout settings are sourced from YAML and the server is observed to start on GPUs 0,1.

## Hypotheses (Not Yet Verified)
- NCCL communicator cannot establish a group connection between trainer and rollout server
  (network interface or port reachability issue).
- Incorrect NCCL interface selection for the environment (multi‑NIC or containerized setup).

## Pending Follow‑ups
- Capture the full NCCL error string with `NCCL_DEBUG=INFO` enabled.
- Verify trainer → rollout host connectivity and group port availability.
- Confirm correct `vllm_server_host` value when training and rollout are on different nodes/containers.
