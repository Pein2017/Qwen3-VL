#!/bin/bash
# Merge LoRA adapter into base model for end2end inference
# Usage: gpus=0 bash scripts/merge_stage2_lora.sh
# Usage: gpus=0,1 bash scripts/merge_stage2_lora.sh

set -euo pipefail

# GPU configuration (unified API)
CUDA_VISIBLE_DEVICES="${gpus:-0}"

adapters=output/12-18/summary/v0-20251218-081823/epoch_4-res_1024-bbu_rru_fused-lrs_1e-4_mlp_1e-6_with_irrelevant_summary-aug_off/checkpoint-384

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open(\"$adapters/adapter_config.json\"))[\"base_model_name_or_path\"])")


output_dir=output/12-18/summary_merged/res_1024-with_irrelevant_summary

echo "Detected base model: $base_model"
echo "Adapters: $adapters"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
swift export \
    --model $base_model \
    --adapters $adapters \
    --merge_lora true \
    --output_dir $output_dir \
    --safe_serialization true \
    --max_shard_size 5GB

echo "Merged model saved to: $output_dir"
echo ""
echo "Test inference with:"
echo "gpus=${CUDA_VISIBLE_DEVICES} swift infer --model $output_dir --stream true"
