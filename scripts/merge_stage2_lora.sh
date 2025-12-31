#!/bin/bash
# Merge LoRA adapter into base model for end2end inference
# Usage: gpus=0 bash scripts/merge_stage2_lora.sh
# Usage: gpus=0,1 bash scripts/merge_stage2_lora.sh

set -euo pipefail

# GPU configuration (unified API)
CUDA_VISIBLE_DEVICES="${gpus:-0}"

adapters=output/12-30/new_schema-4B-summary-grpo/v0-20251231-032857/temperature_0.7-num_generations_4-vllm_colocate-eff_bs_24-gen_bs_24/checkpoint-1252

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open(\"$adapters/adapter_config.json\"))[\"base_model_name_or_path\"])")


output_dir=output/12-30/grpo-merged


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
