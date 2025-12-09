#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

adapters=output/12-4/summary/v0-20251208-084239/res_1024-bbu_rru_fused-lrs_1e-4-dlora_16_32/checkpoint-690

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open(\"$adapters/adapter_config.json\"))[\"base_model_name_or_path\"])")


output_dir=output/12-9/summary_merged/checkpoint-690

echo "Detected base model: $base_model"
echo "Adapters: $adapters"


CUDA_VISIBLE_DEVICES=0 \
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
echo "CUDA_VISIBLE_DEVICES=0 swift infer --model $output_dir --stream true"
