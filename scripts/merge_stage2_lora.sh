#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

adapters=output/12-1/fusion_dlora/v1-20251201-151258/epoch_50-lrs_2_1_6-all_linear_plus_lm_head-bs_32/checkpoint-700

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open(\"$adapters/adapter_config.json\"))[\"base_model_name_or_path\"])")


output_dir=output/12-1/fusion_dlora_merged/lm_head/checkpoint-700

echo "Detected base model: $base_model"
echo "Adapters: $adapters"


CUDA_VISIBLE_DEVICES=3 \
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
