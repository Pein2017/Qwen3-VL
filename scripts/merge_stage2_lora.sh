#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

adapters=output/12-4/res_1024_fusion/v0-20251204-120915/epoch_12-lrs_5_8_10/checkpoint-1116

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open(\"$adapters/adapter_config.json\"))[\"base_model_name_or_path\"])")


output_dir=output/12-4/res_1024_fusion_merged/checkpoint-1116

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
