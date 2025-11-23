#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

adapters=output/11-23/stage_1.5/v0-20251122-171537/lvis_ratio_0.08-epoch_30-dlora-lrs_4_2_8-llm_kd_0.01-vision_kd_0.01-last_6_vision-all_llm-from_base/checkpoint-2250

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open('$adapters/adapter_config.json'))['base_model_name_or_path'])")


output_dir=output/11-23/stage_1.5_merged/lvis_ratio_0.08-last_6-checkpoint-2250

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

