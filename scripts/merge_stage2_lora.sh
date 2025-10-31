#!/bin/bash
# Merge LoRA adapter into base model for end2end inference
# The base model path is automatically extracted from adapter_config.json

adapters=output/10-30/stage_3_gkd/v5-20251030-145352/gkd-last_6_4-epoch_50-eff_batch_32-ref_base_model-lan_kd_0.04-vision_kd_0.3-weaker_color_aug/checkpoint-3000

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open('$adapters/adapter_config.json'))['base_model_name_or_path'])")

echo "Detected base model: $base_model"
echo "Adapters: $adapters"

output_dir=output/stage_3_gkd-merged/10-30/lan_kd_0.04-vision_kd_0.3-weaker_color_aug-checkpoint-3000

CUDA_VISIBLE_DEVICES=1 \
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

