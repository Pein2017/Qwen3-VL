#!/bin/bash
# Merge LoRA adapter into base model for end2end inference
base_model=output/10-28/stage_4_merged/checkpoint-5000

adapters=output/11-01/stage_3_gkd/v0-20251101-154916/gkd-last_12_4-epoch_50-eff_batch_32-ref_base_model-lan_kd_0.04-vision_kd_0.08-stronger_aug-from_stage_1/checkpoint-3450

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open('$adapters/adapter_config.json'))['base_model_name_or_path'])")


output_dir=output/stage_3_gkd-merged/11-01/checkpoint-3450

echo "Detected base model: $base_model"
echo "Adapters: $adapters"


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

