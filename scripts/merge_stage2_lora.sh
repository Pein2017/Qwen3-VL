#!/bin/bash
# Merge LoRA adapter into base model for end2end inference
base_model=output/10-25/stage_1_from_base/v2-20251025-140114/eff_batch_64-per_image_2-from_base/checkpoint-200
adapters=output/10-25/stage_2_llm_lora/v0-20251025-153140/lora_8_16-5e-4-eff_batch_32-last_4/checkpoint-260
output_dir=output/stage_2_merged-10-25

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

