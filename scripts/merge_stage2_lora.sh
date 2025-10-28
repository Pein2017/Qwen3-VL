#!/bin/bash
# Merge LoRA adapter into base model for end2end inference
base_model=output/stage_3_merged-10-27/v_2
adapters=output/10-28/stage_4/v0-20251028-013852/more_lora_last_4_8-weaker_aug-epoch_20-resumed/checkpoint-2484
output_dir=output/stage_4_merged-10-28/resumed-checkpoint-2484

CUDA_VISIBLE_DEVICES=2 \
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

