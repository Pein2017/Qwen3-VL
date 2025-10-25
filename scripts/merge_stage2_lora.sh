#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model output/stage_3_merged/data_aug_on-epoch_50 \
    --adapters output/summary/10-25/per_image_1/v0-20251025-084845/epoch_10-ratio_1.0-per_image_1/checkpoint-300 \
    --merge_lora true \
    --output_dir output/summary_merged/10-25-per_image_1 \
    --safe_serialization true \
    --max_shard_size 5GB

echo "Merged model saved to: output/stage_2_merged/checkpoint-2"
echo ""
echo "Test inference with:"
echo "CUDA_VISIBLE_DEVICES=0 swift infer --model output/stage_2_merged/checkpoint-2 --stream true"

