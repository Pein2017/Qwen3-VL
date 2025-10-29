#!/bin/bash
# Merge LoRA adapter into base model for end2end inference
base_model=output/stage_2_merged-10-25

adapters=output/10-28/stage_3/v1-20251028-135626/last_4_2-epoch_100-aug_conservative-eff_batch_32/checkpoint-5175

output_dir=output/stage_3_merged/10-29

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

