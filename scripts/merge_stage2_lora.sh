#!/bin/bash
# Merge LoRA adapter into base model for end2end inference
base_model=output/10-28/stage_4_merged/checkpoint-5000

adapters=output/summary/10-30/summary/v0-20251030-141905/last_llm_4-aug_conservative-eff_batch_32-epoch_5/checkpoint-300

output_dir=output/10-30/summary_merged

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

