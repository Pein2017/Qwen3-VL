#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model output/stage_1_full_aligner_only/best/eff_batch_32-lr_1e-4/checkpoint-200 \
    --adapters output/stage_2_llm_lora/v1-20251020-121949/lora_8_32-lr_5e-4-eff_batch-32/checkpoint-40 \
    --merge_lora true \
    --output_dir output/stage_2_merged/lora_8_32-lr_5e-4-eff_batch-32 \
    --safe_serialization true \
    --max_shard_size 5GB

echo "Merged model saved to: output/stage_2_merged/checkpoint-2"
echo ""
echo "Test inference with:"
echo "CUDA_VISIBLE_DEVICES=0 swift infer --model output/stage_2_merged/checkpoint-2 --stream true"

