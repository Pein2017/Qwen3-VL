#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model output/stage_1_full_aligner_only/v3-20251021-022419/eff_batch_16-lr_5e-4-epoch_10/checkpoint-1100 \
    --adapters output/stage_3_vision_llm_loRA/v1-20251021-094218/lora_8_32-eff_batch-16/checkpoint-1240 \
    --merge_lora true \
    --output_dir output/stage_3_merged/lora_8_32-eff_batch-16 \
    --safe_serialization true \
    --max_shard_size 5GB

echo "Merged model saved to: output/stage_2_merged/checkpoint-2"
echo ""
echo "Test inference with:"
echo "CUDA_VISIBLE_DEVICES=0 swift infer --model output/stage_2_merged/checkpoint-2 --stream true"

