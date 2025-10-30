#!/bin/bash
# Merge LoRA adapter into base model for end2end inference
base_model=output/10-29/stage_1-gkd/v3-20251029-135454/gkd-eff_batch_32-epoch_10/checkpoint-520

adapters=output/10-30/stage_3_gkd/v1-20251030-031808/gkd-last_4_2-epoch_20-eff_batch_32-ref_base_model-lan_kd_0.04-vision_kd_0.15-weaker_color_aug/checkpoint-400

output_dir=output/stage_3_gkd-merged/10-29/lan_kd_0.04-vision_kd_0.15-weaker_color_aug-checkpoint-400

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

