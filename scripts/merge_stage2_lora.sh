#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

adapters=output/11-12/stage_3_gkd/v0-20251112-083254/eff_batch_32-epoch_10-last_2_vision-last_1_llm-from_stage_2_checkpoint-400-small_lrs/checkpoint-690

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open('$adapters/adapter_config.json'))['base_model_name_or_path'])")


output_dir=output/11-10/stage_3_gkd_merged_small_lrs-good/checkpoint-690

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

