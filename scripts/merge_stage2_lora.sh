#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

adapters=output/11-08/stage_3_gkd/v0-20251108-165251/eff_batch_32-epoch_150-all_vision-last_4_llm/checkpoint-7500

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open('$adapters/adapter_config.json'))['base_model_name_or_path'])")


output_dir=output/stage_3_gkd_merged/11-08/checkpoint-7500

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

