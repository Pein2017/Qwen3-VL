#!/bin/bash
# Merge LoRA adapter into base model for end2end inference

adapters=output/11-27/res_1024/v6-20251130-153929/epoch_30-sft_base-LRs_2_1_6-resume_from_sft_base-last_6_llm-with_lm_head/checkpoint-2070

# Extract base model path from adapter_config.json
base_model=$(python3 -c "import json; print(json.load(open('$adapters/adapter_config.json'))['base_model_name_or_path'])")


output_dir=output/11-27/res_1024_resumed_from_last_6-merged

echo "Detected base model: $base_model"
echo "Adapters: $adapters"


CUDA_VISIBLE_DEVICES=3 \
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

