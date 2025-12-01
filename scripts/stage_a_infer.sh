#!/usr/bin/env bash
set -euo pipefail

# Stage-A inference convenience launcher
# Example: mission=挡风板安装检查 gpu=0 bash scripts/stage_a_infer.sh
# Example: mission=BBU安装方式检查（正装） gpu=1 bash scripts/stage_a_infer.sh

# Fixed configuration
CHECKPOINT="output/11-30/summary_merged/epoch_10-lr_2e-4-bs_32-res_1024"
INPUT_DIR="group_data/bbu_scene_2.0_order"
OUTPUT_DIR="output_post/stage_a"

# Environment variable overrides (lowercase)
MISSION="${mission:-BBU线缆布放要求}"
gpu_id="${gpu:-7}"
no_mission_flag="${no_mission:-true}"
verify_flag="${verify_inputs:-true}"
if [[ "$gpu_id" == "cpu" ]]; then
  DEVICE="cpu"
  export CUDA_VISIBLE_DEVICES=""
else
  # When CUDA_VISIBLE_DEVICES is set, the specified GPU becomes cuda:0
  export CUDA_VISIBLE_DEVICES=$gpu_id
  DEVICE="cuda:0"
fi

# Fixed parameters
BATCH_SIZE="16"
MAX_PIXELS="1048576"
MAX_NEW_TOKENS="1024"
TEMPERATURE="0.0001"
TOP_P="1.0"
REP_PENALTY="1.05"
LOG_LEVEL="INFO"

# Print configuration
echo "=================================="
echo "Stage-A Inference Launcher"
echo "=================================="
echo "Mission:      $MISSION"
echo "Checkpoint:   $CHECKPOINT"
echo "Input:        $INPUT_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Device:       $DEVICE"
echo "Batch size:   $BATCH_SIZE"
echo "Max pixels:   $MAX_PIXELS"
echo "Temperature:  $TEMPERATURE"
echo "=================================="

# Optional flags
EXTRA_FLAGS=""
case "${no_mission_flag,,}" in
  1|true|yes)
    EXTRA_FLAGS+=" --no_mission_focus"
    ;;
esac
case "${verify_flag,,}" in
  1|true|yes)
    EXTRA_FLAGS+=" --verify_inputs"
    ;;
esac

# Run inference
exec python -m src.stage_a.cli \
  --checkpoint "$CHECKPOINT" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mission "$MISSION" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --max_pixels "$MAX_PIXELS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --repetition_penalty "$REP_PENALTY" \
  --log_level "$LOG_LEVEL" \
  ${EXTRA_FLAGS}

