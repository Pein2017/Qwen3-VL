#!/usr/bin/env bash
set -euo pipefail

# Stage-A inference convenience launcher
# Example: mission=挡风板安装检查 gpus=0 bash scripts/stage_a.sh
# Example: mission=BBU安装方式检查（正装） gpus=1 bash scripts/stage_a.sh
# Example: mission=挡风板安装检查 gpus=0,1,2,3 bash scripts/stage_a.sh  # multi-GPU

# Fixed configuration
# CHECKPOINT="output/11-30/summary_merged/epoch_10-lr_2e-4-bs_32-res_1024"
CHECKPOINT="output/12-18/summary_merged/res_1024-with_irrelevant_summary"
INPUT_DIR="group_data/bbu_scene_2.0_order"
OUTPUT_DIR="output_post/stage_a"

# Environment variable overrides (lowercase)
# BBU接地线检查
# BBU线缆布放要求
# 挡风板安装检查
# BBU安装方式检查（正装）


MISSION="${mission:-挡风板安装检查}"
CUDA_VISIBLE_DEVICES="${gpus:-0}"
no_mission_flag="${no_mission:-true}"
verify_flag="${verify_inputs:-true}"
DEBUG_FLAG="${debug:-false}"
PASS_GROUP_NUMBER="${pass_group_number:-1200}"
FAIL_GROUP_NUMBER="${fail_group_number:-1000}"
SAMPLE_SEED="${sample_seed:-42}"

# Derive number of GPUs (ignore empty/whitespace tokens)
IFS=',' read -r -a _raw_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
gpu_array=()
for _dev in "${_raw_gpu_array[@]}"; do
  _trimmed="${_dev//[[:space:]]/}"
  [[ -n "${_trimmed}" ]] && gpu_array+=("${_trimmed}")
done
# Normalize CUDA_VISIBLE_DEVICES to a comma-separated list without whitespace.
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_array[*]}")"
NUM_GPUS="${#gpu_array[@]}"

if [[ "${CUDA_VISIBLE_DEVICES}" == "cpu" ]]; then
  DEVICE="cpu"
  export CUDA_VISIBLE_DEVICES=""
  NUM_GPUS=0
elif [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[ERROR] Invalid gpus list: '${CUDA_VISIBLE_DEVICES}'" >&2
  exit 1
else
  # When CUDA_VISIBLE_DEVICES is set, the specified GPU becomes cuda:0
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
  DEVICE="cuda:0"
fi

# Fixed parameters
BATCH_SIZE_PER_RANK="32"
MAX_PIXELS="1048576"
MAX_NEW_TOKENS="1024"
TEMPERATURE="0.001"
TOP_P="1.0"
REP_PENALTY="1.1"
LOG_LEVEL="INFO"
SHARDING_MODE="${sharding_mode:-per_group}"  # per_group | per_image
KEEP_INTERMEDIATE_OUTPUTS="${keep_intermediate_outputs:-false}"

# Resolve repository root from this script's location and set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Default conda environment
CONDA_ENV="${conda_env:-ms}"

# Print configuration
echo "=================================="
echo "Stage-A Inference Launcher"
echo "=================================="
echo "Mission:      $MISSION"
echo "Checkpoint:   $CHECKPOINT"
echo "Input:        $INPUT_DIR"
echo "Output:       $OUTPUT_DIR"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES} (num=${NUM_GPUS})"
echo "Device:       $DEVICE"
echo "Batch size:   $BATCH_SIZE_PER_RANK"
echo "Max pixels:   $MAX_PIXELS"
echo "Temperature:  $TEMPERATURE"
echo "Sharding:     $SHARDING_MODE"
echo "Conda env:    $CONDA_ENV"
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
case "${DEBUG_FLAG,,}" in
  1|true|yes)
    EXTRA_FLAGS+=" --debug"
    ;;
esac
case "${KEEP_INTERMEDIATE_OUTPUTS,,}" in
  1|true|yes)
    EXTRA_FLAGS+=" --keep_intermediate_outputs"
    ;;
esac

SAMPLING_ARGS=""
if [[ -n "${PASS_GROUP_NUMBER// }" ]]; then
  SAMPLING_ARGS+=" --pass_group_number ${PASS_GROUP_NUMBER}"
fi
if [[ -n "${FAIL_GROUP_NUMBER// }" ]]; then
  SAMPLING_ARGS+=" --fail_group_number ${FAIL_GROUP_NUMBER}"
fi
SAMPLING_ARGS+=" --sample_seed ${SAMPLE_SEED}"

# Run from repo root using conda run
cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Force unbuffered Python output so logging flushes progressively
export PYTHONUNBUFFERED=1

# Run inference with torchrun for multi-GPU, single-process for single-GPU
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  # Fail fast on errors; single node only.
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$((10000 + RANDOM % 55536))
  fi
  conda run -n "${CONDA_ENV}" --no-capture-output \
    torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" --max_restarts=0 \
    -m src.stage_a.cli \
    --checkpoint "$CHECKPOINT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mission "$MISSION" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE_PER_RANK" \
    --sharding_mode "$SHARDING_MODE" \
    --max_pixels "$MAX_PIXELS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REP_PENALTY" \
    --log_level "$LOG_LEVEL" \
    ${EXTRA_FLAGS} ${SAMPLING_ARGS}
else
  conda run -n "${CONDA_ENV}" --no-capture-output python -u -m src.stage_a.cli \
    --checkpoint "$CHECKPOINT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mission "$MISSION" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE_PER_RANK" \
    --sharding_mode "$SHARDING_MODE" \
    --max_pixels "$MAX_PIXELS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REP_PENALTY" \
    --log_level "$LOG_LEVEL" \
    ${EXTRA_FLAGS} ${SAMPLING_ARGS}
fi
STATUS=$?
if [[ ${STATUS} -ne 0 ]]; then
  exit ${STATUS}
fi
