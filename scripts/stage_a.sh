#!/usr/bin/env bash
set -euo pipefail

# Stage-A inference convenience launcher
# Example: mission=挡风板安装检查 gpus=0 bash scripts/stage_a.sh
# Example: mission=BBU安装方式检查（正装） gpus=1 bash scripts/stage_a.sh
# Example: mission=挡风板安装检查 gpus=0,1,2,3 bash scripts/stage_a.sh  # multi-GPU
# Example: mission=BBU安装方式检查（正装） gpus=0 add_gt_fail_reason=true bash scripts/stage_a.sh  # with gt_fail_reason_text

# Fixed configuration
# CHECKPOINT="output/11-30/summary_merged/epoch_10-lr_2e-4-bs_32-res_1024"
CHECKPOINT="output/1-1/grpo-merged/checkpoint-500"
BASE_INPUT_DIR="group_data/scene_2.0_order"
BASE_OUTPUT_DIR="output_post/stage_a_grpo-1-4"

# Environment variable overrides (lowercase)
# BBU接地线检查
# BBU线缆布放要求
# 挡风板安装检查
# BBU安装方式检查（正装）
# prompt_profile=summary_runtime (explicit profile override)
# output_dir=... (override output directory)
# checkpoint=... (override checkpoint path)

DATASET="${dataset:-}"
MISSION="${mission:-}"
if [[ -n "${checkpoint:-}" ]]; then
  CHECKPOINT="${checkpoint}"
fi


if [[ -n "${input_dir:-}" ]]; then
  INPUT_DIR="${input_dir}"
else
  INPUT_DIR="${BASE_INPUT_DIR}"
fi
# Normalize when input_dir mistakenly points to a mission subdir.
if [[ "${INPUT_DIR}" == */"${MISSION}" ]]; then
  echo "[WARN] input_dir points to mission subdir; using parent instead: ${INPUT_DIR}"
  INPUT_DIR="$(dirname "${INPUT_DIR}")"
fi

if [[ -n "${output_dir:-}" ]]; then
  OUTPUT_DIR="${output_dir}"
else
  OUTPUT_DIR="${BASE_OUTPUT_DIR}"
fi
PROMPT_PROFILE="${prompt_profile:-summary_runtime}"
CUDA_VISIBLE_DEVICES="${gpus:-0}"
verify_flag="${verify_inputs:-true}"
DEBUG_FLAG="${debug:-false}"
PASS_GROUP_NUMBER="${pass_group_number:-200}"
FAIL_GROUP_NUMBER="${fail_group_number:-50}"
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

# Fixed parameters (with env overrides)
BATCH_SIZE_PER_RANK="${batch_size_per_rank:-32}"
MAX_PIXELS="${max_pixels:-1048576}"
MAX_NEW_TOKENS="${max_new_tokens:-1024}"
TEMPERATURE="0.0001"
TOP_P="1.0"
REP_PENALTY="1.05"
LOG_LEVEL="INFO"
SHARDING_MODE="${sharding_mode:-per_group}"  # per_group | per_image
KEEP_INTERMEDIATE_OUTPUTS="${keep_intermediate_outputs:-false}"
POSTPROCESS="${postprocess:-false}"  # Optional: postprocess Stage-A JSONL
ADD_GT_FAIL_REASON="${add_gt_fail_reason:-false}"  # Optional: add gt_fail_reason_text from Excel
EXCEL_PATH="${excel_path:-output_post/BBU_scene_latest.xlsx}"  # Excel file path for gt_fail_reason_text

# RRU watermark/OCR tends to fail when images are downscaled too aggressively.
# Use a higher pixel budget by default for RRU (can be overridden via max_pixels),
# and reduce batch size to keep GPU memory roughly stable.
if [[ "${DATASET}" == "rru" ]]; then
  if [[ -z "${max_pixels:-}" ]]; then
    MAX_PIXELS="4194304"  # ~2048x2048
  fi
fi

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
echo "Dataset:      $DATASET"
echo "Profile:      $PROMPT_PROFILE"
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
    --dataset "$DATASET" \
    --prompt_profile "$PROMPT_PROFILE" \
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
    --dataset "$DATASET" \
    --prompt_profile "$PROMPT_PROFILE" \
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

# Optional postprocess cleanup (RRU/BBU-specific)
if [[ "${POSTPROCESS,,}" == "true" ]] || [[ "${POSTPROCESS}" == "1" ]]; then
  echo ""
  echo "=================================="
  echo "Running Stage-A Postprocess"
  echo "=================================="
  OUTPUT_FILE="${OUTPUT_DIR}/${MISSION}_stage_a.jsonl"
  if [[ ! -f "${OUTPUT_FILE}" ]]; then
    echo "[WARNING] Output file not found: ${OUTPUT_FILE}"
    echo "Skipping postprocess."
  else
    echo "Processing: ${OUTPUT_FILE}"
    conda run -n "${CONDA_ENV}" --no-capture-output python -u -m src.stage_a.postprocess \
      --input "${OUTPUT_FILE}" \
      --dataset "${DATASET}" \
      --inplace
  fi
  echo "=================================="
fi

# Optional: Add gt_fail_reason_text from Excel (BBU-specific only)
if [[ "${ADD_GT_FAIL_REASON,,}" == "true" ]] || [[ "${ADD_GT_FAIL_REASON}" == "1" ]]; then
  # Only process BBU dataset; RRU is not supported yet
  if [[ "${DATASET}" == "bbu" ]]; then
    echo ""
    echo "=================================="
    echo "Adding GT Fail Reason Text (BBU only)"
    echo "=================================="
    OUTPUT_FILE="${OUTPUT_DIR}/${MISSION}_stage_a.jsonl"
    if [[ ! -f "${OUTPUT_FILE}" ]]; then
      echo "[WARNING] Output file not found: ${OUTPUT_FILE}"
      echo "Skipping gt_fail_reason addition."
    else
      echo "Processing: ${OUTPUT_FILE}"
      echo "Excel file: ${EXCEL_PATH}"
      if [[ ! -f "${EXCEL_PATH}" ]]; then
        echo "[WARNING] Excel file not found: ${EXCEL_PATH}"
        echo "Skipping gt_fail_reason addition."
      else
        conda run -n "${CONDA_ENV}" --no-capture-output python -u \
          "${REPO_DIR}/scripts/add_gt_fail_reason_to_stage_a.py" \
          "${OUTPUT_FILE}" \
          --excel "${EXCEL_PATH}" \
          --inplace 2>&1
        
        ADD_REASON_STATUS=$?
        if [[ ${ADD_REASON_STATUS} -eq 0 ]]; then
          echo ""
          echo "[SUCCESS] GT fail reason text addition completed successfully"
          echo "Original file updated: ${OUTPUT_FILE}"
        else
          echo ""
          echo "[ERROR] GT fail reason text addition failed with status ${ADD_REASON_STATUS}"
          echo "Original file preserved: ${OUTPUT_FILE}"
          # Don't fail the entire script if addition fails
        fi
      fi
    fi
    echo "=================================="
  else
    echo ""
    echo "[INFO] gt_fail_reason_text addition is only supported for dataset=bbu"
    echo "[INFO] Current dataset=${DATASET}, skipping gt_fail_reason addition."
  fi
fi
