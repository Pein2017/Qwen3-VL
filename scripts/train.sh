#!/usr/bin/env bash
# Pure config-driven training script
# All hyperparameters are in YAML config files
# This script only handles runtime settings

set -euo pipefail

# ============================================================================
# Runtime Settings (NOT training hyperparameters)
# ============================================================================

CONDA_ENV="${CONDA_ENV:-ms}"
CONFIG="${config:-configs/debug.yaml}"
BASE_CONFIG="${BASE_CONFIG:-}"  # Optional base config for inheritance
DEBUG="${DEBUG:-false}"

# GPU configuration
CUDA_VISIBLE_DEVICES="${gpus:-0}"

# Derive number of GPUs
IFS=',' read -r -a gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#gpu_array[@]}"

# ============================================================================
# Build Command
# ============================================================================

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${NUM_GPUS} -m src.sft --config ${CONFIG}"
else
  CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m src.sft --config ${CONFIG}"
fi

if [[ -n "${BASE_CONFIG}" ]]; then
  CMD+=" --base_config ${BASE_CONFIG}"
fi

if [[ "${DEBUG}" == "true" ]]; then
  CMD+=" --debug"
fi

# ============================================================================
# Display Info & Execute
# ============================================================================

echo "========================================================================"
echo "  MS-Swift Training with YAML Configuration"
echo "========================================================================"
echo "[INFO] Config file: ${CONFIG}"
if [[ -n "${BASE_CONFIG}" ]]; then
  echo "[INFO] Base config: ${BASE_CONFIG}"
fi
echo "[INFO] GPUs: ${CUDA_VISIBLE_DEVICES} (num=${NUM_GPUS})"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Debug mode: ${DEBUG}"
echo "========================================================================"
echo ""
echo "[RUN] ${CMD}"
echo ""

# Initialize conda and run the command
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
eval "${CMD}"

