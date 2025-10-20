#!/usr/bin/env bash
# Pure config-driven training script
# All hyperparameters are in YAML config files
# This script only handles runtime settings

set -euo pipefail

# Resolve repository root from this script's location and set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================================
# Runtime Settings (NOT training hyperparameters)
# ============================================================================

CONDA_ENV="${CONDA_ENV:-ms}"
CONFIG_RAW="${config:-debug}"
DEBUG="${DEBUG:-false}"

# GPU configuration
CUDA_VISIBLE_DEVICES="${gpus:-0}"

# Derive number of GPUs
IFS=',' read -r -a gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#gpu_array[@]}"

# ============================================================================
# Build Command
# ============================================================================

## Resolve CONFIG_RAW to a full path under repo configs/ with .yaml extension if needed
if [[ "${CONFIG_RAW}" != *.yaml ]]; then
  CONFIG_REL="configs/${CONFIG_RAW}.yaml"
else
  CONFIG_REL="${CONFIG_RAW}"
fi
CONFIG_PATH="${REPO_DIR}/${CONFIG_REL}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  # Choose a random free MASTER_PORT if not provided
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)"
  fi
  CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} -m src.sft --config ${CONFIG_PATH}"
else
  CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m src.sft --config ${CONFIG_PATH}"
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
echo "[INFO] Config file: ${CONFIG_PATH}"
echo "[INFO] GPUs: ${CUDA_VISIBLE_DEVICES} (num=${NUM_GPUS})"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
echo "[INFO] Master port: ${MASTER_PORT}"
fi
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Debug mode: ${DEBUG}"
echo "========================================================================"
echo ""
echo "[RUN] (cwd=${REPO_DIR}) ${CMD}"
echo ""

# Initialize conda and run the command from repo root
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
eval "${CMD}"

