#!/usr/bin/env bash
# Pure config-driven training script
# All hyperparameters are in YAML config files
# This script only handles runtime settings

set -euo pipefail

# CUDA / NCCL runtime defaults (can be overridden by caller)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-67108864}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Resolve repository root from this script's location and set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================================
# Runtime Settings (NOT training hyperparameters)
# ============================================================================

CONDA_ENV="${CONDA_ENV:-ms}"
CONFIG_RAW="${config:-debug}"
DEBUG="${debug:-false}"

# GPU configuration
CUDA_VISIBLE_DEVICES="${gpus:-0}"

# Derive number of GPUs (ignore empty/whitespace tokens)
IFS=',' read -r -a _raw_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
gpu_array=()
for _dev in "${_raw_gpu_array[@]}"; do
  [[ -n "${_dev// }" ]] && gpu_array+=("${_dev}")
done
NUM_GPUS="${#gpu_array[@]}"

# ============================================================================
# Build Command
# ============================================================================

## Resolve CONFIG_RAW to absolute path or repo-relative
if [[ "${CONFIG_RAW}" = /* ]]; then
  CONFIG_PATH="${CONFIG_RAW}"
elif [[ "${CONFIG_RAW}" == *.yaml ]]; then
  CONFIG_PATH="${REPO_DIR}/${CONFIG_RAW}"
else
  CONFIG_PATH="${REPO_DIR}/configs/${CONFIG_RAW}.yaml"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  # Generate random port in range [10000, 65535] if not already set
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$((10000 + RANDOM % 55536))
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
# Check if conda environment is already active
CURRENT_ENV="${CONDA_DEFAULT_ENV:-}"
if [[ "${CURRENT_ENV}" == "${CONDA_ENV}" ]]; then
    echo "[INFO] Conda environment '${CONDA_ENV}' is already active, skipping activation."
else
    # Try to find conda initialization script in common locations
    CONDA_INIT_SCRIPT=""
    if [ -f "${CONDA_BASE:-}/etc/profile.d/conda.sh" ]; then
        CONDA_INIT_SCRIPT="${CONDA_BASE}/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_INIT_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_INIT_SCRIPT="/root/miniconda3/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        # If conda is in PATH, try to get base path
        CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            CONDA_INIT_SCRIPT="$CONDA_BASE/etc/profile.d/conda.sh"
        fi
    fi

    if [ -n "$CONDA_INIT_SCRIPT" ]; then
        # Temporarily disable 'set -u' to avoid errors from conda deactivation scripts
        set +u
        source "$CONDA_INIT_SCRIPT" 2>/dev/null || true
        conda activate "${CONDA_ENV}" 2>/dev/null || {
            echo "[WARNING] Failed to activate conda environment '${CONDA_ENV}'. Continuing anyway." >&2
        }
        set -u
    else
        echo "[WARNING] Could not find conda initialization script. Assuming conda is already initialized." >&2
    fi
fi

cd "${REPO_DIR}"
eval "${CMD}"
