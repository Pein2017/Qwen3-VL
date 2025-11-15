#!/usr/bin/env bash
set -euo pipefail

# Stage-B training-free pipeline convenience launcher
# Usage examples:
#   gpus=0 bash scripts/stage_b_run.sh
#   gpus=1 log_level=debug bash scripts/stage_b_run.sh
#   config=configs/stage_b/run.yaml gpus=0 bash scripts/stage_b_run.sh
#   gpus=0 bash scripts/stage_b_run.sh --step all --log-level warning

# Resolve repository root from this script's location and set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Default configuration (can be overridden via environment variable)
CONFIG_PATH="${config:-configs/stage_b/debug.yaml}"
CONDA_ENV="ms"
LOG_LEVEL="${log_level:-logging}"

case "${LOG_LEVEL,,}" in
  debug|logging|warning)
    LOG_LEVEL="${LOG_LEVEL,,}"
    ;;
  *)
    echo "[ERROR] Unsupported log_level: ${LOG_LEVEL}. Use one of: debug, logging, warning." >&2
    exit 1
    ;;
esac

# Environment variable overrides (lowercase)
gpu_id="${gpus:-0}"
CUDA_VISIBLE_DEVICES="${gpu_id}"

# Resolve CONFIG_PATH to absolute if relative
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_DIR}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# Build arguments array
ARGS=("--config" "${CONFIG_PATH}" "--log-level" "${LOG_LEVEL}")

# Print configuration
echo "=================================="
echo "Stage-B Training-Free Pipeline"
echo "=================================="
echo "Config file:  ${CONFIG_PATH}"
echo "GPU:          ${CUDA_VISIBLE_DEVICES}"
echo "Conda env:    ${CONDA_ENV}"
echo "Log level:    ${LOG_LEVEL}"
echo "=================================="
echo ""

# Run from repo root using conda run
cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Force unbuffered Python output so tqdm/logging flush progressively
PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
# Run with any additional arguments (e.g., --step all)
# Note: Additional args can override defaults (e.g., --log-level warning)
exec conda run -n "${CONDA_ENV}" python -u -m src.stage_b.runner "${ARGS[@]}" "$@"

