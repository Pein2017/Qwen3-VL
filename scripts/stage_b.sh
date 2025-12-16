#!/usr/bin/env bash
set -euo pipefail

# Stage-B training-free pipeline convenience launcher
# Usage examples:
#   gpus=0 bash scripts/stage_b.sh
#   gpus=1 log_level=debug bash scripts/stage_b.sh
#   gpus=0,1,2,3,4,5,6,7 bash scripts/stage_b.sh   # multi-GPU (single node)
#   config=configs/stage_b/run.yaml gpus=0 bash scripts/stage_b.sh
#   gpus=0 bash scripts/stage_b.sh --step all --log-level warning
#   bash scripts/stage_b.sh smoke                # no-model audit (fast)

# Resolve repository root from this script's location and set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Default configuration (can be overridden via environment variable)
CONFIG_PATH="${config:-configs/stage_b/debug.yaml}"
CONDA_ENV="ms"
LOG_LEVEL="${log_level:-debug}"
DEBUG_FLAG="${debug:-false}"

if [[ "${1:-}" == "smoke" ]]; then
  shift
  cd "${REPO_DIR}"
  exec conda run -n "${CONDA_ENV}" --no-capture-output python -u scripts/stage_b_smoke.py "$@"
fi

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
CUDA_VISIBLE_DEVICES="${gpus:-1}"

# Derive number of GPUs (ignore empty/whitespace tokens)
IFS=',' read -r -a _raw_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
gpu_array=()
for _dev in "${_raw_gpu_array[@]}"; do
  [[ -n "${_dev// }" ]] && gpu_array+=("${_dev}")
done
NUM_GPUS="${#gpu_array[@]}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[ERROR] Invalid gpus list: '${CUDA_VISIBLE_DEVICES}'" >&2
  exit 1
fi

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
case "${DEBUG_FLAG,,}" in
  1|true|yes)
    ARGS+=("--debug")
    ;;
esac

# Print configuration
echo "=================================="
echo "Stage-B Training-Free Pipeline"
echo "=================================="
echo "Config file:  ${CONFIG_PATH}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES} (num=${NUM_GPUS})"
echo "Conda env:    ${CONDA_ENV}"
echo "Log level:    ${LOG_LEVEL}"
echo "Debug mode:   ${DEBUG_FLAG}"
echo "=================================="
echo ""

# Run from repo root using conda run
cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Force unbuffered Python output so tqdm/logging flush progressively
export PYTHONUNBUFFERED=1
# Run with any additional arguments (e.g., --step all)
# Note: Additional args can override defaults (e.g., --log-level warning)
# Pass PYTHONUNBUFFERED explicitly to conda run to ensure it's inherited
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  # Fail fast on errors; single node only.
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$((10000 + RANDOM % 55536))
  fi
  conda run -n "${CONDA_ENV}" --no-capture-output \
    torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" --max_restarts=0 \
    -m src.stage_b.runner "${ARGS[@]}" "$@"
else
  conda run -n "${CONDA_ENV}" --no-capture-output python -u -m src.stage_b.runner "${ARGS[@]}" "$@"
fi
STATUS=$?
if [[ ${STATUS} -ne 0 ]]; then
  exit ${STATUS}
fi

echo ""
echo "=================================="
echo "Stage-B Postprocess: Review Checklist"
echo "=================================="
conda run -n "${CONDA_ENV}" --no-capture-output python -u scripts/stage_b_postprocess_review_checklist.py --config "${CONFIG_PATH}"
