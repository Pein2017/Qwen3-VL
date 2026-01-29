#!/usr/bin/env bash
set -euo pipefail

# Stage-B training-free pipeline convenience launcher
# Usage examples:
#   gpus=0 bash scripts/stage_b.sh
#   gpus=1 log_level=debug bash scripts/stage_b.sh
#   gpus=0,1,2,3,4,5,6,7 bash scripts/stage_b.sh   # multi-GPU (single node)
#   config=bbu_line gpus=0 bash scripts/stage_b.sh  # config name (resolves to configs/stage_b/bbu_line.yaml)[[[]]
#   config=configs/stage_b/bbu_line.yaml gpus=0 bash scripts/stage_b.sh  # relative path
#   config=/absolute/path/to/config.yaml gpus=0 bash scripts/stage_b.sh  # absolute path
#   gpus=0 bash scripts/stage_b.sh --step all --log-level warning
#   bash scripts/stage_b.sh smoke                # no-model audit (fast)

# Resolve repository root from this script's location and set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Default configuration (can be overridden via environment variable)
CONFIG_RAW="${config:-debug}"
CONDA_ENV="ms"
LOG_LEVEL="${log_level:-debug}"
DEBUG_FLAG="${debug:-false}"
# Convenience flag: skip proposer/reflection and only run baseline rollouts + dumps.
JUMP_REFLECTION="${jump_reflection:-false}"
# Launch mode controls how we use multiple GPUs:
# - auto (default): 1 GPU → single-process; >1 GPU → torchrun ticket-parallel
# - model_parallel: always single-process Python, even if multiple GPUs are visible
STAGE_B_MODE="${stage_b_mode:-auto}"

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
  _trimmed="${_dev//[[:space:]]/}"
  [[ -n "${_trimmed}" ]] && gpu_array+=("${_trimmed}")
done
# Normalize CUDA_VISIBLE_DEVICES to a comma-separated list without whitespace.
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_array[*]}")"
NUM_GPUS="${#gpu_array[@]}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[ERROR] Invalid gpus list: '${CUDA_VISIBLE_DEVICES}'" >&2
  exit 1
fi

# Resolve CONFIG_RAW to absolute path or repo-relative
# Similar to train.sh: supports absolute paths, .yaml files, or config names
if [[ "${CONFIG_RAW}" = /* ]]; then
  CONFIG_PATH="${CONFIG_RAW}"
elif [[ "${CONFIG_RAW}" == *.yaml ]]; then
  CONFIG_PATH="${REPO_DIR}/${CONFIG_RAW}"
else
  CONFIG_PATH="${REPO_DIR}/configs/stage_b/${CONFIG_RAW}.yaml"
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
case "${JUMP_REFLECTION,,}" in
  1|true|yes)
    ARGS+=("--jump-reflection")
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
echo "Jump reflect: ${JUMP_REFLECTION}  # env override; can also be set in YAML via jump_reflection: true"
echo "Stage-B mode: ${STAGE_B_MODE}  # auto | model_parallel"
echo "=================================="
echo ""

# Run from repo root using conda run
cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Force unbuffered Python output so logging flushes progressively
export PYTHONUNBUFFERED=1
# Run with any additional arguments (e.g., --step all)
# Note: Additional args can override defaults (e.g., --log-level warning)
# Pass PYTHONUNBUFFERED explicitly to conda run to ensure it's inherited
if [[ "${STAGE_B_MODE}" == "model_parallel" ]]; then
  # Single-process launch even when multiple GPUs are visible. This allows
  # Hugging Face/Qwen device_map-based sharding across CUDA_VISIBLE_DEVICES.
  conda run -n "${CONDA_ENV}" --no-capture-output python -u -m src.stage_b.runner "${ARGS[@]}" "$@"
else
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
fi
STATUS=$?
if [[ ${STATUS} -ne 0 ]]; then
  exit ${STATUS}
fi
