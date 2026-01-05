#!/usr/bin/env bash
# Launch vLLM server (swift rollout) + GRPO training in one entrypoint.

set -euo pipefail

# Allow passing key=value pairs as positional args (common launcher convention).
# Example:
#   bash scripts/grpo_server_train.sh server_gpus=0,1 train_gpus=2,3 config=configs/train/grpo/summary_server.yaml
for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "[ERROR] Unknown argument: ${arg} (expected key=value)" >&2
    exit 2
  fi
  key="${arg%%=*}"
  if [[ ! "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "[ERROR] Invalid argument name in: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

# Defaults (override via env vars)
CONDA_ENV="${CONDA_ENV:-ms}"
SERVER_GPUS="${server_gpus:-0,1}"
TRAIN_GPUS="${train_gpus:-2,3,4,5,6,7}"
WAIT_TIMEOUT="${wait_timeout:-120}"
WAIT_INTERVAL="${wait_interval:-2}"
CONFIG_RAW="${config:-configs/train/grpo/summary_server.yaml}"
DEBUG="${debug:-false}"
TRAIN_ENV="${train_env:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve CONFIG_RAW to absolute path or repo-relative (same rules as scripts/train.sh)
if [[ "${CONFIG_RAW}" = /* ]]; then
  CONFIG_PATH="${CONFIG_RAW}"
elif [[ "${CONFIG_RAW}" == configs/* ]]; then
  CONFIG_PATH="${REPO_DIR}/${CONFIG_RAW}"
elif [[ "${CONFIG_RAW}" == *.yaml ]]; then
  CONFIG_PATH="${REPO_DIR}/configs/${CONFIG_RAW}"
else
  CONFIG_PATH="${REPO_DIR}/configs/${CONFIG_RAW}.yaml"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

PY_CODE=$(cat <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print("[ERROR] PyYAML is required to parse training configs.", file=sys.stderr)
    raise


def die(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        die(f"Top-level YAML config must be a mapping: {path}")
    return data


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def merge_configs(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = merge_configs(existing, value)
        else:
            merged[key] = value
    return merged


def load_with_extends(path: Path, visited: set[Path] | None = None) -> dict:
    abs_path = path.resolve()
    visited = set() if visited is None else visited
    if abs_path in visited:
        die(f"Cyclic config inheritance detected at: {abs_path}")
    visited.add(abs_path)

    config = load_yaml(abs_path)
    if "inherit" in config:
        die("Config inheritance uses 'extends'; 'inherit' is not supported.")
    extends_value = config.pop("extends", None)

    merged_base: dict = {}
    for base_ref in normalize_to_list(extends_value):
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (abs_path.parent / base_path).resolve()
        merged_base = merge_configs(merged_base, load_with_extends(base_path, visited))

    return merge_configs(merged_base, config)


config_path = os.environ.get("CONFIG_PATH")
if not config_path:
    die("CONFIG_PATH is required for rollout parsing.")

cfg = load_with_extends(Path(config_path))
rlhf = cfg.get("rlhf") or {}
model = cfg.get("model") or {}

global_max_length = cfg.get("global_max_length")
if rlhf.get("vllm_max_model_len") is None and global_max_length is not None:
    rlhf["vllm_max_model_len"] = global_max_length

if not rlhf.get("use_vllm", False):
    die("rlhf.use_vllm must be true for server rollout.")

if rlhf.get("vllm_mode") != "server":
    die(f"rlhf.vllm_mode must be 'server', got {rlhf.get('vllm_mode')!r}.")

model_path = model.get("model")
if not model_path:
    die("model.model must be set (rollout model path).")

tp = rlhf.get("vllm_tensor_parallel_size")
dp = rlhf.get("vllm_data_parallel_size")
if tp is None or dp is None:
    die("rlhf.vllm_tensor_parallel_size and rlhf.vllm_data_parallel_size are required.")
try:
    tp = int(tp)
    dp = int(dp)
except (TypeError, ValueError) as exc:
    die(f"vllm_tensor_parallel_size and vllm_data_parallel_size must be integers: {exc}")

hosts = rlhf.get("vllm_server_host")
if hosts is None:
    die("rlhf.vllm_server_host is required.")
if isinstance(hosts, str):
    hosts = [hosts]
if not isinstance(hosts, list):
    die("rlhf.vllm_server_host must be a list or string.")
if len(hosts) != 1:
    die("grpo_server_train.sh supports exactly one vllm_server_host; use a multi-node launcher for more.")
host = str(hosts[0])

ports = rlhf.get("vllm_server_port")
if ports is None:
    die("rlhf.vllm_server_port is required.")
if isinstance(ports, (int, str)):
    ports = [ports]
if not isinstance(ports, list):
    die("rlhf.vllm_server_port must be a list or integer.")
if len(ports) != 1:
    die("grpo_server_train.sh supports exactly one vllm_server_port; use a multi-node launcher for more.")
try:
    port = int(ports[0])
except (TypeError, ValueError) as exc:
    die(f"vllm_server_port must be an integer: {exc}")

if rlhf.get("vllm_reasoning_parser") not in (None, ""):
    die("vllm_reasoning_parser is not supported for swift rollout; remove it from rlhf.*.")

rollout_args = [
    "--vllm_tensor_parallel_size",
    str(tp),
    "--vllm_data_parallel_size",
    str(dp),
]

optional_flags = {
    "vllm_gpu_memory_utilization": "--vllm_gpu_memory_utilization",
    "vllm_max_model_len": "--vllm_max_model_len",
    "vllm_max_num_seqs": "--vllm_max_num_seqs",
    "vllm_enable_prefix_caching": "--vllm_enable_prefix_caching",
    "vllm_disable_custom_all_reduce": "--vllm_disable_custom_all_reduce",
    "vllm_enforce_eager": "--vllm_enforce_eager",
    "vllm_limit_mm_per_prompt": "--vllm_limit_mm_per_prompt",
    "vllm_enable_lora": "--vllm_enable_lora",
    "vllm_max_lora_rank": "--vllm_max_lora_rank",
    "vllm_use_async_engine": "--vllm_use_async_engine",
    "vllm_engine_kwargs": "--vllm_engine_kwargs",
    "vllm_mm_processor_cache_gb": "--vllm_mm_processor_cache_gb",
    "vllm_disable_cascade_attn": "--vllm_disable_cascade_attn",
    "vllm_quantization": "--vllm_quantization",
}

for key, flag in optional_flags.items():
    if key not in rlhf:
        continue
    value = rlhf.get(key)
    if value is None:
        continue
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, bool):
        value = "true" if value else "false"
    else:
        value = str(value)
    rollout_args.extend([flag, value])


def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")


emit("SERVER_HOST", host)
emit("SERVER_PORT", port)
emit("SERVER_MODEL", model_path)
emit("VLLM_TP", tp)
emit("VLLM_DP", dp)
print("ROLLOUT_ARGS=(" + " ".join(shlex.quote(a) for a in rollout_args) + ")")
PY
)

CONFIG_VARS="$(
  CONFIG_PATH="${CONFIG_PATH}" conda run -n "${CONDA_ENV}" python -c "${PY_CODE}"
)" || {
  echo "[ERROR] Failed to resolve rollout settings from ${CONFIG_PATH}" >&2
  exit 1
}

if [[ -z "${CONFIG_VARS}" ]]; then
  echo "[ERROR] Failed to resolve rollout settings from ${CONFIG_PATH}" >&2
  exit 1
fi

eval "${CONFIG_VARS}"

echo "========================================================================"
echo "  GRPO Server + Training Launcher"
echo "========================================================================"
echo "[INFO] Server GPUs: ${SERVER_GPUS}"
echo "[INFO] Train GPUs:  ${TRAIN_GPUS}"
echo "[INFO] Server:      ${SERVER_HOST}:${SERVER_PORT}"
echo "[INFO] Model:       ${SERVER_MODEL}"
echo "[INFO] TP/DP:       ${VLLM_TP}/${VLLM_DP}"
echo "[INFO] Rollout:     ${ROLLOUT_ARGS[*]}"
echo "[INFO] Config:      ${CONFIG_PATH}"
echo "========================================================================"

SERVER_ENV=(CUDA_VISIBLE_DEVICES="${SERVER_GPUS}")
SERVER_CMD=(conda run -n "${CONDA_ENV}" swift rollout \
  --model "${SERVER_MODEL}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  "${ROLLOUT_ARGS[@]}")

echo "[RUN] ${SERVER_ENV[*]} ${SERVER_CMD[*]}"
if command -v setsid >/dev/null 2>&1; then
  setsid env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}" &
  SERVER_PID=$!
  SERVER_PGID="$(ps -o pgid= "${SERVER_PID}" | tr -d ' ')"
else
  env "${SERVER_ENV[@]}" "${SERVER_CMD[@]}" &
  SERVER_PID=$!
  SERVER_PGID=""
fi

cleanup() {
  if [[ -n "${SERVER_PGID}" ]]; then
    echo "[INFO] Stopping vLLM server process group (pgid ${SERVER_PGID})"
    kill -TERM "-${SERVER_PGID}" >/dev/null 2>&1 || true
    sleep 3
    kill -KILL "-${SERVER_PGID}" >/dev/null 2>&1 || true
  elif kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[INFO] Stopping vLLM server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

HEALTH_HOST="${SERVER_HOST}"
if [[ "${HEALTH_HOST}" == "0.0.0.0" ]]; then
  HEALTH_HOST="127.0.0.1"
fi
HEALTH_URL="http://${HEALTH_HOST}:${SERVER_PORT}/health/"

echo "[INFO] Waiting for vLLM server readiness: ${HEALTH_URL}"
start_ts=$(date +%s)
while true; do
  http_code=$(curl -s -o /dev/null -w '%{http_code}' "${HEALTH_URL}" || true)
  if [[ "${http_code}" == "200" ]]; then
    break
  fi
  now_ts=$(date +%s)
  if (( now_ts - start_ts > WAIT_TIMEOUT )); then
    echo "[ERROR] vLLM server did not become ready within ${WAIT_TIMEOUT}s. Last HTTP code: ${http_code}" >&2
    exit 1
  fi
  sleep "${WAIT_INTERVAL}"
done
echo "[INFO] vLLM server is ready."

TRAIN_CMD="config=${CONFIG_PATH} gpus=${TRAIN_GPUS} bash ${REPO_DIR}/scripts/train.sh"
if [[ "${DEBUG}" == "true" ]]; then
  TRAIN_CMD="debug=true ${TRAIN_CMD}"
fi
if [[ -n "${TRAIN_ENV}" ]]; then
  TRAIN_CMD="${TRAIN_ENV} ${TRAIN_CMD}"
fi

echo "[RUN] ${TRAIN_CMD}"
eval "${TRAIN_CMD}"
