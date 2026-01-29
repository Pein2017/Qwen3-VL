#!/usr/bin/env bash
# Unified GRPO server-mode launcher: starts rollout server in the background, then runs learner in foreground.
#
# Goals:
# - Single entrypoint for operators (one command).
# - Only show learner logs in the console; rollout server logs go to a file.
# - Record PIDs and ensure rollout server is terminated when this script exits (CTRL-C, error, normal exit).
#
# Usage:
#   bash scripts/grpo_server_mode.sh \
#     config=configs/train/grpo/summary_1024_attr_key_recall.yaml \
#     server_gpus=0,1,2,3,4,5 \
#     train_gpus=6,7
#
# Notes:
# - Config must be server-mode (`rlhf.vllm_mode=server`) and local-only host/port (127.0.0.1/localhost).
# - `server_gpus` count must match `custom.extra.rollout_server.vllm_tensor_parallel_size * vllm_data_parallel_size`.

set -euo pipefail

# Allow passing key=value pairs as positional args (common launcher convention).
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

CONDA_ENV="${CONDA_ENV:-ms}"
CONFIG_RAW="${config:-}"
SERVER_GPUS="${server_gpus:-0,1,2,3,4,5}"
TRAIN_GPUS="${train_gpus:-6,7}"
WAIT_TIMEOUT="${wait_timeout:-300}"
WAIT_INTERVAL="${wait_interval:-2}"
DEBUG="${debug:-false}"
TRAIN_ENV="${train_env:-}"
FORCE_KILL_PORT="${force_kill_port:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${CONFIG_RAW}" ]]; then
  echo "[ERROR] config=<path> is required (e.g. config=configs/train/grpo/summary_1024_attr_key_recall.yaml)" >&2
  exit 2
fi

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

# Derive number of server GPUs (ignore empty/whitespace tokens)
IFS=',' read -r -a _raw_gpu_array <<< "${SERVER_GPUS}"
server_gpu_array=()
for _dev in "${_raw_gpu_array[@]}"; do
  [[ -n "${_dev// }" ]] && server_gpu_array+=("${_dev}")
done
VISIBLE_SERVER_GPU_COUNT="${#server_gpu_array[@]}"

# Extract and validate rollout-server launch settings from the config.
# This enforces:
# - local-only host/port
# - server-mode config present
# - TP*DP == visible server GPU count
# - vllm_max_model_len >= global_max_length
# - vLLM LoRA forbidden
PY_EXTRACT=$(cat <<'PY'
import os
import shlex
import sys

from src.config.loader import ConfigLoader
from src.rlhf.grpo.rollout_server_config import extract_rollout_server_launch_config


def die(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


cfg = ConfigLoader.load_yaml_with_extends(os.environ["CONFIG_PATH"])
visible_gpu_count_raw = os.environ.get("VISIBLE_GPU_COUNT", "0")
try:
    visible_gpu_count = int(visible_gpu_count_raw)
except ValueError:
    die(f"VISIBLE_GPU_COUNT must be an int, got {visible_gpu_count_raw!r}")

launch = extract_rollout_server_launch_config(cfg, visible_gpu_count=visible_gpu_count)

print(f"SERVER_HOST={shlex.quote(launch.connectivity.host)}")
print(f"SERVER_PORT={shlex.quote(str(launch.connectivity.port))}")
print(f"SERVER_MODEL={shlex.quote(launch.model_path)}")
print("ROLLOUT_ARGS=(" + " ".join(shlex.quote(a) for a in launch.rollout_args) + ")")
PY
)
# NOTE: Some dependencies can emit noisy logs to stdout during import (e.g., swift dataset registry logs).
# Because we `eval` the output as bash assignments, we must filter to only keep known assignment lines.
set +e
_extract_raw_out="$(
  CONFIG_PATH="${CONFIG_PATH}" VISIBLE_GPU_COUNT="${VISIBLE_SERVER_GPU_COUNT}" \
    conda run -n "${CONDA_ENV}" python -c "${PY_EXTRACT}"
)"
_extract_rc=$?
set -e

if [[ "${_extract_rc}" -ne 0 ]]; then
  echo "[ERROR] Failed to extract rollout server launch config from YAML (rc=${_extract_rc})." >&2
  echo "[ERROR] Raw output:" >&2
  printf '%s\n' "${_extract_raw_out}" >&2
  exit 1
fi

_extract_filtered="$(
  printf '%s\n' "${_extract_raw_out}" | sed -n -E 's/^(SERVER_HOST|SERVER_PORT|SERVER_MODEL|ROLLOUT_ARGS)=.*$/&/p'
)"
if [[ "${DEBUG}" == "true" ]]; then
  _discarded="$(
    printf '%s\n' "${_extract_raw_out}" | sed -n -E '/^(SERVER_HOST|SERVER_PORT|SERVER_MODEL|ROLLOUT_ARGS)=/!p'
  )"
  if [[ -n "${_discarded}" ]]; then
    echo "[DEBUG] Discarded non-assignment output from extractor:" >&2
    printf '%s\n' "${_discarded}" >&2
  fi
fi

if [[ -z "${_extract_filtered}" ]]; then
  echo "[ERROR] Rollout server extractor produced no assignment lines. Raw output:" >&2
  printf '%s\n' "${_extract_raw_out}" >&2
  exit 1
fi

eval "${_extract_filtered}"

timestamp="$(date +%Y%m%d-%H%M%S)"
SERVER_LOG="${server_log:-${REPO_DIR}/output/grpo_rollout_server/${timestamp}-port${SERVER_PORT}.log}"
mkdir -p "$(dirname "${SERVER_LOG}")"

echo "========================================================================"
echo "  GRPO Server-Mode Unified Launcher"
echo "========================================================================"
echo "[INFO] Config:       ${CONFIG_PATH}"
echo "[INFO] Server GPUs:  ${SERVER_GPUS} (num=${VISIBLE_SERVER_GPU_COUNT})"
echo "[INFO] Train GPUs:   ${TRAIN_GPUS}"
echo "[INFO] Server:       ${SERVER_HOST}:${SERVER_PORT}"
echo "[INFO] Server log:   ${SERVER_LOG}"
echo "[INFO] Conda env:    ${CONDA_ENV}"
echo "========================================================================"

SERVER_PID=""
SERVER_PGID=""

cleanup() {
  if [[ -n "${SERVER_PGID}" ]] && [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    # Extra safety: only kill the recorded PGID if it still matches the rollout wrapper PID.
    current_pgid="$(ps -o pgid= -p "${SERVER_PID}" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "${current_pgid}" ]] && [[ "${current_pgid}" == "${SERVER_PGID}" ]]; then
      echo "[INFO] Stopping rollout server process group (pgid ${SERVER_PGID})" >&2
      kill -TERM "-${SERVER_PGID}" >/dev/null 2>&1 || true
      sleep 3
      kill -KILL "-${SERVER_PGID}" >/dev/null 2>&1 || true
      return 0
    fi
    echo "[WARN] Skipping PGID kill (expected pgid=${SERVER_PGID}, got pgid=${current_pgid:-<unknown>}); falling back to PID kill." >&2
  fi

  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[INFO] Stopping rollout server (pid ${SERVER_PID})" >&2
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Starting rollout server (logs redirected to ${SERVER_LOG})"
# Ensure the requested port is available; optionally kill any listener that occupies it.
#
# NOTE: Killing by port can terminate unrelated processes. This is intended for
# operator workflows on dedicated training nodes. Set force_kill_port=false to disable.
kill_listeners_on_port() {
  local port="$1"
  local pids=""
  local have_tool="false"

  if command -v lsof >/dev/null 2>&1; then
    have_tool="true"
    # -t prints PIDs only; LISTEN only to avoid killing clients.
    pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' | xargs echo -n || true)"
  elif command -v ss >/dev/null 2>&1; then
    have_tool="true"
    # Example line: LISTEN 0 4096 0.0.0.0:8080 ... users:(("python",pid=123,fd=3))
    pids="$(ss -H -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u | tr '\n' ' ' | xargs echo -n || true)"
  fi

  if [[ "${have_tool}" != "true" ]]; then
    echo "[ERROR] Cannot kill occupied port ${port}: neither 'lsof' nor 'ss' is available." >&2
    return 1
  fi

  if [[ -z "${pids}" ]]; then
    return 0
  fi

  echo "[WARN] Port ${port} is occupied by PID(s): ${pids}" >&2
  # Provide visibility before killing.
  ps -o pid=,ppid=,pgid=,cmd= -p ${pids} 2>/dev/null || true

  echo "[WARN] Sending SIGTERM to PID(s) on port ${port}..." >&2
  kill -TERM ${pids} >/dev/null 2>&1 || true
  sleep 2

  # If any are still alive, force kill.
  for _pid in ${pids}; do
    if kill -0 "${_pid}" >/dev/null 2>&1; then
      echo "[WARN] PID ${_pid} still alive; sending SIGKILL" >&2
      kill -KILL "${_pid}" >/dev/null 2>&1 || true
    fi
  done
  sleep 1

  # Best-effort re-check.
  if command -v lsof >/dev/null 2>&1; then
    if lsof -tiTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "[ERROR] Port ${port} still has a listener after kill attempts." >&2
      return 1
    fi
  elif command -v ss >/dev/null 2>&1; then
    if ss -H -ltn "sport = :${port}" >/dev/null 2>&1; then
      echo "[ERROR] Port ${port} still appears to be listening after kill attempts." >&2
      return 1
    fi
  fi
}

PY_PORT_CHECK=$(cat <<'PY'
import errno
import os
import socket
import sys


def die(message: str, exit_code: int = 1) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.exit(exit_code)


host = os.environ.get("SERVER_HOST")
port_raw = os.environ.get("SERVER_PORT")
if not host or not port_raw:
    die("SERVER_HOST and SERVER_PORT are required for port preflight check.")
try:
    port = int(port_raw)
except ValueError:
    die(f"SERVER_PORT must be an int, got {port_raw!r}")

try:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
except socket.gaierror as exc:
    die(f"Failed to resolve host {host!r}: {exc}")

seen = set()
for family, socktype, proto, canonname, sockaddr in infos:
    key = (family, sockaddr)
    if key in seen:
        continue
    seen.add(key)
    sock = socket.socket(family, socktype, proto)
    try:
        sock.bind(sockaddr)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            die(f"Rollout port already in use: {host}:{port}", exit_code=100)
        die(f"Failed to bind {host}:{port} for preflight check: {exc}")
    finally:
        sock.close()
PY
)
set +e
SERVER_HOST="${SERVER_HOST}" SERVER_PORT="${SERVER_PORT}" \
  conda run -n "${CONDA_ENV}" python -c "${PY_PORT_CHECK}"
port_check_rc=$?
set -e

if [[ "${port_check_rc}" -ne 0 ]]; then
  if [[ "${port_check_rc}" -eq 100 ]] && [[ "${FORCE_KILL_PORT}" == "true" ]]; then
    echo "[WARN] Attempting to free occupied rollout port ${SERVER_PORT} (force_kill_port=true)" >&2
    kill_listeners_on_port "${SERVER_PORT}"
    SERVER_HOST="${SERVER_HOST}" SERVER_PORT="${SERVER_PORT}" \
      conda run -n "${CONDA_ENV}" python -c "${PY_PORT_CHECK}"
  else
    echo "[ERROR] Rollout server port preflight check failed (rc=${port_check_rc})." >&2
    echo "[ERROR] If you want to kill any process listening on the port, run with: force_kill_port=true" >&2
    exit 1
  fi
fi

if command -v setsid >/dev/null 2>&1; then
  # Start in its own process group so ctrl-c only interrupts the learner;
  # cleanup trap will then tear down the server.
  # stdbuf reduces log buffering (best-effort).
  if command -v stdbuf >/dev/null 2>&1; then
    setsid env CUDA_VISIBLE_DEVICES="${SERVER_GPUS}" \
      conda run -n "${CONDA_ENV}" stdbuf -oL -eL swift rollout \
        --model "${SERVER_MODEL}" \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        "${ROLLOUT_ARGS[@]}" \
        >"${SERVER_LOG}" 2>&1 &
  else
    setsid env CUDA_VISIBLE_DEVICES="${SERVER_GPUS}" \
    conda run -n "${CONDA_ENV}" swift rollout \
      --model "${SERVER_MODEL}" \
      --host "${SERVER_HOST}" \
      --port "${SERVER_PORT}" \
      "${ROLLOUT_ARGS[@]}" \
      >"${SERVER_LOG}" 2>&1 &
  fi
  SERVER_PID=$!
  SERVER_PGID="$(ps -o pgid= "${SERVER_PID}" | tr -d ' ')"
else
  if command -v stdbuf >/dev/null 2>&1; then
    env CUDA_VISIBLE_DEVICES="${SERVER_GPUS}" \
      conda run -n "${CONDA_ENV}" stdbuf -oL -eL swift rollout \
        --model "${SERVER_MODEL}" \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        "${ROLLOUT_ARGS[@]}" \
        >"${SERVER_LOG}" 2>&1 &
  else
    env CUDA_VISIBLE_DEVICES="${SERVER_GPUS}" \
      conda run -n "${CONDA_ENV}" swift rollout \
        --model "${SERVER_MODEL}" \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        "${ROLLOUT_ARGS[@]}" \
        >"${SERVER_LOG}" 2>&1 &
  fi
  SERVER_PID=$!
  SERVER_PGID=""
fi

HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/health/"
echo "[INFO] Waiting for rollout server readiness: ${HEALTH_URL}"
start_ts=$(date +%s)
while true; do
  http_code=$(curl -s --connect-timeout 2 --max-time 2 -o /dev/null -w '%{http_code}' "${HEALTH_URL}" || true)
  if [[ "${http_code}" == "200" ]]; then
    break
  fi
  # If the server died, fail fast and point to the log.
  if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[ERROR] Rollout server process exited early. See log: ${SERVER_LOG}" >&2
    exit 1
  fi
  now_ts=$(date +%s)
  if (( now_ts - start_ts > WAIT_TIMEOUT )); then
    echo "[ERROR] Rollout server did not become ready within ${WAIT_TIMEOUT}s. Last HTTP code: ${http_code}" >&2
    echo "[ERROR] See server log: ${SERVER_LOG}" >&2
    exit 1
  fi
  sleep "${WAIT_INTERVAL}"
done
echo "[INFO] Rollout server is ready."

TRAIN_CMD="config=${CONFIG_PATH} gpus=${TRAIN_GPUS} bash ${REPO_DIR}/scripts/train.sh"
if [[ "${DEBUG}" == "true" ]]; then
  TRAIN_CMD="debug=true ${TRAIN_CMD}"
fi
if [[ -n "${TRAIN_ENV}" ]]; then
  TRAIN_CMD="${TRAIN_ENV} ${TRAIN_CMD}"
fi
TRAIN_CMD="CONDA_ENV=${CONDA_ENV} ${TRAIN_CMD}"

echo "[RUN] ${TRAIN_CMD}"
eval "${TRAIN_CMD}"
