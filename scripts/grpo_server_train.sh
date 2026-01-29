#!/usr/bin/env bash
# Backward-compatible wrapper for the historical combined launcher.
#
# `scripts/grpo_server_mode.sh` is the canonical unified launcher for server-mode GRPO
# (starts rollout server + learner). This wrapper keeps old entrypoints working while
# steering operators to the new script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[WARN] scripts/grpo_server_train.sh is deprecated; use scripts/grpo_server_mode.sh instead." >&2

# Preserve the old default config if the caller didn't provide one.
has_config="false"
for arg in "$@"; do
  if [[ "${arg}" == config=* ]]; then
    has_config="true"
    break
  fi
done

args=()
for arg in "$@"; do
  if [[ "${arg}" == CONDA_ENV=* ]]; then
    args+=("conda_env=${arg#CONDA_ENV=}")
  else
    args+=("${arg}")
  fi
done

if [[ "${has_config}" != "true" ]]; then
  args+=("config=configs/train/grpo/summary_server.yaml")
fi

if [[ -n "${CONDA_ENV:-}" ]] && [[ -z "${conda_env:-}" ]]; then
  export conda_env="${CONDA_ENV}"
fi

exec bash "${SCRIPT_DIR}/grpo_server_mode.sh" "${args[@]}"
