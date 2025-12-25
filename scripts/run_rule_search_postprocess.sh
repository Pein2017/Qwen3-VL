#!/usr/bin/env bash
set -euo pipefail

# === Configuration ===
HARD_CASES_ROOT="/data/Qwen3-VL/output_post/stage_b"
EXCEL_PATH="/data/Qwen3-VL/output_post/BBU_scene_latest.xlsx"
OUTPUT_ROOT=""
RECURSIVE=true
OVERWRITE=true
MISSION="BBU安装方式检查（正装）"

# === Execution ===
SCRIPT_PATH="/data/Qwen3-VL/scripts/postprocess_rule_search_hard_cases.py"

args=(--hard-cases "$HARD_CASES_ROOT" --excel "$EXCEL_PATH")

if [[ -n "$MISSION" ]]; then
  args=(--hard-cases "$HARD_CASES_ROOT/$MISSION" --excel "$EXCEL_PATH")
fi
if [[ -n "$OUTPUT_ROOT" ]]; then
  args+=(--output "$OUTPUT_ROOT")
fi
if [[ "$RECURSIVE" == "true" ]]; then
  args+=(--recursive)
fi
if [[ "$OVERWRITE" == "true" ]]; then
  args+=(--overwrite)
fi

conda run -n ms python "$SCRIPT_PATH" "${args[@]}"
