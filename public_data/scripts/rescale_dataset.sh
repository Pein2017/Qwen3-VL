#!/usr/bin/env bash
# Thin wrapper to offline-resize a dataset (images + geometry) using the shared resizer.
# Edit the variables below to match your dataset; no CLI arguments required.

set -euo pipefail

# --------------------------------------------------------------------------- #
# EDIT THESE FOR YOUR RUN
# --------------------------------------------------------------------------- #
INPUT_DIR="public_data/lvis/processed"          # folder with images/ + JSONLs
OUTPUT_DIR="public_data/lvis/processed_resized_768" # destination folder
FACTOR="32"                                                    # grid alignment (e.g., 32 or 28)
MAX_BLOCKS="768"                                               # pixel budget = MAX_BLOCKS * FACTOR^2
MIN_BLOCKS="4"                                                 # minimum pixel budget blocks
# --------------------------------------------------------------------------- #

echo "Resizing dataset"
echo "  input: ${INPUT_DIR}"
echo "  output: ${OUTPUT_DIR}"
echo "  factor: ${FACTOR}"
echo "  max_blocks: ${MAX_BLOCKS}"
echo "  min_blocks: ${MIN_BLOCKS}"

PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"
# Validate input
if [ ! -d "${INPUT_DIR}" ]; then
  echo "Input dir does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" data_conversion/resize_dataset.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --factor "${FACTOR}" \
  --max_pixel_blocks "${MAX_BLOCKS}" \
  --min_pixel_blocks "${MIN_BLOCKS}"
