#!/usr/bin/env bash
# End-to-end pipeline to convert a source detection dataset and smart-resize
# images/geometry using the shared preprocessor. Produces resized images + JSONL.
#
# Defaults target LVIS; edit the variables below as needed.

set -euo pipefail

# --------------------------------------------------------------------------- #
# EDIT THESE FOR YOUR RUN
# --------------------------------------------------------------------------- #
RAW_ROOT="public_data/lvis/raw"              # contains annotations/ and images/
OUTPUT_ROOT="public_data/lvis/resized_32_768_v2" # final resized images + JSONLs
RUN_VAL="false"                                             # set to "true" to also process val split
FACTOR="32"                                                 # grid alignment
MAX_BLOCKS="768"                                            # pixel budget, blocks * FACTOR^2
MIN_BLOCKS="4"                                              # minimum pixel budget blocks
PYTHON_BIN="/root/miniconda3/envs/ms/bin/python"
# --------------------------------------------------------------------------- #

echo "=== Convert + Resize Source Dataset ==="
echo "RAW_ROOT:     ${RAW_ROOT}"
echo "OUTPUT_ROOT:  ${OUTPUT_ROOT}"
echo "FACTOR:       ${FACTOR}"
echo "MAX_BLOCKS:   ${MAX_BLOCKS}"
echo "MIN_BLOCKS:   ${MIN_BLOCKS}"
echo "RUN_VAL:      ${RUN_VAL}"

# Clean stale temp (legacy flow) and ensure output root exists
rm -rf "${OUTPUT_ROOT}/_tmp_unscaled"
mkdir -p "${OUTPUT_ROOT}/images"

MAX_PIXELS=$((FACTOR * FACTOR * MAX_BLOCKS))
MIN_PIXELS=$((FACTOR * FACTOR * MIN_BLOCKS))

echo "--- Converting + smart-resizing train split ---"
${PYTHON_BIN} public_data/scripts/convert_lvis.py \
  --split train \
  --use-polygon \
  --annotation "${RAW_ROOT}/annotations/lvis_v1_train.json" \
  --image_root "${RAW_ROOT}/images" \
  --smart-resize \
  --image_factor "${FACTOR}" \
  --max_pixels "${MAX_PIXELS}" \
  --min_pixels "${MIN_PIXELS}" \
  --resize_output_root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/train.jsonl"

if [ "${RUN_VAL}" = "true" ]; then
  echo "--- Converting + smart-resizing val split ---"
  ${PYTHON_BIN} public_data/scripts/convert_lvis.py \
    --split val \
    --use-polygon \
    --annotation "${RAW_ROOT}/annotations/lvis_v1_val.json" \
    --image_root "${RAW_ROOT}/images" \
    --smart-resize \
    --image_factor "${FACTOR}" \
    --max_pixels "${MAX_PIXELS}" \
    --min_pixels "${MIN_PIXELS}" \
    --resize_output_root "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/val.jsonl"
fi

echo "Done. Resized images + JSONL are in: ${OUTPUT_ROOT}"
