#!/bin/bash

# Data Conversion Pipeline - Manual Configuration
#
# Processes raw dataset directories into training-ready format:
# /data/{dataset_name}/
#   ├── images/*.jpeg        # Smart-resized images
#   ├── all_samples.jsonl    # All processed samples
#   ├── train.jsonl          # Training split
#   ├── val.jsonl            # Validation split
#   ├── label_vocabulary.json # Statistics
#   ├── validation_results.json # Validation results
#   ├── invalid_objects.jsonl # Invalid objects with details
#   ├── invalid_samples.jsonl # Invalid samples for visualization
#   └── fixed_objects.jsonl # Fixed objects (if enabled)
#
# REQUIRED: Set all configuration variables below before running!

set -e

# Set proper locale for UTF-8 handling
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

# Environment setup
# Prioritize current project directory to use updated code
# Get project root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Dynamically determine project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set PROJECT_ROOT dynamically
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODELSCOPE_CACHE="./modelscope/hub"

# ============================================================================
# CONFIGURATION (can be overridden via environment variables)
# ============================================================================

# Mode: prod | smoke
MODE="${MODE:-prod}"

# Dataset selector: bbu | rru
DATASET="${DATASET:-bbu}"

# Processing parameters
VAL_RATIO="${VAL_RATIO:-0.2}"
RESIZE="${RESIZE:-true}"
MAX_PIXELS="${MAX_PIXELS:-786432}"
IMAGE_FACTOR="${IMAGE_FACTOR:-32}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SEED="${SEED:-17}"
STRIP_OCCLUSION="${STRIP_OCCLUSION:-true}"
SANITIZE_TEXT="${SANITIZE_TEXT:-true}"
STANDARDIZE_LABEL_DESC="${STANDARDIZE_LABEL_DESC:-true}"
FAIL_FAST="${FAIL_FAST:-true}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# Limit is derived from MODE unless explicitly set
LIMIT="${LIMIT:-}"

# ============================================================================
# DATASET PRESETS
# ============================================================================

case "$DATASET" in
  bbu)
    DEFAULT_INPUT="raw_ds/bbu_scene_2.0/bbu_scene_2.0"
    DEFAULT_NAME="bbu_full_768_poly-need_review"
    ;;
  rru)
    DEFAULT_INPUT="raw_ds/rru_scene/rru_scene"
    DEFAULT_NAME="rru_full_768_poly"
    ;;
  *)
    echo "❌ Unknown DATASET: $DATASET (expected 'bbu' or 'rru')"
    exit 1
    ;;
esac

# ============================================================================
# MODE PRESETS
# ============================================================================

if [ "$MODE" = "smoke" ]; then
  OUTPUT_DIR="${OUTPUT_DIR:-data_smoke}"
  DATASET_NAME="${DATASET_NAME:-${DEFAULT_NAME}_smoke}"
  LIMIT="${LIMIT:--1}"
  # For smoke tests we keep limit small if not overridden
  if [ "$LIMIT" = "-1" ]; then
    LIMIT="2"
  fi
else
  OUTPUT_DIR="${OUTPUT_DIR:-data}"
  DATASET_NAME="${DATASET_NAME:-$DEFAULT_NAME}"
  LIMIT="${LIMIT:--1}"
fi

# Allow manual override of input dir after presets
INPUT_DIR="${INPUT_DIR:-$DEFAULT_INPUT}"

# Validation settings - OPTIONAL (currently hardcoded in pipeline/unified_processor.py)
# TODO: These parameters will be configurable in a future update
VALIDATION_MODE="strict"            # e.g., "strict", "lenient", "warning_only"
MIN_OBJECT_SIZE="1"               # e.g., "10" for minimum object size in pixels
ENABLE_VALIDATION_REPORTS="true"   # "true" or "false" to enable detailed validation reports

# ============================================================================
# SIMPLIFIED VALIDATION - Python config handles detailed validation
# ============================================================================

echo "🔍 Basic configuration check..."

# Only check critical path existence - Python handles the rest
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Auto-detect dataset name if not provided
if [ -z "$DATASET_NAME" ]; then
    DATASET_NAME=$(basename "$INPUT_DIR")
    echo "🔄 Auto-detected dataset name: $DATASET_NAME"
fi

echo "✅ Basic validation passed - Python will handle detailed validation"

# ============================================================================
# PROCESSING
# ============================================================================

echo ""
echo "🚀 Starting Data Conversion Pipeline"
echo "======================================"
echo "📋 Configuration:"
echo "   Input Dir: $INPUT_DIR"
echo "   Output Dir: $OUTPUT_DIR"
echo "   Dataset Name: $DATASET_NAME"
echo "   Val Ratio: $VAL_RATIO"
echo "   Smart Resize: $RESIZE"
echo "   Max Pixels: $MAX_PIXELS"
echo "   Image Factor: $IMAGE_FACTOR"
echo "   Log Level: $LOG_LEVEL"
echo "   Seed: $SEED"
echo "   Fail Fast: $FAIL_FAST"
if [ "$LIMIT" = "-1" ]; then
    echo "   Limit: $LIMIT (all images)"
else
    echo "   Limit: $LIMIT images"
fi
echo "   Validation Mode: $VALIDATION_MODE (hardcoded: strict)"
echo "   Min Object Size: $MIN_OBJECT_SIZE (hardcoded: 10px)"
echo "   Validation Reports: Always enabled"
echo "   Parallel Workers: $NUM_WORKERS"
echo ""

# Build command arguments
# Use conda run to ensure correct environment, or fallback to direct python if conda not available
if command -v conda &> /dev/null; then
    PYTHON_CMD="conda run -n ms python -m data_conversion.pipeline.unified_processor"
else
    # Fallback: try to find python in conda env
    CONDA_PYTHON="${CONDA_PREFIX:-/root/miniconda3/envs/ms}/bin/python"
    if [ -f "$CONDA_PYTHON" ]; then
        PYTHON_CMD="$CONDA_PYTHON -m data_conversion.pipeline.unified_processor"
    else
        PYTHON_CMD="python -m data_conversion.pipeline.unified_processor"
    fi
fi
ARGS="--input_dir \"$INPUT_DIR\""
ARGS="$ARGS --output_dir \"$OUTPUT_DIR\""
ARGS="$ARGS --dataset_name \"$DATASET_NAME\""
ARGS="$ARGS --val_ratio \"$VAL_RATIO\""
ARGS="$ARGS --seed \"$SEED\""
ARGS="$ARGS --log_level \"$LOG_LEVEL\""
ARGS="$ARGS --max_pixels \"$MAX_PIXELS\""
ARGS="$ARGS --image_factor \"$IMAGE_FACTOR\""
ARGS="$ARGS --validation_mode \"$VALIDATION_MODE\""
ARGS="$ARGS --min_object_size \"$MIN_OBJECT_SIZE\""

# Add optional arguments if provided
if [ "$RESIZE" = "true" ]; then
    ARGS="$ARGS --resize"
fi

if [ "$STRIP_OCCLUSION" = "true" ]; then
    ARGS="$ARGS --strip_occlusion"
fi
if [ "$SANITIZE_TEXT" = "true" ]; then
    ARGS="$ARGS --sanitize_text"
fi
if [ "$STANDARDIZE_LABEL_DESC" = "true" ]; then
    ARGS="$ARGS --standardize_label_desc"
fi

if [ "$FAIL_FAST" = "true" ]; then
    ARGS="$ARGS --fail_fast"
else
    ARGS="$ARGS --no_fail_fast"
fi

if [ "$ENABLE_VALIDATION_REPORTS" = "true" ]; then
    ARGS="$ARGS --enable_validation_reports"
else
    ARGS="$ARGS --disable_validation_reports"
fi

if [ -n "$LIMIT" ] && [ "$LIMIT" != "-1" ]; then
    ARGS="$ARGS --limit \"$LIMIT\""
fi

# Add num_workers argument
if [ -n "$NUM_WORKERS" ]; then
    ARGS="$ARGS --num_workers \"$NUM_WORKERS\""
fi

echo "🔄 Processing dataset: $DATASET_NAME ($INPUT_DIR)"
echo "  └─ Executing: $PYTHON_CMD"
echo "  └─ Logging to: convert.log"
echo ""

# Execute the command and redirect output to convert.log
# Use tee to also show output on terminal while logging to file
eval "$PYTHON_CMD $ARGS" 2>&1 | tee convert.log
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Dataset $DATASET_NAME processed successfully!"
    echo "📁 Output: $OUTPUT_DIR/$DATASET_NAME/"
    echo "📝 Log file: convert.log"
    echo ""

    # Build tiny debug splits for quick smoke tests
    OUTPUT_DATASET_DIR="$OUTPUT_DIR/$DATASET_NAME"
    export OUTPUT_DATASET_DIR
    if [ -f "$OUTPUT_DATASET_DIR/train.jsonl" ] && [ -f "$OUTPUT_DATASET_DIR/val.jsonl" ]; then
        echo "🪄 Creating tiny debug datasets (seed: $SEED)..."
        python - <<'PY'
import os
import random

dataset_dir = os.environ["OUTPUT_DATASET_DIR"]
train_path = os.path.join(dataset_dir, "train.jsonl")
val_path = os.path.join(dataset_dir, "val.jsonl")

seed_env = os.environ.get("SEED", "")
try:
    seed = int(seed_env)
except (TypeError, ValueError):
    seed = 17

rng = random.Random(seed)


def sample_file(src_path: str, dst_path: str, target_count: int) -> None:
    if not os.path.isfile(src_path):
        print(f"   ⚠️ Missing source: {src_path}")
        return

    with open(src_path, "r", encoding="utf-8") as src:
        lines = src.readlines()

    if not lines:
        print(f"   ⚠️ Source empty: {src_path}")
        open(dst_path, "w", encoding="utf-8").close()
        return

    k = min(target_count, len(lines))
    indices = rng.sample(range(len(lines)), k)
    indices.sort()

    with open(dst_path, "w", encoding="utf-8") as dst:
        for idx in indices:
            dst.write(lines[idx])

    print(f"   ✅ {os.path.basename(dst_path)}: {k}/{len(lines)} samples (seed={seed})")


sample_file(train_path, os.path.join(dataset_dir, "train_tiny.jsonl"), 20)
sample_file(val_path, os.path.join(dataset_dir, "val_tiny.jsonl"), 8)
PY
    else
        echo "⚠️ train.jsonl or val.jsonl missing; skipping tiny datasets"
    fi
    
    # Validation step has been removed for simplification
    
    echo "🚀 Ready for training!"
    exit 0
else
    echo ""
    echo "❌ Dataset $DATASET_NAME processing failed"
    echo "📝 Check convert.log for details"
    exit $EXIT_CODE
fi
