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
export PYTHONPATH=/data/Qwen3-VL:$PYTHONPATH

# Dynamically determine project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set PROJECT_ROOT dynamically
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODELSCOPE_CACHE="./modelscope/hub"

# ============================================================================
# MANUAL CONFIGURATION - EDIT THESE VALUES BEFORE RUNNING
# ============================================================================

# Required paths - YOU MUST SET THESE
INPUT_DIR="raw_ds/bbu_scene_2.0/bbu_scene_2.0"  # e.g., "ds_v2" or "my_dataset"
OUTPUT_DIR="data"                   # e.g., "data" or "/path/to/output"
DATASET_NAME="bbu_full_768_poly-parallel"         # e.g., "experiment_1" or leave empty to auto-detect


# Processing parameters - YOU MUST SET THESE
VAL_RATIO="0.2"                    # e.g., "0.1" for 10% validation split
RESIZE="true"                       # "true" or "false" for image resizing

# Image resize parameters - REQUIRED (NO DEFAULTS ALLOWED)
MAX_PIXELS="786432"                # Maximum pixels for image resizing (e.g., 786432 for 768*32*32)
IMAGE_FACTOR="32"                  # Factor for image dimensions (e.g., 32)

# Optional settings
LOG_LEVEL="INFO"                    # e.g., "INFO", "DEBUG", "WARNING", "ERROR" or leave empty
SEED="17"                         # e.g., "17" or leave empty
STRIP_OCCLUSION="true"            # "true" to remove tokens containing '遮挡'; default disabled to preserve data
SANITIZE_TEXT="true"              # "true" to normalize text (spaces/hyphens/fullwidth/circled numbers)
STANDARDIZE_LABEL_DESC="true"     # "true" to map empty-like 标签/* to 标签/无法识别
FAIL_FAST="true"                  # "true" to stop immediately on invalid samples, "false" to continue with warnings
LIMIT="-1"                        # Limit number of images to process (e.g., "10" for 10 images, "-1" for all images)

# Performance settings
NUM_WORKERS="8"                   # Number of parallel workers (1=sequential, >1=parallel). Recommended: 4-8 for multi-core systems

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
PYTHON_CMD="/root/miniconda3/envs/ms/bin/python -m data_conversion.pipeline.unified_processor"
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
    
    # Validation step has been removed for simplification
    
    echo "🚀 Ready for training!"
    exit 0
else
    echo ""
    echo "❌ Dataset $DATASET_NAME processing failed"
    echo "📝 Check convert.log for details"
    exit $EXIT_CODE
fi
