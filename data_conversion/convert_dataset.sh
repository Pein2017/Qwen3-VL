#!/bin/bash

# Data Conversion Pipeline - Manual Configuration
# 
# Processes raw dataset directories into training-ready format:
# /data/{dataset_name}/
#   ├── images/*.jpeg        # Smart-resized images  
#   ├── all_samples.jsonl    # All processed samples
#   ├── train.jsonl          # Training split
#   ├── val.jsonl            # Validation split  
#   ├── teacher.jsonl        # Teacher samples
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
DATASET_NAME="bbu_full_768_poly"         # e.g., "experiment_1" or leave empty to auto-detect

# Optional configuration files - SET THESE IF YOU HAVE THEM
HIERARCHY_FILE=""               # e.g., "data_conversion/label_hierarchy.json" or leave empty

# Processing parameters - YOU MUST SET THESE
VAL_RATIO="0.2"                    # e.g., "0.1" for 10% validation split
MAX_TEACHERS="0"                 # e.g., "10" for max teacher samples, or "0" to disable teacher-pool (for dynamic teacher-sampling)
RESIZE="true"                       # "true" or "false" for image resizing
OBJECT_TYPES="full"               # e.g., "bbu label" or "fiber wire" (space-separated, arbitrary combinations), or "full" for all types

# Optional settings
LOG_LEVEL="INFO"                    # e.g., "INFO", "DEBUG", "WARNING", "ERROR" or leave empty
SEED="17"                         # e.g., "17" or leave empty
STRIP_OCCLUSION="true"            # "true" to remove tokens containing '遮挡'; default disabled to preserve data
SANITIZE_TEXT="true"              # "true" to normalize text (spaces/hyphens/fullwidth/circled numbers)
STANDARDIZE_LABEL_DESC="true"     # "true" to map empty-like 标签/* to 标签/无法识别

# Validation settings - OPTIONAL (currently hardcoded in unified_processor.py)
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
echo "   Language: Chinese (default)"
echo "   Hierarchy File: ${HIERARCHY_FILE:-'(not set)'}"
echo "   Val Ratio: $VAL_RATIO"
if [ "$MAX_TEACHERS" = "0" ]; then
    echo "   Max Teachers: $MAX_TEACHERS (teacher-pool disabled for dynamic sampling)"
else
    echo "   Max Teachers: $MAX_TEACHERS"
fi
echo "   Smart Resize: $RESIZE"
echo "   Object Types: $OBJECT_TYPES"
echo "   Log Level: $LOG_LEVEL"
echo "   Seed: $SEED"
echo "   Validation Mode: $VALIDATION_MODE (hardcoded: strict)"
echo "   Min Object Size: $MIN_OBJECT_SIZE (hardcoded: 10px)"
echo "   Validation Reports: Always enabled"
echo ""

# Build command arguments
PYTHON_CMD="/root/miniconda3/envs/ms/bin/python data_conversion/unified_processor.py"
ARGS="--input_dir \"$INPUT_DIR\""
ARGS="$ARGS --output_dir \"$OUTPUT_DIR\""
ARGS="$ARGS --language chinese"
ARGS="$ARGS --dataset_name \"$DATASET_NAME\""
ARGS="$ARGS --val_ratio \"$VAL_RATIO\""
ARGS="$ARGS --max_teachers \"$MAX_TEACHERS\""
ARGS="$ARGS --seed \"$SEED\""
ARGS="$ARGS --log_level \"$LOG_LEVEL\""
ARGS="$ARGS --object_types $OBJECT_TYPES"

# Add optional arguments if provided
if [ -n "$HIERARCHY_FILE" ]; then
    ARGS="$ARGS --hierarchy_path \"$HIERARCHY_FILE\""
fi


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

echo "🔄 Processing dataset: $DATASET_NAME ($INPUT_DIR)"
echo "  └─ Executing: $PYTHON_CMD"

# Execute the command
eval "$PYTHON_CMD $ARGS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dataset $DATASET_NAME processed successfully!"
    echo "📁 Output: $OUTPUT_DIR/$DATASET_NAME/"
    
    # Validation step has been removed for simplification
    
    echo "🚀 Ready for training!"
else
    echo ""
    echo "❌ Dataset $DATASET_NAME processing failed"
    exit 1
fi