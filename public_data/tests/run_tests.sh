#!/bin/bash
# Test runner for LVIS converter
# Always uses 'ms' conda environment

set -e

echo "Running tests in conda environment: ms"
echo "======================================"

cd /data/public_data

# Run converter tests
echo -e "\n[1/1] Running LVIS Converter Tests..."
conda run -n ms python tests/test_lvis_converter.py

echo -e "\n======================================"
echo "All tests completed!"
echo "======================================"

