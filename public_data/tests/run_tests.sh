#!/bin/bash
# Test runner for LVIS converter
# Always uses 'ms' conda environment

set -e

echo "Running tests in conda environment: ms"
echo "======================================"

cd /data/Qwen3-VL/public_data

# Run converter tests
echo -e "\n[1/2] Running LVIS Converter Tests..."
conda run -n ms python tests/test_lvis_converter.py

# Run polygon cap smoke tests
echo -e "\n[2/2] Running Polygon Cap Smoke Tests..."
conda run -n ms python tests/test_poly_cap.py

echo -e "\n======================================"
echo "All tests completed!"
echo "======================================"

