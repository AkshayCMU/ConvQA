#!/bin/bash
# HW6 RAG System - Run Script
# Executes the output analysis and generates visualizations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "HW6 RAG System - Analysis Runner"
echo "=============================================="

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "[ERROR] Python not found. Please install Python 3.8+"
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON --version)"

# Install dependencies if needed
echo "[INFO] Checking dependencies..."
$PYTHON -m pip install matplotlib numpy --quiet 2>/dev/null || true

# Run analysis
echo "[INFO] Running output analysis..."
$PYTHON src/output_analysis.py

echo ""
echo "=============================================="
echo "Analysis complete. Check output/analysis/"
echo "=============================================="

# Open images on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "[INFO] Opening generated charts..."
    open output/analysis/*.png 2>/dev/null || true
fi
