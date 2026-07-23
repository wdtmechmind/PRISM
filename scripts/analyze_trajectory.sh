#!/bin/bash

# Script to analyze trajectory files after collection
# Usage: ./scripts/analyze_trajectory.sh <task_directory>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: ./scripts/analyze_trajectory.sh <task_directory> [output_directory]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/analyze_trajectory.sh data/raw/task_20260723_120000_grasp-demo"
    echo "  ./scripts/analyze_trajectory.sh data/raw/task_20260723_120000_grasp-demo --output-dir ./analysis"
    exit 1
fi

# Activate camera environment
if command -v mamba &> /dev/null; then
    eval "$(mamba shell.bash hook)"
    mamba activate camera
    echo "Activated mamba camera environment"
else
    echo "Warning: mamba not found, using current Python environment"
fi

PYTHONPATH=src python -m prism.processing.trajectory_analyzer "$@"
