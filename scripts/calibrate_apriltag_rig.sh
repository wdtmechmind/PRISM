#!/bin/bash
# Calibrate the relative poses of AprilTags fixed on a rigid body.
# Usage: ./scripts/calibrate_apriltag_rig.sh <task_dir> --tag-size 0.01 [extra args]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: ./scripts/calibrate_apriltag_rig.sh <task_dir> --tag-size <m> [args]"
    echo ""
    echo "Example:"
    echo "  ./scripts/calibrate_apriltag_rig.sh data/raw/task_xxx_calib --tag-size 0.01"
    exit 1
fi

if command -v mamba &> /dev/null; then
    eval "$(mamba shell.bash hook)"
    mamba activate camera
    echo "Activated mamba camera environment"
else
    echo "Warning: mamba not found, using current Python environment"
fi

PYTHONPATH=src python tools/calibrate_apriltag_rig.py "$@"
