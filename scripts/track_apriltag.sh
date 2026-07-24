#!/bin/bash
# Track a rigid body's 6DOF trajectory from AprilTags using a calibrated rig.
# Usage: ./scripts/track_apriltag.sh <task_dir> [--rig configs/devices/apriltag_rig.json] [args]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: ./scripts/track_apriltag.sh <task_dir> [--rig <rig.json>] [args]"
    echo ""
    echo "Example:"
    echo "  ./scripts/track_apriltag.sh data/raw/task_xxx --rig configs/devices/apriltag_rig.json"
    exit 1
fi

if command -v mamba &> /dev/null; then
    eval "$(mamba shell.bash hook)"
    mamba activate camera
    echo "Activated mamba camera environment"
else
    echo "Warning: mamba not found, using current Python environment"
fi

PYTHONPATH=src python tools/track_apriltag_trajectory.py "$@"
