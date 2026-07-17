#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PRISM_PYTHON:-/home/daotan/miniforge3/envs/camera/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <trial_or_cameras_or_segment_dir> [extra args...]"
  echo "example: $0 data/raw/task_YYYYmmdd_HHMMSS_name/trial_000001 --time-range overlap --include-rs y"
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
exec "$PYTHON_BIN" -m prism.processing.offline_rebuild "$@"