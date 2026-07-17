#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MVS_ROOT="${MVS_ROOT:-/opt/MVS}"
PYTHON_BIN="${PRISM_PYTHON:-/home/daotan/miniforge3/envs/camera/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

set +u
source "$MVS_ROOT/bin/set_env_path.sh" "$MVS_ROOT"
set -u

export PRISM_LEGACY_RECORDING_DIR="${PRISM_LEGACY_RECORDING_DIR:-$MVS_ROOT/Samples/64/Python/General/Recording}"
export PRISM_MVIMPORT_DIR="${PRISM_MVIMPORT_DIR:-$MVS_ROOT/Samples/64/Python/MvImport}"
export PYTHONPATH="$ROOT_DIR/src:$PRISM_MVIMPORT_DIR:$PRISM_LEGACY_RECORDING_DIR:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -m prism.cli.collect "$@"
