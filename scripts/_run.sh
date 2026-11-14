#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
DEFAULT_PYTHON=python
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi
exec "${TUTORIAL_PYTHON:-$DEFAULT_PYTHON}" "$PROJECT_ROOT/tools/tutorial.py" "$@"
