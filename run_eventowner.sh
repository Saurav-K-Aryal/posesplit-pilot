#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TASK_ROOT"
TASK_PYTHON=${PYTHON:-python3}
TASK_OUTPUT=${OUTPUT:-reproductions/eventowner-$(date +%Y%m%d-%H%M%S)-$$}
if [ -e "$TASK_OUTPUT" ]; then
  echo 'Choose a new OUTPUT directory; existing results are preserved.' >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then "$TASK_PYTHON" -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py --output "$TASK_OUTPUT"
