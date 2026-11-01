#!/bin/sh
# Compatibility launcher for the current EventOwner study.
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$TASK_ROOT/run_eventowner.sh"
