#!/bin/bash
# A fresh, reviewed PoseSplit reproduction. No cloud or native model calls.
set -euo pipefail
posesplit_project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$posesplit_project"
if ! command -v python3 >/dev/null 2>&1; then
  printf 'Python 3 is required. Install Python, then run this file again.\n' >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  printf 'FFmpeg is required. On this Mac, install it with: brew install ffmpeg\n' >&2
  exit 1
fi
posesplit_python="$posesplit_project/.venv-pilots/bin/python"
if [ ! -x "$posesplit_python" ]; then
  python3 -m venv "$posesplit_project/.venv-pilots"
fi
# Install missing Python packages only, inside the project environment.
while IFS= read -r posesplit_requirement; do
  [ -z "$posesplit_requirement" ] || "$posesplit_python" -m pip install "$posesplit_requirement"
done < <("$posesplit_python" - <<'PY'
import importlib.util
for module, requirement in [('numpy','numpy==2.5.4'),('PIL','pillow==12.3.0'),('fsspec','fsspec==2026.9.0'),('aiohttp','aiohttp==3.14.5')]:
    if importlib.util.find_spec(module) is None:
        print(requirement)
PY
)
posesplit_run_id="posesplit-replay-$(date -u +%Y%m%d-%H%M%S)-$$"
posesplit_out="$posesplit_project/runs/$posesplit_run_id"
posesplit_logs="$posesplit_project/work/posesplit_launcher_logs"
mkdir -p "$posesplit_logs"
posesplit_log="$posesplit_logs/$posesplit_run_id.log"
printf 'PoseSplit creates a new run at:\n%s\nLog:\n%s\n' "$posesplit_out" "$posesplit_log"
posesplit_cache="${POSESPLIT_SOURCE_CACHE:-$posesplit_project/runs/renewed-09-posesplit-20261031}"
"$posesplit_python" -m pilots.renewed09.reproduce --out "$posesplit_out" --cache "$posesplit_cache" 2>&1 | tee "$posesplit_log"
printf '\nComplete. Open result.json in the run folder above.\n'
