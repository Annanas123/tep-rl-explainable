#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
windows_venv_python="$repo_root/.venv/Scripts/python.exe"
posix_venv_python="$repo_root/.venv/bin/python"

if [[ -x "$windows_venv_python" ]]; then
  python_bin="$windows_venv_python"
elif [[ -x "$posix_venv_python" ]]; then
  python_bin="$posix_venv_python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
else
  python_bin="python"
fi

"$python_bin" "$repo_root/scripts/run_thesis_pipeline.py" "$@"
