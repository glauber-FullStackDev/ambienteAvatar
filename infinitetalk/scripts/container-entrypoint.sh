#!/usr/bin/env bash
set -euo pipefail

COMFYUI_LOG_PATH="${COMFYUI_LOG_PATH:-/var/log/portal/comfyui.log}"
mkdir -p "$(dirname "${COMFYUI_LOG_PATH}")"
touch "${COMFYUI_LOG_PATH}"

# Preserve Docker/Vast stdout while also exposing a file that can be tailed
# from the terminal inside the instance. Python remains PID 1 after exec.
exec > >(tee -a "${COMFYUI_LOG_PATH}") 2>&1
exec python /opt/infinitetalk-scripts/entrypoint.py "$@"
