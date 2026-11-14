#!/usr/bin/env bash
set -euo pipefail

COMFYUI_LOG_PATH="${COMFYUI_LOG_PATH:-/var/log/portal/comfyui.log}"
mkdir -p "$(dirname "${COMFYUI_LOG_PATH}")"
touch "${COMFYUI_LOG_PATH}"

# Keep Docker/Vast stdout and a file that is easy to tail from Jupyter/SSH.
exec > >(tee -a "${COMFYUI_LOG_PATH}") 2>&1
exec python /opt/ltx23-scripts/entrypoint.py "$@"
