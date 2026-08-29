#!/usr/bin/env bash
# Prepare the persistent workspace and, when requested, download the gated
# weights.  This script is intentionally separate from start.sh so Vast.ai's
# managed Jupyter mode can run it without launching a second Jupyter server.
set -Eeuo pipefail

/usr/local/bin/init_workspace.sh

# DOWNLOAD_MODELS_ON_START is the Vast-facing name, matching the existing
# LTX 2.3 image.  AUTO_DOWNLOAD_MODELS remains supported for compatibility.
download_models="${DOWNLOAD_MODELS_ON_START:-${AUTO_DOWNLOAD_MODELS:-0}}"

case "${download_models}" in
  1|true|TRUE|yes|YES)
    echo "Automatic model download is enabled; downloading/verifying LTX-2.3 and Gemma."
    /usr/local/bin/download_ltx_models.sh
    ;;
  0|false|FALSE|no|NO|"")
    echo "Automatic model download is disabled. Run download_ltx_models.sh when ready."
    ;;
  *)
    echo "[ERROR] DOWNLOAD_MODELS_ON_START must be 0 or 1." >&2
    exit 1
    ;;
esac
