#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORKSPACE=/workspace
readonly TEMPLATE=/opt/ltx23-a2v/configs/a2v_personal.yaml.example

mkdir -p \
  "${WORKSPACE}/models/ltx-2.3" \
  "${WORKSPACE}/models/gemma-3-12b" \
  "${WORKSPACE}/dataset/raw/videos" \
  "${WORKSPACE}/dataset/preprocessed/latents" \
  "${WORKSPACE}/dataset/preprocessed/audio_latents" \
  "${WORKSPACE}/dataset/preprocessed/conditions" \
  "${WORKSPACE}/validation/audio" \
  "${WORKSPACE}/validation/reference" \
  "${WORKSPACE}/configs" \
  "${WORKSPACE}/outputs" \
  "${WORKSPACE}/notebooks" \
  "${WORKSPACE}/logs" \
  "${HF_HOME:-${WORKSPACE}/.cache/huggingface}" \
  "${WANDB_DIR:-${WORKSPACE}/logs/wandb}"

if [[ ! -e "${WORKSPACE}/configs/a2v_personal.yaml.example" ]]; then
  install -m 0644 "${TEMPLATE}" "${WORKSPACE}/configs/a2v_personal.yaml.example"
fi

case "${AUTO_DOWNLOAD_MODELS:-0}" in
  1|true|TRUE|yes|YES)
    echo "AUTO_DOWNLOAD_MODELS is enabled; downloading/verifying LTX-2.3 and Gemma before Jupyter starts."
    /usr/local/bin/download_ltx_models.sh
    ;;
  0|false|FALSE|no|NO|"")
    ;;
  *)
    echo "[ERROR] AUTO_DOWNLOAD_MODELS must be 0 or 1." >&2
    exit 1
    ;;
esac

echo "=== LTX-2.3 A2V container ==="
echo "LTX commit: ${LTX_COMMIT:-unknown}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "[WARN] nvidia-smi is installed but no GPU is currently visible."
else
  echo "[WARN] nvidia-smi is unavailable. Start the container with --gpus all."
fi

/opt/LTX-2/.venv/bin/python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA visible to PyTorch: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

df -h "${WORKSPACE}"

if (( $# > 0 )); then
  exec "$@"
fi

if [[ -n "${JUPYTER_TOKEN:-}" ]]; then
  token="${JUPYTER_TOKEN}"
  echo "Using JUPYTER_TOKEN supplied at runtime (value hidden)."
else
  token="$(/opt/LTX-2/.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo "Generated one-time Jupyter token: ${token}"
fi

exec /opt/LTX-2/.venv/bin/jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.root_dir="${WORKSPACE}" \
  --IdentityProvider.token="${token}"
