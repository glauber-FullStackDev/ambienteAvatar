#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORKSPACE=/workspace

/usr/local/bin/bootstrap_a2v.sh

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
