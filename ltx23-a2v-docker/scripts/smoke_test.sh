#!/usr/bin/env bash
set -Eeuo pipefail

echo "=== NVIDIA ==="
nvidia-smi

echo "=== PyTorch and imports ==="
/opt/LTX-2/.venv/bin/python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not available to PyTorch")

print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory)

import ltx_core
import ltx_trainer
import wandb

print("ltx_core import: OK")
print("ltx_trainer import: OK")
print("wandb import: OK")
PY

echo "=== FFmpeg ==="
ffmpeg -version | head -n 1
echo "[OK] Smoke test passed. No models were downloaded."
