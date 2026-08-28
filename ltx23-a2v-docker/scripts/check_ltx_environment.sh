#!/usr/bin/env bash
set -uo pipefail

critical_failures=0
warnings=0
readonly LTX_ROOT=/opt/LTX-2
readonly PYTHON=${LTX_ROOT}/.venv/bin/python
readonly MODELS_DIR="${LTX_MODELS_DIR:-/workspace/models}"
readonly LTX_MODEL="${MODELS_DIR}/ltx-2.3/ltx-2.3-22b-dev.safetensors"
readonly GEMMA_DIR="${MODELS_DIR}/gemma-3-12b"
readonly RAW_DATASET=/workspace/dataset/raw/dataset.json

ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; warnings=$((warnings + 1)); }
fail() { echo "[FAIL] $*"; critical_failures=$((critical_failures + 1)); }

echo "=== LTX-2.3 A2V environment report ==="

if [[ "$(uname -s 2>/dev/null)" == Linux ]]; then ok "Linux"; else fail "Linux is required"; fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if gpu_line="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1)" && [[ -n "${gpu_line}" ]]; then
    ok "NVIDIA GPU: ${gpu_line}"
  else
    fail "nvidia-smi cannot see an NVIDIA GPU (use docker run --gpus all)"
  fi
else
  fail "nvidia-smi is not installed"
fi

if [[ -x "${PYTHON}" ]]; then
  torch_report="$(${PYTHON} - <<'PY' 2>&1
import torch
print(f"available={torch.cuda.is_available()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"vram_gib={props.total_memory / 1024**3:.2f}")
PY
)"
  while IFS= read -r report_line; do
    printf '       %s\n' "${report_line}"
  done <<<"${torch_report}"
  if grep -q '^available=True$' <<<"${torch_report}"; then
    ok "CUDA available to PyTorch"
  else
    fail "torch.cuda.is_available() is false"
  fi
else
  fail "LTX Python environment is missing: ${PYTHON}"
fi

if [[ -d "${LTX_ROOT}/packages/ltx-trainer" ]]; then
  actual_commit="$(git -C "${LTX_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "${LTX_COMMIT:-}" && "${actual_commit}" != "${LTX_COMMIT}" ]]; then
    fail "LTX repository commit mismatch: expected ${LTX_COMMIT}, got ${actual_commit:-unknown}"
  else
    ok "LTX repository (${actual_commit:-commit unavailable})"
  fi
else
  fail "LTX repository is missing at ${LTX_ROOT}"
fi

if command -v uv >/dev/null 2>&1; then ok "uv $(uv --version | awk '{print $2}')"; else fail "uv is unavailable"; fi
if command -v ffmpeg >/dev/null 2>&1; then ok "FFmpeg $(ffmpeg -version 2>/dev/null | head -n1)"; else fail "FFmpeg is unavailable"; fi

if [[ -s "${LTX_MODEL}" ]]; then ok "LTX checkpoint: ${LTX_MODEL}"; else fail "LTX checkpoint missing: ${LTX_MODEL}"; fi

if [[ -s "${GEMMA_DIR}/config.json" ]] \
  && find "${GEMMA_DIR}" -type f -name '*.safetensors' -size +0c -print -quit 2>/dev/null | grep -q .; then
  ok "Gemma encoder: ${GEMMA_DIR}"
else
  fail "Gemma encoder is incomplete: ${GEMMA_DIR}"
fi

if [[ -s "${RAW_DATASET}" ]] && find /workspace/dataset/raw/videos -type f -name '*.mp4' -print -quit 2>/dev/null | grep -q .; then
  ok "Raw dataset uploaded"
else
  warn "Dataset not uploaded yet (expected ${RAW_DATASET} and raw/videos/*.mp4)"
fi

echo "Summary: ${critical_failures} blocking failure(s), ${warnings} warning(s)."
(( critical_failures == 0 ))
