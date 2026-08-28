#!/usr/bin/env bash
set -Eeuo pipefail

readonly MODELS_DIR="${LTX_MODELS_DIR:-/workspace/models}"
readonly LTX_DIR="${MODELS_DIR}/ltx-2.3"
readonly LTX_FILE="${LTX_DIR}/ltx-2.3-22b-dev.safetensors"
readonly GEMMA_DIR="${MODELS_DIR}/gemma-3-12b"
readonly GEMMA_MARKER="${GEMMA_DIR}/.download-complete"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

command -v hf >/dev/null 2>&1 || die "Hugging Face CLI ('hf') is not installed."
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is not set. Export a Hugging Face read token at runtime."

mkdir -p "${LTX_DIR}" "${GEMMA_DIR}"

if [[ -s "${LTX_FILE}" ]]; then
  echo "[OK] LTX checkpoint already exists: ${LTX_FILE}"
else
  echo "Downloading Lightricks/LTX-2.3 checkpoint..."
  if ! hf download Lightricks/LTX-2.3 \
      ltx-2.3-22b-dev.safetensors \
      --local-dir "${LTX_DIR}"; then
    die "LTX download failed. Verify HF_TOKEN, network access, free disk space, and acceptance of the Hugging Face model terms."
  fi
fi

[[ -s "${LTX_FILE}" ]] || die "Download reported success but the LTX checkpoint is missing or empty: ${LTX_FILE}"

gemma_is_complete() {
  MODEL_ROOT="${GEMMA_DIR}" /opt/LTX-2/.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_ROOT"])
if not (root / "config.json").is_file():
    raise SystemExit(1)

indexes = list(root.glob("*.safetensors.index.json"))
if indexes:
    for index in indexes:
        data = json.loads(index.read_text())
        shards = set(data.get("weight_map", {}).values())
        if not shards or any(not (root / shard).is_file() or (root / shard).stat().st_size == 0 for shard in shards):
            raise SystemExit(1)
    raise SystemExit(0)

if not any(path.stat().st_size > 0 for path in root.rglob("*.safetensors")):
    raise SystemExit(1)
PY
}

gemma_ready=false
if gemma_is_complete; then
  gemma_ready=true
  touch "${GEMMA_MARKER}"
fi

if [[ "${gemma_ready}" == true ]]; then
  echo "[OK] Gemma model already exists: ${GEMMA_DIR}"
else
  echo "Downloading google/gemma-3-12b-it-qat-q4_0-unquantized..."
  if ! hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
      --local-dir "${GEMMA_DIR}"; then
    die "Gemma download failed. Verify HF_TOKEN and accept the gated Gemma license/model terms on Hugging Face."
  fi
  [[ -s "${GEMMA_DIR}/config.json" ]] || die "Gemma config.json is missing after download."
  find "${GEMMA_DIR}" -type f -name '*.safetensors' -size +0c -print -quit | grep -q . \
    || die "No non-empty Gemma safetensors file was found after download."
  touch "${GEMMA_MARKER}"
fi

echo "[OK] Model downloads verified under ${MODELS_DIR}."
