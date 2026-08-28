#!/usr/bin/env bash
set -Eeuo pipefail

readonly RUN_DIR="${1:-/workspace/outputs/personal_a2v}"
readonly CHECKPOINT_DIR="${RUN_DIR}/checkpoints"

[[ -d "${CHECKPOINT_DIR}" ]] || {
  echo "[ERROR] Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
}

latest="$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'lora_weights_step_*.safetensors' -print | sort -V | tail -n 1)"
[[ -n "${latest}" ]] || {
  echo "[ERROR] No LoRA checkpoints found in ${CHECKPOINT_DIR}" >&2
  exit 1
}

filename="$(basename "${latest}")"
step="${filename#lora_weights_step_}"
step="${step%.safetensors}"
state="${CHECKPOINT_DIR}/training_state_step_${step}.pt"

echo "Latest candidate in the explicitly selected run:"
echo "  ${latest}"
if [[ -s "${state}" ]]; then
  echo "Matching training state:"
  echo "  ${state}"
  echo
  echo "Review the path above, then resume this exact checkpoint with:"
  printf 'LTX_RESUME_CHECKPOINT=%q train_a2v.sh\n' "${latest}"
else
  echo "[WARN] Matching training state is missing: ${state}"
  echo "Weights can be loaded, but optimizer/scheduler/step state cannot be restored."
  echo "No command was launched. Choose the intended checkpoint explicitly."
fi
