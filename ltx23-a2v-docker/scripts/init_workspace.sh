#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORKSPACE=/workspace
readonly MODELS_DIR="${LTX_MODELS_DIR:-${WORKSPACE}/models}"
readonly TEMPLATE=/opt/ltx23-a2v/configs/a2v_personal.yaml.example

mkdir -p \
  "${MODELS_DIR}/ltx-2.3" \
  "${MODELS_DIR}/gemma-3-12b" \
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
