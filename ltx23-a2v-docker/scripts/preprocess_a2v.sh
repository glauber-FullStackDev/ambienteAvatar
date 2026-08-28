#!/usr/bin/env bash
set -Eeuo pipefail

readonly LTX_ROOT=/opt/LTX-2
readonly TRAINER_ROOT=${LTX_ROOT}/packages/ltx-trainer
readonly DATASET=/workspace/dataset/raw/dataset.json
readonly OUTPUT=/workspace/dataset/preprocessed
readonly MODELS_DIR="${LTX_MODELS_DIR:-/workspace/models}"
readonly MODEL="${MODELS_DIR}/ltx-2.3/ltx-2.3-22b-dev.safetensors"
readonly GEMMA="${MODELS_DIR}/gemma-3-12b"
readonly FRAME_COUNT="${LTX_FRAME_COUNT:-89}"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ -s "${DATASET}" ]] || die "Dataset metadata is missing: ${DATASET}"
[[ -s "${MODEL}" ]] || die "LTX checkpoint is missing: ${MODEL}. Run download_ltx_models.sh first."
[[ -s "${GEMMA}/config.json" ]] || die "Gemma encoder is incomplete: ${GEMMA}. Run download_ltx_models.sh first."
find "${GEMMA}" -type f -name '*.safetensors' -size +0c -print -quit | grep -q . \
  || die "No non-empty Gemma safetensors file found under ${GEMMA}."
[[ -n "${LTX_RESOLUTION_BUCKET:-}" ]] \
  || die "Set LTX_RESOLUTION_BUCKET in frames x height x width order, for example: export LTX_RESOLUTION_BUCKET=89x448x768"
[[ "${FRAME_COUNT}" =~ ^[0-9]+$ ]] || die "LTX_FRAME_COUNT must be an integer."

# The task-facing format is FxHxW (for example 89x544x960). The current
# official Typer CLI expects WxHxF, so translate and validate every bucket.
IFS=';' read -r -a requested_buckets <<<"${LTX_RESOLUTION_BUCKET}"
official_buckets=()
for requested in "${requested_buckets[@]}"; do
  IFS='xX' read -r frames height width extra <<<"${requested}"
  [[ -n "${frames:-}" && -n "${height:-}" && -n "${width:-}" && -z "${extra:-}" ]] \
    || die "Invalid bucket '${requested}'. Expected FxHxW; multiple buckets use semicolons."
  [[ "${frames}" =~ ^[0-9]+$ && "${height}" =~ ^[0-9]+$ && "${width}" =~ ^[0-9]+$ ]] \
    || die "Bucket '${requested}' must contain only positive integers."
  (( frames == FRAME_COUNT )) \
    || die "Bucket '${requested}' uses ${frames} frames but LTX_FRAME_COUNT=${FRAME_COUNT}."
  (( frames % 8 == 1 )) \
    || die "Frame count ${frames} is invalid for the default LTX temporal VAE factor (frames % 8 must equal 1)."
  (( height % 32 == 0 && width % 32 == 0 )) \
    || die "Height and width must be divisible by 32; got ${height}x${width}."
  official_buckets+=("${width}x${height}x${frames}")
done

official_resolution="$(IFS=';'; echo "${official_buckets[*]}")"
mkdir -p "${OUTPUT}"

command=(
  uv run --frozen --no-sync python scripts/process_dataset.py
  "${DATASET}"
  --resolution-buckets "${official_resolution}"
  --model-path "${MODEL}"
  --text-encoder-path "${GEMMA}"
  --output-dir "${OUTPUT}"
  --device cuda
  --batch-size "${LTX_PREPROCESS_BATCH_SIZE:-1}"
)

if [[ "${LTX_VAE_TILING:-0}" == 1 ]]; then
  command+=(--vae-tiling)
fi
if [[ "${LTX_PREPROCESS_OVERWRITE:-0}" == 1 ]]; then
  command+=(--overwrite)
fi

echo "Input bucket(s), FxHxW: ${LTX_RESOLUTION_BUCKET}"
echo "Official CLI bucket(s), WxHxF: ${official_resolution}"
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${TRAINER_ROOT}"
"${command[@]}"

for required_dir in latents audio_latents conditions; do
  if ! find "${OUTPUT}/${required_dir}" -type f -name '*.pt' -size +0c -print -quit 2>/dev/null | grep -q .; then
    die "Preprocessing completed without non-empty .pt files in ${OUTPUT}/${required_dir}."
  fi
done

echo "[OK] Video latents, embedded-audio latents, and text conditions are ready in ${OUTPUT}."
