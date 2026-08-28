#!/usr/bin/env bash
set -Eeuo pipefail

readonly LTX_ROOT=/opt/LTX-2
readonly TRAINER_ROOT=${LTX_ROOT}/packages/ltx-trainer
readonly CONFIG_PATH="${1:-/workspace/configs/a2v_personal.yaml}"
readonly EFFECTIVE_CONFIG=/workspace/logs/effective_a2v_config.yaml
readonly PREPROCESSED=/workspace/dataset/preprocessed
readonly MODELS_DIR="${LTX_MODELS_DIR:-/workspace/models}"
readonly MODEL="${MODELS_DIR}/ltx-2.3/ltx-2.3-22b-dev.safetensors"
readonly GEMMA="${MODELS_DIR}/gemma-3-12b"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

/usr/local/bin/check_ltx_environment.sh || die "Environment check failed. Fix the blocking items above."

[[ -s "${CONFIG_PATH}" ]] || die "Training config is missing: ${CONFIG_PATH}. Copy the .example file and edit it first."
[[ -s "${MODEL}" ]] || die "LTX checkpoint is missing: ${MODEL}"
[[ -s "${GEMMA}/config.json" ]] || die "Gemma encoder is incomplete: ${GEMMA}"

for required_dir in latents audio_latents conditions; do
  find "${PREPROCESSED}/${required_dir}" -type f -name '*.pt' -size +0c -print -quit 2>/dev/null | grep -q . \
    || die "Preprocessed ${required_dir} are missing under ${PREPROCESSED}. Run preprocess_a2v.sh first."
done

if [[ -n "${LTX_RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "${LTX_RESUME_CHECKPOINT}" && "${LTX_RESUME_CHECKPOINT}" == *.safetensors ]] \
    || die "LTX_RESUME_CHECKPOINT must name one explicit .safetensors file; directories are rejected to avoid resuming the wrong run."
fi

mkdir -p /workspace/logs

SOURCE_CONFIG="${CONFIG_PATH}" EFFECTIVE_CONFIG="${EFFECTIVE_CONFIG}" \
WANDB_PROJECT="${WANDB_PROJECT:-ltx23-personal-a2v}" \
LTX_RESUME_CHECKPOINT="${LTX_RESUME_CHECKPOINT:-}" \
LTX_VALIDATION_INTERVAL="${LTX_VALIDATION_INTERVAL:-}" \
LTX_CHECKPOINT_INTERVAL="${LTX_CHECKPOINT_INTERVAL:-}" \
/opt/LTX-2/.venv/bin/python - <<'PY'
import os
from pathlib import Path
import yaml

source = Path(os.environ["SOURCE_CONFIG"])
target = Path(os.environ["EFFECTIVE_CONFIG"])
config = yaml.safe_load(source.read_text())

strategy = config.get("training_strategy", {})
if strategy.get("name") != "flexible":
    raise SystemExit("training_strategy.name must be 'flexible' for A2V")
if strategy.get("video", {}).get("is_generated") is not True:
    raise SystemExit("A2V safety check failed: training_strategy.video.is_generated must be true")
if strategy.get("audio", {}).get("is_generated") is not False:
    raise SystemExit("A2V safety check failed: training_strategy.audio.is_generated must be false")
if config.get("model", {}).get("training_mode") != "lora":
    raise SystemExit("A2V safety check failed: model.training_mode must be 'lora'")

configured_resume = config.get("model", {}).get("load_checkpoint")
if configured_resume and not Path(configured_resume).is_file():
    raise SystemExit(
        "model.load_checkpoint must name one existing checkpoint file; "
        "directories are rejected to avoid selecting the wrong run"
    )

for sample in config.get("validation", {}).get("samples", []):
    for condition in sample.get("conditions", []):
        if condition.get("type") == "audio_to_video":
            audio = Path(condition.get("audio", ""))
            if not audio.is_file():
                raise SystemExit(f"Validation audio is missing: {audio}")

config.setdefault("wandb", {})["project"] = os.environ["WANDB_PROJECT"]
resume = os.environ.get("LTX_RESUME_CHECKPOINT")
if resume:
    config.setdefault("model", {})["load_checkpoint"] = resume

for env_name, section in (
    ("LTX_VALIDATION_INTERVAL", "validation"),
    ("LTX_CHECKPOINT_INTERVAL", "checkpoints"),
):
    value = os.environ.get(env_name)
    if value:
        if not value.isdigit() or int(value) < 1:
            raise SystemExit(f"{env_name} must be a positive integer")
        config.setdefault(section, {})["interval"] = int(value)

target.write_text(yaml.safe_dump(config, sort_keys=False))
PY

wandb_enabled="$(CONFIG="${EFFECTIVE_CONFIG}" /opt/LTX-2/.venv/bin/python -c 'import os,yaml; print(str(bool(yaml.safe_load(open(os.environ["CONFIG"])).get("wandb",{}).get("enabled"))).lower())')"
if [[ "${wandb_enabled}" == true && -z "${WANDB_API_KEY:-}" ]]; then
  die "W&B is enabled in the config, but WANDB_API_KEY is not set at runtime."
fi

echo "GPU memory:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "Effective training config (${EFFECTIVE_CONFIG}):"
sed -n '1,420p' "${EFFECTIVE_CONFIG}"

cd "${TRAINER_ROOT}"
exec uv run --frozen --no-sync python scripts/train.py "${EFFECTIVE_CONFIG}"
