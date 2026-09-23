#!/usr/bin/env python3
"""Download only the LTX 2.5 files used by the IA2V workflow.

The DiT checkpoint follows WORKFLOW_PATH: the default (and any int8
template) downloads the INT8 ConvRoT transformer, while a bf16 template
downloads the BF16 transformer.  Set LTX25_UNET_CHECKPOINT to "int8" or
"bf16" to override the detection explicitly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS_HOME = Path("/opt/ltx23-scripts")
sys.path.insert(0, str(SCRIPTS_HOME))
import download_models  # noqa: E402

INT8_UNET = "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
BF16_UNET = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"

BASE_REQUIRED_PATHS = {
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "text_encoders/gemma4_e2b_it_int8_convrot.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

UNET_CHOICES = {"int8": INT8_UNET, "bf16": BF16_UNET}


def resolve_unet_checkpoint() -> str:
    override = os.environ.get("LTX25_UNET_CHECKPOINT", "").strip().lower()
    if override:
        if override not in UNET_CHOICES:
            raise SystemExit(
                f"LTX25_UNET_CHECKPOINT invalido: {override!r}; use int8 ou bf16"
            )
        return UNET_CHOICES[override]
    workflow_path = os.environ.get(
        "WORKFLOW_PATH", "/opt/defaults/workflows/video_ltx2_5_ia2v_api.json"
    )
    return BF16_UNET if "bf16" in Path(workflow_path).name else INT8_UNET


def required_paths() -> set[str]:
    return BASE_REQUIRED_PATHS | {resolve_unet_checkpoint()}


def main() -> None:
    selected = required_paths()
    download_models.MODEL_FILES = tuple(
        model for model in download_models.MODEL_FILES if model.relative_path in selected
    )
    missing = selected - {model.relative_path for model in download_models.MODEL_FILES}
    if missing:
        raise SystemExit(f"Modelos LTX 2.5 ausentes na lista: {sorted(missing)}")
    download_models.main()


if __name__ == "__main__":
    main()
