#!/usr/bin/env python3
"""Download only the LTX 2.5 files used by the IA2V workflow."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_HOME = Path("/opt/ltx23-scripts")
sys.path.insert(0, str(SCRIPTS_HOME))
import download_models  # noqa: E402


REQUIRED_PATHS = {
    "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "text_encoders/gemma4_e2b_it_int8_convrot.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}


def main() -> None:
    download_models.MODEL_FILES = tuple(
        model for model in download_models.MODEL_FILES if model.relative_path in REQUIRED_PATHS
    )
    missing = REQUIRED_PATHS - {model.relative_path for model in download_models.MODEL_FILES}
    if missing:
        raise SystemExit(f"Modelos LTX 2.5 ausentes na lista: {sorted(missing)}")
    download_models.main()


if __name__ == "__main__":
    main()
