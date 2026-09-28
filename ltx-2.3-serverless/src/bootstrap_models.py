#!/usr/bin/env python3
"""Download only the LTX 2.3 files used by the Personal LoRA IA2V workflow.

This image is pinned to the LTX 2.3 dev FP8 checkpoint: the bootstrap
downloads the 2.3 model set (checkpoint, Gemma 3 12B FP4 text encoder,
distilled dynamic LoRA, prompt-enhancement LoRA and spatial upscaler) and
never touches the LTX 2.5 variants.  The 2.5 experiment lives in
../ltx-2.3-serverless-bf16/ with its own image; rolling back to the 2.5
workflow requires the previous :v2.5 image tags still published in GHCR.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_HOME = Path("/opt/ltx23-scripts")
sys.path.insert(0, str(SCRIPTS_HOME))
import download_models  # noqa: E402


REQUIRED_PATHS = {
    "checkpoints/ltx-2.3-22b-dev-fp8.safetensors",
    "text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
    "loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
    "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}


def main() -> None:
    download_models.MODEL_FILES = tuple(
        model for model in download_models.MODEL_FILES if model.relative_path in REQUIRED_PATHS
    )
    missing = REQUIRED_PATHS - {model.relative_path for model in download_models.MODEL_FILES}
    if missing:
        raise SystemExit(f"Modelos LTX 2.3 ausentes na lista: {sorted(missing)}")
    download_models.main()


if __name__ == "__main__":
    main()
