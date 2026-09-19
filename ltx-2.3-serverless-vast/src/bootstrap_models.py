#!/usr/bin/env python3
"""Download only the LTX 2.3 files used by the Personal LoRA IA2V workflow."""
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
