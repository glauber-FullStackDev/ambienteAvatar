#!/usr/bin/env python3
"""Download the GGUF 16 GB model set used by the IA2V API workflow (~26 GiB).

Reuses the pinned downloader from ../ltx-2.3-16gb/scripts/download_models.py
(repositorios unsloth + Gemma QAT, tudo publico):

- DiT LTX-2.3-22B-distilled-1.1 em GGUF (LTX23_GGUF_QUANT, padrao Q4_K_M)
  -> models/diffusion_models;
- Gemma 3 12B QAT GGUF (UD-Q4_K_XL) + embeddings connectors -> models/text_encoders;
- video VAE e audio VAE standalone -> models/vae;
- spatial upscaler x2 1.1 -> models/latent_upscale_models;
- LoRA de prompt-enhancement do Gemma -> models/loras.

O LoRA pessoal glauberavatar.safetensors nao e baixado: vem incorporado na
imagem e o handler o instala no volume (_ensure_personal_lora).
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_HOME = Path("/opt/ltx23-scripts")
sys.path.insert(0, str(SCRIPTS_HOME))
import download_models  # noqa: E402


def main() -> None:
    download_models.main()


if __name__ == "__main__":
    main()
