#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


required_modules = (
    "accelerate",
    "av",
    "decord",
    "diffusers",
    "flash_attn",
    "gradio",
    "huggingface_hub",
    "torch",
    "transformers",
)
missing_modules = [name for name in required_modules if find_spec(name) is None]
if missing_modules:
    raise SystemExit(f"Dependencias Python ausentes: {', '.join(missing_modules)}")

required_paths = (
    Path("/opt/wan-animate-2/app.py"),
    Path("/opt/wan-animate-2/scripts/download_models.py"),
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise SystemExit(f"Arquivos da imagem ausentes: {', '.join(missing_paths)}")

from diffusers import ModularPipeline, WanAnimate2Transformer3DModel  # noqa: E402

if ModularPipeline is None or WanAnimate2Transformer3DModel is None:
    raise SystemExit("Integracao Wan-Animate-2 do Diffusers indisponivel")

print("OK: runtime Diffusers do Wan-Animate-2 presente")
