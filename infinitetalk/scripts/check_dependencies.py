#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


required_modules = (
    "accelerate",
    "diffusers",
    "einops",
    "gguf",
    "huggingface_hub",
    "peft",
    "pyloudnorm",
    "rotary_embedding_torch",
    "sentencepiece",
)

missing_modules = [name for name in required_modules if find_spec(name) is None]
if missing_modules:
    raise SystemExit(f"Dependencias Python ausentes: {', '.join(missing_modules)}")

comfyui_home = Path("/opt/ComfyUI")
required_paths = (
    comfyui_home / "main.py",
    comfyui_home / "custom_nodes/ComfyUI-WanVideoWrapper/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-KJNodes/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-MelBandRoFormer/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-VideoHelperSuite/__init__.py",
    Path("/opt/defaults/workflows/infinitetalk-i2v.json"),
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise SystemExit(f"Arquivos da imagem ausentes: {', '.join(missing_paths)}")

print("OK: dependencias e custom nodes do InfiniteTalk presentes")
