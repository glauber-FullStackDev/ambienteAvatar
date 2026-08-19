#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from huggingface_hub import snapshot_download


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))
TOKEN = os.environ.get("HF_TOKEN") or None

MODELS = {
    "base": (
        "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
        "7d48412d7b903ff3a89f4f5a960d99e1899605a1",
    ),
    "distilled": (
        "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
        "59e4141466bcb1bf9733eca1bc78be6891c9fbdf",
    ),
}

REQUIRED_FILES = (
    "modular_model_index.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer/diffusion_pytorch_model-00001-of-00004.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00004.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00004.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "image_encoder/model.safetensors",
)


def verify(variant: str) -> bool:
    model_dir = MODEL_ROOT / variant
    missing = [
        model_dir / relative
        for relative in REQUIRED_FILES
        if not (model_dir / relative).is_file()
        or (model_dir / relative).stat().st_size == 0
    ]
    print(f"\nVerificacao do Wan-Animate-2 {variant} em {model_dir}:")
    if missing:
        for path in missing:
            print(f"  [FALTA] {path}")
        return False
    print("  [OK] checkpoint Diffusers completo")
    return True


def download(variant: str) -> None:
    repo_id, revision = MODELS[variant]
    target = MODEL_ROOT / variant
    target.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {repo_id}@{revision} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        token=TOKEN,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa o Wan-Animate-2")
    parser.add_argument(
        "--variant",
        choices=tuple(MODELS),
        default=os.environ.get("MODEL_VARIANT", "base").strip().lower(),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.verify_only:
        download(args.variant)
    return 0 if verify(args.variant) else 1


if __name__ == "__main__":
    sys.exit(main())
