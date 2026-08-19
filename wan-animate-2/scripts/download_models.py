#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download


MODEL_ROOT = Path(os.environ.get("COMFYUI_MODELS", "/models"))
TOKEN = os.environ.get("HF_TOKEN") or None

MODEL_REPO = "Comfy-Org/Wan-Animate-2"
MODEL_REVISION = "ed158470869ff31fa51cf56012dac33fb00f494b"

# Os tamanhos vem dos metadados LFS da revisao fixada. A verificacao impede que
# um download interrompido seja aceito apenas porque o arquivo existe.
MODEL_FILES = (
    (
        "diffusion_models/wan_animate_2_int8_convrot.safetensors",
        16_653_175_528,
        "Wan-Animate-2 Base INT8 ConvRot",
    ),
    (
        "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        738_005_744,
        "Lightx2v LoRA oficial",
    ),
    (
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        6_735_906_897,
        "UMT5 XXL FP8",
    ),
    (
        "clip_vision/clip_vision_h.safetensors",
        1_264_219_396,
        "CLIP Vision H",
    ),
    (
        "vae/Wan2_1_VAE_bf16.safetensors",
        253_806_278,
        "Wan 2.1 VAE BF16",
    ),
)


def download() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for remote_path, _expected_size, label in MODEL_FILES:
        target = MODEL_ROOT / remote_path
        if target.is_file() and target.stat().st_size == _expected_size:
            print(f"OK existente: {label}: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando {label}: {MODEL_REPO}/{remote_path} -> {target}")
        hf_hub_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            filename=remote_path,
            local_dir=MODEL_ROOT,
            token=TOKEN,
        )


def verify() -> bool:
    failures: list[Path] = []
    total_size = 0
    print(f"\nVerificacao dos modelos Wan-Animate-2 INT8 em {MODEL_ROOT}:")
    for remote_path, expected_size, label in MODEL_FILES:
        target = MODEL_ROOT / remote_path
        actual_size = target.stat().st_size if target.is_file() else 0
        if actual_size == expected_size:
            print(f"  [OK] {label}: {target}")
            total_size += actual_size
        else:
            print(
                f"  [FALTA/INVALIDO] {label}: {target} "
                f"({actual_size} de {expected_size} bytes)"
            )
            failures.append(target)
    if failures:
        print(f"\nFaltam ou estao incompletos {len(failures)} arquivo(s).")
        return False
    print(f"\nConjunto completo: {total_size / 1_000_000_000:.2f} GB.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa o Wan-Animate-2 INT8 ConvRot para o ComfyUI"
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.verify_only:
        download()
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
