#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download


MODELS = Path(os.environ.get("COMFYUI_MODELS", "/opt/ComfyUI/models"))
DOWNLOADS = MODELS / ".downloads"
TOKEN = os.environ.get("HF_TOKEN") or None

ANIMATE_REPO = "Kijai/WanVideo_comfy_fp8_scaled"
ANIMATE_REVISION = "033a4e487f60220b3d6e469599a6aebc46e13cee"
WAN_REPO = "Kijai/WanVideo_comfy"
WAN_REVISION = "8260d429d19fd7a72304cad059160b95d843913f"
COMFY_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
COMFY_REVISION = "617a7633e636506f850e043bc4605f290a466a8e"
WAN_OFFICIAL_REPO = "Wan-AI/Wan2.2-Animate-14B"
WAN_OFFICIAL_REVISION = "cb93a225fbaf1ca100f54e79da8f994995b689b3"
VITPOSE_REPO = "JunkyByte/easy_ViTPose"
VITPOSE_REVISION = "e83805274e89428969355ec4afffcbc413e79188"
SAM2_REPO = "Kijai/sam2-safetensors"
SAM2_REVISION = "f885607d88bb3f9145efa49c3e3c50a9e5bf13eb"

ANIMATE_MODELS = {
    "e4m3fn": "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
    "e5m2": "Wan2_2-Animate-14B_fp8_e5m2_scaled_KJ.safetensors",
}


def download_as(repo: str, revision: str, remote_path: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"OK existente: {target}")
        return
    print(f"Baixando {repo}/{remote_path} -> {target}")
    local_root = DOWNLOADS / repo.replace("/", "--")
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            revision=revision,
            filename=remote_path,
            local_dir=local_root,
            token=TOKEN,
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded), str(target))


def model_files(variant: str) -> list[tuple[str, str, str, Path, str]]:
    animate_name = ANIMATE_MODELS[variant]
    return [
        (
            ANIMATE_REPO,
            ANIMATE_REVISION,
            f"Wan22Animate/{animate_name}",
            MODELS / "diffusion_models/WanVideo/2_2" / animate_name,
            "Wan 2.2 Animate 14B FP8",
        ),
        (
            WAN_REPO,
            WAN_REVISION,
            "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors",
            MODELS / "loras/WanVideo/WanAnimate_relight_lora_fp16.safetensors",
            "Wan Animate Relight LoRA",
        ),
        (
            WAN_REPO,
            WAN_REVISION,
            "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            MODELS
            / "loras/WanVideo/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "Lightx2v 4-step LoRA",
        ),
        (
            WAN_REPO,
            WAN_REVISION,
            "Wan2_1_VAE_bf16.safetensors",
            MODELS / "vae/wanvideo/Wan2_1_VAE_bf16.safetensors",
            "Wan VAE",
        ),
        (
            WAN_REPO,
            WAN_REVISION,
            "umt5-xxl-enc-bf16.safetensors",
            MODELS / "text_encoders/umt5-xxl-enc-bf16.safetensors",
            "UMT5 XXL",
        ),
        (
            COMFY_REPO,
            COMFY_REVISION,
            "split_files/clip_vision/clip_vision_h.safetensors",
            MODELS / "clip_vision/clip_vision_h.safetensors",
            "CLIP Vision H",
        ),
        (
            WAN_OFFICIAL_REPO,
            WAN_OFFICIAL_REVISION,
            "process_checkpoint/det/yolov10m.onnx",
            MODELS / "detection/onnx/yolov10m.onnx",
            "YOLOv10 detector",
        ),
        (
            VITPOSE_REPO,
            VITPOSE_REVISION,
            "onnx/wholebody/vitpose-l-wholebody.onnx",
            MODELS / "detection/vitpose-l-wholebody.onnx",
            "ViTPose whole-body",
        ),
        (
            SAM2_REPO,
            SAM2_REVISION,
            "sam2.1_hiera_base_plus.safetensors",
            MODELS / "sam2/sam2.1_hiera_base_plus.safetensors",
            "SAM2 auxiliar",
        ),
    ]


def verify(variant: str) -> bool:
    missing = []
    print("\nVerificacao dos modelos Wan 2.2 Animate:")
    for _repo, _revision, _remote_path, target, label in model_files(variant):
        if target.exists() and target.stat().st_size > 0:
            print(f"  [OK] {label}: {target}")
        else:
            print(f"  [FALTA] {label}: {target}")
            missing.append(target)
    if missing:
        print(f"\nFaltam {len(missing)} arquivo(s).")
        return False
    print("\nTodos os arquivos essenciais foram encontrados.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa os modelos do Wan 2.2 Animate")
    parser.add_argument(
        "--variant",
        choices=tuple(ANIMATE_MODELS),
        default=os.environ.get("MODEL_VARIANT", "e4m3fn").strip().lower(),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        return 0 if verify(args.variant) else 1
    for repo, revision, remote_path, target, _label in model_files(args.variant):
        download_as(repo, revision, remote_path, target)
    return 0 if verify(args.variant) else 1


if __name__ == "__main__":
    sys.exit(main())
