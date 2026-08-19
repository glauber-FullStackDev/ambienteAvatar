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

BASE_REPO = "city96/Wan2.1-I2V-14B-480P-gguf"
BASE_REVISION = "b4b6b3b3f0f64975fb33b1d5d1a46ca24e065f03"
INFINITETALK_REPO = "Kijai/WanVideo_comfy_GGUF"
INFINITETALK_REVISION = "8418a4ed00314a49f44b73a0d43d40aa539264da"
WAN_REPO = "Kijai/WanVideo_comfy"
WAN_REVISION = "8260d429d19fd7a72304cad059160b95d843913f"
COMFY_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
COMFY_REVISION = "617a7633e636506f850e043bc4605f290a466a8e"
WAV2VEC_REPO = "Kijai/wav2vec2_safetensors"
WAV2VEC_REVISION = "87847d3bc53702afda44078249e7c33e867827c4"
MELBAND_REPO = "Kijai/MelBandRoFormer_comfy"
MELBAND_REVISION = "7dc5fa7824f1f3089a5c4b130d767004ccc1ed12"

QUANTIZED_MODELS = {
    "q4_k_m": (
        "wan2.1-i2v-14b-480p-Q4_K_M.gguf",
        "InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf",
    ),
    "q6_k": (
        "wan2.1-i2v-14b-480p-Q6_K.gguf",
        "InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q6_K.gguf",
    ),
    "q8": (
        "wan2.1-i2v-14b-480p-Q8_0.gguf",
        "InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf",
    ),
}

LIGHTX2V_LORAS = (64, 128, 256)


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


def model_files(quantization: str) -> list[tuple[str, str, str, Path, str]]:
    base_name, infinitetalk_path = QUANTIZED_MODELS[quantization]
    infinitetalk_name = Path(infinitetalk_path).name
    files = [
        (
            BASE_REPO,
            BASE_REVISION,
            base_name,
            MODELS / "diffusion_models/WanVideo" / base_name,
            "Wan 2.1 I2V base",
        ),
        (
            INFINITETALK_REPO,
            INFINITETALK_REVISION,
            infinitetalk_path,
            MODELS / "diffusion_models/WanVideo/InfiniteTalk" / infinitetalk_name,
            "InfiniteTalk Single",
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
            WAV2VEC_REPO,
            WAV2VEC_REVISION,
            "wav2vec2-chinese-base_fp16.safetensors",
            MODELS / "wav2vec2/wav2vec2-chinese-base_fp16.safetensors",
            "Wav2Vec2",
        ),
        (
            MELBAND_REPO,
            MELBAND_REVISION,
            "MelBandRoformer_fp16.safetensors",
            MODELS / "diffusion_models/MelBandRoFormer/MelBandRoformer_fp16.safetensors",
            "MelBand RoFormer",
        ),
    ]
    for rank in LIGHTX2V_LORAS:
        filename = (
            f"lightx2v_I2V_14B_480p_cfg_step_distill_rank{rank}_bf16.safetensors"
        )
        files.append(
            (
                WAN_REPO,
                WAN_REVISION,
                f"Lightx2v/{filename}",
                MODELS / "loras/WanVideo/Lightx2v" / filename,
                f"Lightx2v I2V LoRA rank{rank}",
            )
        )
    return files


def verify(quantization: str) -> bool:
    missing = []
    print("\nVerificacao dos modelos InfiniteTalk:")
    for _repo, _revision, _remote_path, target, label in model_files(quantization):
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
    parser = argparse.ArgumentParser(description="Baixa os modelos do InfiniteTalk")
    parser.add_argument(
        "--quantization",
        choices=tuple(QUANTIZED_MODELS),
        default=os.environ.get("MODEL_QUANTIZATION", "q8").lower(),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        return 0 if verify(args.quantization) else 1
    for repo, revision, remote_path, target, _label in model_files(args.quantization):
        download_as(repo, revision, remote_path, target)
    return 0 if verify(args.quantization) else 1


if __name__ == "__main__":
    sys.exit(main())
