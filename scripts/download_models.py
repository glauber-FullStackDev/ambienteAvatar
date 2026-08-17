#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download, snapshot_download


AVATAR_REPO = "meituan-longcat/LongCat-Video-Avatar-1.5"
AVATAR_REVISION = "92016c71d5d318d0f5d84e4db30015a571484ab6"
BASE_REPO = "meituan-longcat/LongCat-Video"
BASE_REVISION = "03b55529b1d1d4045f5fbe14d65c8c6e8116b278"
COMFY_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
COMFY_REVISION = "617a7633e636506f850e043bc4605f290a466a8e"
MERGED_REPO = "smthem/LongCat-Video-Avatar-1.5-merge"
MERGED_REVISION = "753d7f13ed5dd29e028ee1675912128abfddf6d8"

MODELS = Path(os.environ.get("COMFYUI_MODELS", "/opt/ComfyUI/models"))
DOWNLOADS = MODELS / ".downloads"
TOKEN = os.environ.get("HF_TOKEN") or None


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def download_snapshot(repo: str, revision: str, patterns: list[str], target: Path) -> None:
    print(f"Sincronizando {repo}: {', '.join(patterns)}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=patterns,
        local_dir=target,
        token=TOKEN,
    )


def expected_files(mode: str, text_mode: str, include_vocal: bool) -> list[tuple[Path, str]]:
    expected: list[tuple[Path, str]] = [
        (MODELS / "loras/longcat-avatar-dmd_lora.safetensors", "LoRA DMD"),
        (MODELS / "vae/LongCat-Video-Avatar-vae.safetensors", "VAE"),
        (MODELS / "audio_encoders/whisper-large-v3.safetensors", "Whisper large-v3"),
    ]
    if mode == "official_int8_sharded":
        root = MODELS / "longcat/LongCat-Video-Avatar-1.5/base_model_int8"
        expected.extend(
            [
                (root / "config.json", "configuração DiT INT8"),
                (root / "quantization_config.json", "configuração de quantização INT8"),
                (root / "quantized_model.safetensors.index.json", "índice DiT INT8"),
                *[
                    (
                        root / f"quantized_model-{index:05d}-of-00004.safetensors",
                        f"shard DiT INT8 {index}/4",
                    )
                    for index in range(1, 5)
                ],
            ]
        )
    elif mode == "official_sharded":
        root = MODELS / "longcat/LongCat-Video-Avatar-1.5/base_model"
        expected.extend(
            [
                (root / "config.json", "configuração DiT BF16"),
                (root / "diffusion_pytorch_model.safetensors.index.json", "índice DiT BF16"),
                *[
                    (
                        root / f"diffusion_pytorch_model-{index:05d}-of-00006.safetensors",
                        f"shard DiT BF16 {index}/6",
                    )
                    for index in range(1, 7)
                ],
            ]
        )
    else:
        expected.append(
            (
                MODELS / "diffusion_models/LongCat-Video-Avatar-1.5-int8.safetensors",
                "DiT INT8 em arquivo único",
            )
        )

    if text_mode == "native":
        root = MODELS / "longcat/LongCat-Video"
        expected.extend(
            [
                (root / "tokenizer/tokenizer.json", "tokenizer UMT5"),
                (root / "text_encoder/model.safetensors.index.json", "text encoder UMT5"),
                *[
                    (
                        root / f"text_encoder/model-{index:05d}-of-00005.safetensors",
                        f"text encoder shard {index}/5",
                    )
                    for index in range(1, 6)
                ],
            ]
        )
    else:
        expected.append(
            (MODELS / "clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "UMT5 FP8 ComfyUI")
        )

    if include_vocal:
        expected.append((MODELS / "longcat/Kim_Vocal_2.onnx", "separador vocal"))
    return expected


def verify(mode: str, text_mode: str, include_vocal: bool) -> bool:
    missing = []
    print("\nVerificação dos modelos:")
    for path, label in expected_files(mode, text_mode, include_vocal):
        if path.exists() and path.stat().st_size > 0:
            print(f"  [OK] {label}: {path}")
        else:
            print(f"  [FALTA] {label}: {path}")
            missing.append(path)
    if missing:
        print(f"\nFaltam {len(missing)} arquivo(s).")
        return False
    print("\nTodos os arquivos essenciais foram encontrados.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa os modelos do LongCat Avatar 1.5")
    parser.add_argument(
        "--mode",
        choices=("official_int8_sharded", "official_sharded", "single_file_safetensors"),
        default=os.environ.get("MODEL_MODE", "official_int8_sharded"),
    )
    parser.add_argument(
        "--text-encoder",
        choices=("clip_fp8", "native"),
        default=os.environ.get("TEXT_ENCODER_MODE", "clip_fp8"),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--include-vocal-separator",
        action=argparse.BooleanOptionalAction,
        default=env_flag("INCLUDE_VOCAL_SEPARATOR", True),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        return 0 if verify(args.mode, args.text_encoder, args.include_vocal_separator) else 1

    if args.mode == "official_int8_sharded":
        download_snapshot(
            AVATAR_REPO,
            AVATAR_REVISION,
            ["base_model_int8/*"],
            MODELS / "longcat/LongCat-Video-Avatar-1.5",
        )
    elif args.mode == "official_sharded":
        download_snapshot(
            AVATAR_REPO,
            AVATAR_REVISION,
            ["base_model/*"],
            MODELS / "longcat/LongCat-Video-Avatar-1.5",
        )
    else:
        download_as(
            MERGED_REPO,
            MERGED_REVISION,
            "LongCat-Video-Avatar-1.5-int8.safetensors",
            MODELS / "diffusion_models/LongCat-Video-Avatar-1.5-int8.safetensors",
        )

    download_as(
        AVATAR_REPO,
        AVATAR_REVISION,
        "lora/dmd_lora.safetensors",
        MODELS / "loras/longcat-avatar-dmd_lora.safetensors",
    )
    download_as(
        AVATAR_REPO,
        AVATAR_REVISION,
        "whisper-large-v3/model.safetensors",
        MODELS / "audio_encoders/whisper-large-v3.safetensors",
    )
    download_as(
        BASE_REPO,
        BASE_REVISION,
        "vae/diffusion_pytorch_model.safetensors",
        MODELS / "vae/LongCat-Video-Avatar-vae.safetensors",
    )

    if args.text_encoder == "native":
        download_snapshot(
            BASE_REPO,
            BASE_REVISION,
            ["tokenizer/*", "text_encoder/*"],
            MODELS / "longcat/LongCat-Video",
        )
    else:
        download_as(
            COMFY_REPO,
            COMFY_REVISION,
            "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            MODELS / "clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        )

    if args.include_vocal_separator:
        download_as(
            AVATAR_REPO,
            AVATAR_REVISION,
            "vocal_separator/Kim_Vocal_2.onnx",
            MODELS / "longcat/Kim_Vocal_2.onnx",
        )

    return 0 if verify(args.mode, args.text_encoder, args.include_vocal_separator) else 1


if __name__ == "__main__":
    sys.exit(main())
