#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

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
LATENTSYNC_REPO = "ByteDance/LatentSync-1.6"
LATENTSYNC_REVISION = "c42c7e6c8e9c213626389fa7d9a3c444b8536353"
LATENTSYNC_VAE_REPO = "stabilityai/sd-vae-ft-mse"
LATENTSYNC_VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
FLASHVSR_REPO = "1038lab/FlashVSR"
FLASHVSR_REVISION = "f1bc675696d43f05d183d9b7c49e44d84c843caf"
SADTALKER_REPO = "vinthony/SadTalker"
SADTALKER_REVISION = "4aedd064359e623398a2d73eb8c253ebb2bd516c"
INSIGHTFACE_BUFFALO_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/"
    "buffalo_l.zip"
)
INSIGHTFACE_BUFFALO_SHA256 = (
    "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
)
INSIGHTFACE_BUFFALO_FILES = {
    "genderage.onnx": 1322532,
    "2d106det.onnx": 5030888,
    "det_10g.onnx": 16923827,
    "1k3d68.onnx": 143607619,
    "w600k_r50.onnx": 174383860,
}

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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def insightface_targets() -> list[tuple[Path, str, int]]:
    root = MODELS / "latentsync/auxiliary/models/buffalo_l"
    return [
        (root / filename, f"InsightFace buffalo_l {filename}", size)
        for filename, size in INSIGHTFACE_BUFFALO_FILES.items()
    ]


def download_insightface_models() -> None:
    if all(
        target.exists() and target.stat().st_size == size
        for target, _label, size in insightface_targets()
    ):
        print("OK existente: InsightFace buffalo_l")
        return

    archive = DOWNLOADS / "insightface--buffalo_l.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or file_sha256(archive) != INSIGHTFACE_BUFFALO_SHA256:
        partial = archive.with_suffix(".zip.part")
        print(f"Baixando {INSIGHTFACE_BUFFALO_URL} -> {archive}")
        urllib.request.urlretrieve(INSIGHTFACE_BUFFALO_URL, partial)
        if file_sha256(partial) != INSIGHTFACE_BUFFALO_SHA256:
            raise RuntimeError("Checksum invalido para InsightFace buffalo_l")
        shutil.move(str(partial), str(archive))

    extraction_root = DOWNLOADS / "insightface--buffalo_l"
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        missing = set(INSIGHTFACE_BUFFALO_FILES).difference(names)
        if missing:
            raise RuntimeError(
                "Arquivos ausentes no buffalo_l.zip: " + ", ".join(sorted(missing))
            )
        for target, _label, expected_size in insightface_targets():
            if target.exists() and target.stat().st_size == expected_size:
                continue
            extracted = extraction_root / target.name
            extracted.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(target.name) as source, extracted.open("wb") as output:
                shutil.copyfileobj(source, output)
            if extracted.stat().st_size != expected_size:
                raise RuntimeError(f"Tamanho invalido para {target.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), str(target))


def verify_insightface_detector() -> bool:
    detector = (
        MODELS
        / "latentsync/auxiliary/models/buffalo_l/det_10g.onnx"
    )
    if not detector.exists():
        return False
    try:
        from insightface.model_zoo import model_zoo

        model = model_zoo.get_model(
            str(detector),
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        print(f"  [FALHA] InsightFace detector nao carregou: {error}")
        return False
    if model is None or getattr(model, "taskname", None) != "detection":
        print("  [FALHA] det_10g.onnx nao foi reconhecido como detection")
        return False
    print("  [OK] InsightFace det_10g.onnx reconhecido como detection")
    return True


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
    files.extend(
        [
            *[
                (
                    FLASHVSR_REPO,
                    FLASHVSR_REVISION,
                    filename,
                    MODELS / "FlashVSR" / filename,
                    f"FlashVSR {filename}",
                )
                for filename in (
                    "FlashVSR1_1.safetensors",
                    "Wan2.1_VAE.safetensors",
                    "LQ_proj_in.safetensors",
                    "TCDecoder.safetensors",
                    "Prompt.safetensors",
                )
            ],
            (
                LATENTSYNC_REPO,
                LATENTSYNC_REVISION,
                "latentsync_unet.pt",
                MODELS / "latentsync/latentsync_unet.pt",
                "LatentSync 1.6 UNet 512x512",
            ),
            (
                LATENTSYNC_REPO,
                LATENTSYNC_REVISION,
                "whisper/tiny.pt",
                MODELS / "latentsync/whisper/tiny.pt",
                "LatentSync Whisper tiny",
            ),
            (
                LATENTSYNC_VAE_REPO,
                LATENTSYNC_VAE_REVISION,
                "config.json",
                MODELS / "latentsync/vae/config.json",
                "LatentSync VAE config",
            ),
            (
                LATENTSYNC_VAE_REPO,
                LATENTSYNC_VAE_REVISION,
                "diffusion_pytorch_model.safetensors",
                MODELS / "latentsync/vae/diffusion_pytorch_model.safetensors",
                "LatentSync VAE",
            ),
            (
                SADTALKER_REPO,
                SADTALKER_REVISION,
                "hub/checkpoints/s3fd-619a316812.pth",
                MODELS / "latentsync/s3fd-e19a316812.pth",
                "LatentSync S3FD face detector",
            ),
        ]
    )
    return files


def verify(quantization: str) -> bool:
    missing = []
    print("\nVerificacao dos modelos InfiniteTalk + LatentSync 1.6:")
    for _repo, _revision, _remote_path, target, label in model_files(quantization):
        if target.exists() and target.stat().st_size > 0:
            print(f"  [OK] {label}: {target}")
        else:
            print(f"  [FALTA] {label}: {target}")
            missing.append(target)
    for target, label, expected_size in insightface_targets():
        if target.exists() and target.stat().st_size == expected_size:
            print(f"  [OK] {label}: {target}")
        else:
            print(f"  [FALTA] {label}: {target}")
            missing.append(target)
    if not missing and not verify_insightface_detector():
        missing.append(
            MODELS / "latentsync/auxiliary/models/buffalo_l/det_10g.onnx"
        )
    if missing:
        print(f"\nFaltam {len(missing)} arquivo(s).")
        return False
    print("\nTodos os arquivos essenciais foram encontrados.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa os modelos do InfiniteTalk e LatentSync 1.6"
    )
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
    download_insightface_models()
    return 0 if verify(args.quantization) else 1


if __name__ == "__main__":
    sys.exit(main())
