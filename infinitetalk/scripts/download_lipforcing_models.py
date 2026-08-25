#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

from huggingface_hub import hf_hub_download

from download_models import download_insightface_models, insightface_targets


MODELS = Path(os.environ.get("COMFYUI_MODELS", "/opt/ComfyUI/models"))
ROOT = MODELS / "lipforcing"
DOWNLOADS = MODELS / ".downloads"
TOKEN = os.environ.get("HF_TOKEN") or None
RUNTIME_PYTHON = Path(
    os.environ.get("LIPFORCING_PYTHON", "/opt/lipforcing-venv/bin/python")
)
RUNTIME_ROOT = Path(os.environ.get("LIPFORCING_ROOT", "/opt/LipForcing"))
SCRIPT_ROOT = Path(__file__).resolve().parent

LIPFORCING_REPO = "JinhyukJang/lipforcing"
LIPFORCING_REVISION = "49f1f15bdc0d266e6e6ef64ccaa1ee86367a8799"
WAN_REPO = "Wan-AI/Wan2.1-T2V-14B"
WAN_REVISION = "a064a6c71f5be440641209c07bf2a5ce7a2ff5e4"
WAV2VEC_REPO = "facebook/wav2vec2-base-960h"
WAV2VEC_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
TAEHV_REVISION = "e743234f3217ab3d1570f65642ab06596d1bd7c5"
TAEHV_URL = (
    "https://raw.githubusercontent.com/madebyollin/taehv/"
    f"{TAEHV_REVISION}/taew2_1.pth"
)
TAEHV_SHA256 = "d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e"
LATENTSYNC_REVISION = "a229c3948406bc2cf6eaf4873e662e70c6a04746"
MASK_URL = (
    "https://raw.githubusercontent.com/bytedance/LatentSync/"
    f"{LATENTSYNC_REVISION}/latentsync/utils/mask.png"
)
MASK_SHA256 = "aa233251b9ff5691a1565a4108f0910ab1e5e7ad79a7bb2b741ab4d92c81053c"

TEXT_EMBEDDING = ROOT / "text_emb_a_person_talking.pt"
TEXT_STAGING = DOWNLOADS / "lipforcing-text-encoder"
TEXT_ENCODER = TEXT_STAGING / "models_t5_umt5-xxl-enc-bf16.pth"
TEXT_ENCODER_FILES = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
)
WAV2VEC_FILES = (
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
)


def download_as(
    repo: str,
    revision: str,
    remote: str,
    target: Path,
    minimum_size: int = 1,
) -> None:
    if target.exists() and target.stat().st_size >= minimum_size:
        print(f"OK existente: {target}")
        return
    print(f"Baixando {repo}/{remote} -> {target}")
    local_root = DOWNLOADS / repo.replace("/", "--")
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            revision=revision,
            filename=remote,
            local_dir=local_root,
            token=TOKEN,
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded), str(target))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, target: Path, expected_sha256: str) -> None:
    if target.exists() and sha256(target) == expected_sha256:
        print(f"OK existente: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Baixando {url} -> {target}")
    urllib.request.urlretrieve(url, partial)
    actual = sha256(partial)
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum invalido para {target.name}: {actual}; esperado {expected_sha256}"
        )
    shutil.move(str(partial), str(target))


def embedding_command(verify_only: bool = False) -> list[str]:
    command = [
        str(RUNTIME_PYTHON),
        str(SCRIPT_ROOT / "precompute_lipforcing_text.py"),
        "--runtime-root",
        str(RUNTIME_ROOT),
        "--output",
        str(TEXT_EMBEDDING),
    ]
    if verify_only:
        command.append("--verify-only")
    else:
        command.extend(["--text-encoder", str(TEXT_ENCODER)])
    return command


def embedding_valid() -> bool:
    if not RUNTIME_PYTHON.exists() or not TEXT_EMBEDDING.exists():
        return False
    return subprocess.call(embedding_command(verify_only=True)) == 0


def ensure_text_embedding() -> None:
    if embedding_valid():
        return
    for remote in TEXT_ENCODER_FILES:
        minimum_size = 10_000_000_000 if remote == TEXT_ENCODER.name else 1
        download_as(
            WAN_REPO,
            WAN_REVISION,
            remote,
            TEXT_STAGING / remote,
            minimum_size,
        )
    subprocess.run(embedding_command(), check=True)
    if embedding_valid():
        shutil.rmtree(TEXT_STAGING, ignore_errors=True)
        print("Embedding T5 criado; encoder temporario de 11.4 GB removido.")


def required_targets() -> list[tuple[Path, str, int]]:
    targets = [
        (ROOT / "lipforcing_14b.pth", "Lip Forcing 14B", 20_000_000_000),
        (ROOT / "Wan2.1_VAE.pth", "Wan 2.1 VAE", 400_000_000),
        (ROOT / "taew2_1.pth", "TAEHV", 1_000_000),
        (ROOT / "mask.png", "Mascara LatentSync", 1_000),
    ]
    targets.extend(
        (ROOT / "wav2vec2-base-960h" / filename, f"Wav2Vec2 {filename}", 1)
        for filename in WAV2VEC_FILES
    )
    targets.extend(insightface_targets())
    return targets


def verify() -> bool:
    missing = []
    print("\nVerificacao dos modelos Lip Forcing 14B:")
    for target, label, minimum_size in required_targets():
        if target.exists() and target.stat().st_size >= minimum_size:
            print(f"  [OK] {label}: {target}")
        else:
            print(f"  [FALTA] {label}: {target}")
            missing.append(target)
    if not embedding_valid():
        print(f"  [FALTA] Embedding T5: {TEXT_EMBEDDING}")
        missing.append(TEXT_EMBEDDING)
    if missing:
        print(f"\nFaltam {len(missing)} arquivo(s) do Lip Forcing.")
        return False
    print("\nTodos os arquivos do Lip Forcing foram encontrados.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa e verifica os modelos locais do Lip Forcing 14B"
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        return 0 if verify() else 1
    download_insightface_models()
    download_as(
        LIPFORCING_REPO,
        LIPFORCING_REVISION,
        "lipforcing_14b.pth",
        ROOT / "lipforcing_14b.pth",
        20_000_000_000,
    )
    download_as(
        WAN_REPO,
        WAN_REVISION,
        "Wan2.1_VAE.pth",
        ROOT / "Wan2.1_VAE.pth",
        400_000_000,
    )
    for filename in WAV2VEC_FILES:
        download_as(
            WAV2VEC_REPO,
            WAV2VEC_REVISION,
            filename,
            ROOT / "wav2vec2-base-960h" / filename,
            300_000_000 if filename == "model.safetensors" else 1,
        )
    download_url(TAEHV_URL, ROOT / "taew2_1.pth", TAEHV_SHA256)
    download_url(MASK_URL, ROOT / "mask.png", MASK_SHA256)
    ensure_text_embedding()
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
