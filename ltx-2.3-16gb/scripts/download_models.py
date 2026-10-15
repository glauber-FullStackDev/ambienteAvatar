#!/usr/bin/env python3
"""Baixa os modelos GGUF fixados da variante LTX 2.3 16 GB (~26 GiB).

Pacote (repositorio unsloth/LTX-2.3-GGUF + gemma QAT), tudo publico:

- DiT LTX-2.3-22B-distilled-1.1 em GGUF Q4_K_M (padrao; Q5_K_M e Q6_K
  opcionais via LTX23_GGUF_QUANT) -> models/diffusion_models;
- Gemma 3 12B QAT GGUF (UD-Q4_K_XL) -> models/text_encoders;
- embeddings connectors standalone -> models/text_encoders;
- video VAE e audio VAE standalone -> models/vae;
- spatial upscaler x2 1.1 -> models/latent_upscale_models;
- LoRA de prompt-enhancement do Gemma -> models/loras.

O LoRA pessoal glauberavatar.safetensors nao e baixado: vem incorporado na
imagem (/opt/ltx23-assets) e o entrypoint o instala no volume.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError


MODELS = Path(os.environ.get("COMFYUI_MODELS", "/opt/ComfyUI/models"))
DOWNLOADS = MODELS / ".downloads/ltx23-16gb"
TOKEN = os.environ.get("HF_TOKEN") or None
DOWNLOAD_HEADROOM = 2 * 1024**3

UNSLOLT_LTX23_REVISION = "96e8ed4925ead3db9ff4d0084f165ef6a74f28d0"
GEMMA_QAT_REVISION = "858acec7ec0541a46c39985c95d3b52d8f3ab183"
LIGHTRICKS_LTX23_REVISION = "6b5a83e3045eaf8e46cfa0acce512412aa2b9cce"
COMFY_ORG_LTX2_REVISION = "101c239b4b64dd1b45d645365339c56e0e7df4c3"


@dataclass(frozen=True)
class ModelFile:
    label: str
    repo: str
    revision: str
    remote_path: str
    relative_path: str
    size: int
    sha256: str

    @property
    def target(self) -> Path:
        return MODELS / self.relative_path


# Quants suportados para o DiT distilled 1.1 (caminho dentro do repo unsloth).
GGUF_QUANTS = {
    "Q4_K_M": (
        "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
        14_194_441_248,
        "5d09efdc0b8ec2054c44a05366cd7c6634ffa333b379b1f8baf018a78974b73d",
    ),
    "Q5_K_M": (
        "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q5_K_M.gguf",
        15_930_424_352,
        "",
    ),
    "Q6_K": (
        "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K.gguf",
        17_774_906_400,
        "",
    ),
}
DEFAULT_GGUF_QUANT = "Q4_K_M"

GEMMA_GGUF_FILE = "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
CONNECTORS_FILE = "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
VIDEO_VAE_FILE = "ltx-2.3-22b-distilled_video_vae.safetensors"
AUDIO_VAE_FILE = "ltx-2.3-22b-distilled_audio_vae.safetensors"


def selected_quant() -> str:
    quant = os.environ.get("LTX23_GGUF_QUANT", DEFAULT_GGUF_QUANT).upper()
    if quant not in ("Q4_K_M", "Q5_K_M", "Q6_K"):
        raise SystemExit(
            f"LTX23_GGUF_QUANT invalido: {quant!r}. "
            "Use Q4_K_M (padrao), Q5_K_M ou Q6_K."
        )
    return quant


def model_files(quant: str) -> tuple[ModelFile, ...]:
    quant_files = {
        "Q4_K_M": (
            "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
            14_194_441_248,
            "5d09efdc0b8ec2054c44a05366cd7c6634ffa333b379b1f8baf018a78974b73d",
        ),
        "Q5_K_M": (
            "ltx-2.3-22b-distilled-1.1-Q5_K_M.gguf",
            15_930_424_352,
            "",
        ),
        "Q6_K": (
            "ltx-2.3-22b-distilled-1.1-Q6_K.gguf",
            17_774_906_400,
            "",
        ),
    }
    dit_filename, size, sha256 = quant_files[quant]
    return (
        ModelFile(
            label=f"LTX 2.3 22B distilled 1.1 GGUF {quant}",
            repo="unsloth/LTX-2.3-GGUF",
            revision="96e8ed4925ead3db9ff4d0084f165ef6a74f28d0",
            remote_path=f"distilled-1.1/{dit_filename}",
            relative_path=f"diffusion_models/{dit_filename}",
            size=size,
            sha256=sha256,
        ),
        ModelFile(
            label="Gemma 3 12B QAT GGUF (UD-Q4_K_XL)",
            repo="unsloth/gemma-3-12b-it-qat-GGUF",
            revision="858acec7ec0541a46c39985c95d3b52d8f3ab183",
            remote_path=GEMMA_GGUF_FILE,
            relative_path=f"text_encoders/{GEMMA_GGUF_FILE}",
            size=7_432_229_248,
            sha256="da98f81c86916ed1c76b3eeda56b25cb7b8352b01093e2edb8028110fe2cb53b",
        ),
        ModelFile(
            label="LTX 2.3 distilled embeddings connectors",
            repo="unsloth/LTX-2.3-GGUF",
            revision="96e8ed4925ead3db9ff4d0084f165ef6a74f28d0",
            remote_path=f"text_encoders/{CONNECTORS_FILE}",
            relative_path=f"text_encoders/{CONNECTORS_FILE}",
            size=2_312_144_712,
            sha256="c61cbb396e2a8175d8b2da51f0fdac885a4ccd22c9f64dafa5aa2c455dc8a507",
        ),
        ModelFile(
            label="LTX 2.3 distilled video VAE",
            repo="unsloth/LTX-2.3-GGUF",
            revision="96e8ed4925ead3db9ff4d0084f165ef6a74f28d0",
            remote_path=f"vae/{VIDEO_VAE_FILE}",
            relative_path=f"vae/{VIDEO_VAE_FILE}",
            size=1_452_256_522,
            sha256="e68d6d8f8a42942ac9b862cc315beb3bc30805a8876c7ad63ba5bf7a2b8e168a",
        ),
        ModelFile(
            label="LTX 2.3 distilled audio VAE",
            repo="unsloth/LTX-2.3-GGUF",
            revision="96e8ed4925ead3db9ff4d0084f165ef6a74f28d0",
            remote_path=f"vae/{AUDIO_VAE_FILE}",
            relative_path=f"vae/{AUDIO_VAE_FILE}",
            size=364_853_140,
            sha256="3cd6a6eb8cb28f5ecc12f1f3126952b2a3d2b0b42ad3270e63cefafafe0d9b57",
        ),
        ModelFile(
            label="LTX 2.3 spatial upscaler x2",
            repo="Lightricks/LTX-2.3",
            revision="6b5a83e3045eaf8e46cfa0acce512412aa2b9cce",
            remote_path="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            relative_path=(
                "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
            ),
            size=995_743_560,
            sha256="5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed",
        ),
        ModelFile(
            label="Gemma prompt-enhancement LoRA",
            repo="Comfy-Org/ltx-2",
            revision="101c239b4b64dd1b45d645365339c56e0e7df4c3",
            remote_path=(
                "split_files/loras/"
                "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
            ),
            relative_path=(
                "loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
            ),
            size=628_203_616,
            sha256="87bcabeac9bec9f374232b5122d6511c2b2112d479e50176149e944b3712eb4a",
        ),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(model: ModelFile, verify_sha256: bool = False) -> bool:
    target = model.target
    if not target.is_file() or target.stat().st_size != model.size:
        return False
    if verify_sha256 and model.sha256 and file_sha256(target) != model.sha256:
        return False
    return True


def required_download_bytes(models: tuple[ModelFile, ...]) -> int:
    return sum(model.size for model in models if not is_valid(model))


def check_free_space(models: tuple[ModelFile, ...]) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    required = required_download_bytes(models) + DOWNLOAD_HEADROOM
    free = shutil.disk_usage(MODELS).free
    if free < required:
        raise RuntimeError(
            "Espaco insuficiente no volume de modelos: "
            f"livre={free / 1024**3:.1f} GiB, "
            f"necessario={required / 1024**3:.1f} GiB incluindo margem."
        )


def staging_directory(model: ModelFile) -> Path:
    repo_name = model.repo.replace("/", "--")
    return DOWNLOADS / repo_name / model.revision


def download(model: ModelFile, verify_sha256: bool) -> None:
    if is_valid(model, verify_sha256):
        print(f"  [OK] {model.label}: {model.target}")
        return

    target = model.target
    if target.exists():
        print(f"  [REFazer] arquivo incompleto ou invalido: {target}")
        target.unlink()

    stage_root = staging_directory(model)
    stage_root.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {model.repo}/{model.remote_path} -> {target}")
    downloaded = Path(
        hf_hub_download(
            repo_id=model.repo,
            revision=model.revision,
            filename=model.remote_path,
            local_dir=stage_root,
            token=TOKEN,
        )
    )
    if downloaded.stat().st_size != model.size:
        raise RuntimeError(
            f"Tamanho invalido para {model.label}: "
            f"{downloaded.stat().st_size} != {model.size}"
        )
    if verify_sha256 and model.sha256 and file_sha256(downloaded) != model.sha256:
        raise RuntimeError(f"SHA256 invalido para {model.label}")

    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(downloaded, target)
    print(f"  [OK] {model.label}: {target}")


def verify(models: tuple[ModelFile, ...], verify_sha256: bool) -> bool:
    success = True
    for model in models:
        if is_valid(model, verify_sha256):
            detail = "tamanho e SHA256" if verify_sha256 else "tamanho"
            print(f"  [OK] {model.label} ({detail}): {model.target}")
        else:
            print(f"  [AUSENTE/INVALIDO] {model.label}: {model.target}")
            success = False
    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baixa os modelos GGUF fixados do LTX 2.3 (variante 16 GB)"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        default=os.environ.get("VERIFY_MODEL_SHA256", "0") == "1",
        help="calcula o SHA256 completo dos modelos selecionados",
    )
    args = parser.parse_args()

    quant = selected_quant()
    models = model_files(quant)

    if args.verify_only:
        raise SystemExit(0 if verify(models, args.verify_sha256) else 1)

    check_free_space(models)
    for model in models:
        download(model, args.verify_sha256)
    if not verify(models, args.verify_sha256):
        raise SystemExit(1)
    print(f"Todos os modelos GGUF ({quant}) da variante 16 GB estao prontos.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Download interrompido; execute novamente para retomar.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
