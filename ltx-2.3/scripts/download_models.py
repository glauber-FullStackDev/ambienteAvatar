#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download


MODELS = Path(os.environ.get("COMFYUI_MODELS", "/opt/ComfyUI/models"))
DOWNLOADS = MODELS / ".downloads/ltx23"
TOKEN = os.environ.get("HF_TOKEN") or None
DOWNLOAD_HEADROOM = 2 * 1024**3


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


MODEL_FILES = (
    ModelFile(
        label="LTX 2.3 22B dev FP8",
        repo="Lightricks/LTX-2.3-fp8",
        revision="1d756cd27fa11c0896c4dfee093cd1bf36c7f7a1",
        remote_path="ltx-2.3-22b-dev-fp8.safetensors",
        relative_path="checkpoints/ltx-2.3-22b-dev-fp8.safetensors",
        size=29_145_431_166,
        sha256="28606c5b5a06ce56f896d4dfcb20f212739e07a68fbe48e53638188449d26450",
    ),
    ModelFile(
        label="Gemma 3 12B FP4 mixed",
        repo="Comfy-Org/ltx-2",
        revision="101c239b4b64dd1b45d645365339c56e0e7df4c3",
        remote_path="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        relative_path="text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        size=9_447_702_218,
        sha256="aaca463d11e6d8d2a4bdb0d6299214c15ef78a3f73e0ef8113d5a9d0219b3f6d",
    ),
    ModelFile(
        label="LTX 2.3 distilled dynamic LoRA",
        repo="Comfy-Org/ltx-2.3",
        revision="f246c0865f5214499a12b72d47464ac8f4f54bee",
        remote_path=(
            "split_files/loras/"
            "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
        ),
        relative_path=(
            "loras/"
            "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
        ),
        size=2_741_024_390,
        sha256="31e0c0195fb841bf31af78e8b60858f489e87ddcea4a5239abc80943da65e3ac",
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
    if verify_sha256 and file_sha256(target) != model.sha256:
        return False
    return True


def required_download_bytes() -> int:
    return sum(model.size for model in MODEL_FILES if not is_valid(model))


def check_free_space() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    required = required_download_bytes() + DOWNLOAD_HEADROOM
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
    if verify_sha256 and file_sha256(downloaded) != model.sha256:
        raise RuntimeError(f"SHA256 invalido para {model.label}")

    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(downloaded, target)
    print(f"  [OK] {model.label}: {target}")


def verify(verify_sha256: bool) -> bool:
    success = True
    for model in MODEL_FILES:
        if is_valid(model, verify_sha256):
            detail = "tamanho e SHA256" if verify_sha256 else "tamanho"
            print(f"  [OK] {model.label} ({detail}): {model.target}")
        else:
            print(f"  [AUSENTE/INVALIDO] {model.label}: {model.target}")
            success = False
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa os modelos fixados do LTX 2.3")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        default=os.environ.get("VERIFY_MODEL_SHA256", "0") == "1",
        help="calcula o SHA256 completo dos aproximadamente 40 GiB",
    )
    args = parser.parse_args()

    if args.verify_only:
        raise SystemExit(0 if verify(args.verify_sha256) else 1)

    check_free_space()
    for model in MODEL_FILES:
        download(model, args.verify_sha256)
    if not verify(args.verify_sha256):
        raise SystemExit(1)
    print("Todos os modelos do LTX 2.3 estao prontos.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Download interrompido; execute novamente para retomar.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
