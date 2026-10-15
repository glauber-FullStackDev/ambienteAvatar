#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
SCRIPTS_HOME = Path(__file__).resolve().parent
DEFAULT_IA2V_PERSONAL_LORA_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_PERSONAL_LORA_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_16gb.json",
    )
)
DEFAULT_A2V_PERSONAL_LORA_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_A2V_PERSONAL_LORA_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_a2v_personal_lora_16gb.json",
    )
)
SEEDED_IA2V_PERSONAL_LORA_WORKFLOW = (
    COMFYUI_HOME
    / "user/default/workflows/video_ltx2_3_ia2v_personal_lora_16gb-docker.json"
)
SEEDED_A2V_PERSONAL_LORA_WORKFLOW = (
    COMFYUI_HOME
    / "user/default/workflows/video_ltx2_3_a2v_personal_lora_16gb-docker.json"
)
PERSONAL_LORA_SOURCE = Path("/opt/ltx23-assets/glauberavatar.safetensors")
PERSONAL_LORA_TARGET = COMFYUI_HOME / "models/loras/glauberavatar.safetensors"
PERSONAL_LORA_SHA256 = "c9443d3aad3151125c086c10e6a423ccc973bd445aa67cd5a537bdd57f6df78a"


def prepare_directories() -> None:
    for relative in (
        "input",
        "models/diffusion_models",
        "models/latent_upscale_models",
        "models/loras",
        "models/text_encoders",
        "models/vae",
        "output",
        "user/default/workflows",
    ):
        (COMFYUI_HOME / relative).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_personal_lora() -> None:
    if not PERSONAL_LORA_SOURCE.is_file():
        raise SystemExit(f"LoRA pessoal incorporado ausente: {PERSONAL_LORA_SOURCE}")
    if sha256(PERSONAL_LORA_SOURCE) != PERSONAL_LORA_SHA256:
        raise SystemExit("checksum inválido do LoRA pessoal incorporado")
    if PERSONAL_LORA_TARGET.is_file():
        if sha256(PERSONAL_LORA_TARGET) != PERSONAL_LORA_SHA256:
            raise SystemExit(
                f"LoRA pessoal existente tem checksum diferente: {PERSONAL_LORA_TARGET}"
            )
        return
    temporary = PERSONAL_LORA_TARGET.with_suffix(".safetensors.tmp")
    shutil.copyfile(PERSONAL_LORA_SOURCE, temporary)
    os.replace(temporary, PERSONAL_LORA_TARGET)
    print(f"LoRA pessoal instalado no volume de modelos: {PERSONAL_LORA_TARGET}")


def read_schema_version(path: Path, marker: str) -> int | None:
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = workflow.get("extra", {}).get(marker)
    return version if isinstance(version, int) else None


def backup_path_for(target: Path, schema_version: int) -> Path:
    base = target.with_name(
        f"{target.stem}.schema-v{schema_version}.backup{target.suffix}"
    )
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = target.with_name(
            f"{target.stem}.schema-v{schema_version}.backup-{index}{target.suffix}"
        )
        if not candidate.exists():
            return candidate
        index += 1


def seed_one_workflow(
    source: Path,
    target: Path,
    label: str,
    *,
    schema_marker: str | None = None,
    schema_version: int | None = None,
) -> None:
    installed_version = None
    if target.exists():
        installed_version = (
            read_schema_version(target, schema_marker) if schema_marker else None
        )
        needs_upgrade = (
            schema_version is not None
            and installed_version is not None
            and installed_version < schema_version
        )
        if not needs_upgrade:
            print(f"Workflow preservado: {target}")
            return
    if not source.is_file():
        raise SystemExit(f"Workflow padrao ausente: {source}")
    if schema_marker and schema_version is not None:
        source_version = read_schema_version(source, schema_marker)
        if source_version != schema_version:
            raise SystemExit(
                f"Workflow padrao {label} usa schema {source_version!r}; "
                f"esperado {schema_version}"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert installed_version is not None
        backup = backup_path_for(target, installed_version)
        shutil.copyfile(target, backup)
        print(f"Backup do workflow antigo criado: {backup}")
    temporary = target.with_suffix(".json.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    action = "atualizado" if installed_version is not None else "instalado"
    print(f"Workflow {label} {action}: {target}")


def seed_workflows() -> None:
    seed_one_workflow(
        DEFAULT_IA2V_PERSONAL_LORA_WORKFLOW,
        COMFYUI_HOME
        / "user/default/workflows/video_ltx2_3_ia2v_personal_lora_16gb-docker.json",
        "LTX 2.3 IA2V + LoRA pessoal (GGUF 16 GB)",
        schema_marker="ltx23_16gb_gguf_schema",
        schema_version=1,
    )
    seed_one_workflow(
        DEFAULT_A2V_PERSONAL_LORA_WORKFLOW,
        COMFYUI_HOME
        / "user/default/workflows/video_ltx2_3_a2v_personal_lora_16gb-docker.json",
        "LTX 2.3 A2V + LoRA pessoal (GGUF 16 GB)",
        schema_marker="ltx23_16gb_gguf_schema",
        schema_version=1,
    )


def run_downloader(*arguments: str) -> None:
    command = [sys.executable, str(SCRIPTS_HOME / "download_models.py"), *arguments]
    subprocess.run(command, check=True)


def serve() -> None:
    prepare_directories()
    ensure_personal_lora()
    if os.environ.get("DOWNLOAD_MODELS_ON_START", "1") == "1":
        print("Verificando os modelos GGUF do LTX 2.3 antes de iniciar o ComfyUI...")
        run_downloader()
    else:
        print("Download automatico desativado (DOWNLOAD_MODELS_ON_START=0).")
    seed_workflows()

    port = os.environ.get("COMFYUI_PORT", "8188")
    extra_args = shlex.split(
        os.environ.get("COMFYUI_ARGS", "--preview-method auto --reserve-vram 1.5")
    )
    command = [
        sys.executable,
        str(COMFYUI_HOME / "main.py"),
        "--listen",
        "0.0.0.0",
        "--port",
        port,
        *extra_args,
    ]
    print("Iniciando ComfyUI:", shlex.join(command))
    os.chdir(COMFYUI_HOME)
    os.execvp(command[0], command)


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if action == "serve":
        serve()
    elif action == "download-models":
        prepare_directories()
        run_downloader(*sys.argv[2:])
    elif action == "verify":
        prepare_directories()
        run_downloader("--verify-only", *sys.argv[2:])
    else:
        raise SystemExit(
            f"Acao desconhecida: {action!r}. Use serve, download-models ou verify."
        )


if __name__ == "__main__":
    main()
