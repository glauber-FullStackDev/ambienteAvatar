#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
SCRIPTS_HOME = Path(__file__).resolve().parent
DEFAULT_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v.json",
    )
)
DEFAULT_ID_LORA_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_ID_LORA_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_id_lora.json",
    )
)
DEFAULT_IA2V_TALKVID_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_TALKVID_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_talkvid.json",
    )
)
SEEDED_WORKFLOW = COMFYUI_HOME / "user/default/workflows/video_ltx2_3_ia2v-docker.json"
SEEDED_ID_LORA_WORKFLOW = (
    COMFYUI_HOME / "user/default/workflows/video_ltx2_3_id_lora-docker.json"
)
SEEDED_IA2V_TALKVID_WORKFLOW = (
    COMFYUI_HOME / "user/default/workflows/video_ltx2_3_ia2v_talkvid-docker.json"
)


def prepare_directories() -> None:
    for relative in (
        "input",
        "models/checkpoints",
        "models/latent_upscale_models",
        "models/loras",
        "models/text_encoders",
        "output",
        "user/default/workflows",
    ):
        (COMFYUI_HOME / relative).mkdir(parents=True, exist_ok=True)


def seed_one_workflow(source: Path, target: Path, label: str) -> None:
    if target.exists():
        print(f"Workflow preservado: {target}")
        return
    if not source.is_file():
        raise SystemExit(f"Workflow padrao ausente: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    print(f"Workflow {label} instalado: {target}")


def seed_workflows() -> None:
    seed_one_workflow(DEFAULT_WORKFLOW, SEEDED_WORKFLOW, "LTX 2.3 IA2V")
    seed_one_workflow(
        DEFAULT_ID_LORA_WORKFLOW,
        SEEDED_ID_LORA_WORKFLOW,
        "LTX 2.3 ID-LoRA",
    )
    seed_one_workflow(
        DEFAULT_IA2V_TALKVID_WORKFLOW,
        SEEDED_IA2V_TALKVID_WORKFLOW,
        "LTX 2.3 IA2V + TalkVid",
    )


def run_downloader(*arguments: str) -> None:
    command = [sys.executable, str(SCRIPTS_HOME / "download_models.py"), *arguments]
    subprocess.run(command, check=True)


def serve() -> None:
    prepare_directories()
    if os.environ.get("DOWNLOAD_MODELS_ON_START", "1") == "1":
        print("Verificando os modelos do LTX 2.3 antes de iniciar o ComfyUI...")
        run_downloader()
    else:
        print("Download automatico desativado (DOWNLOAD_MODELS_ON_START=0).")
    seed_workflows()

    port = os.environ.get("COMFYUI_PORT", "8188")
    extra_args = shlex.split(os.environ.get("COMFYUI_ARGS", "--preview-method auto"))
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
