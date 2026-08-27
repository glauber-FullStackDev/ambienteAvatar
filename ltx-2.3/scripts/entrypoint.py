#!/usr/bin/env python3
from __future__ import annotations

import json
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
DEFAULT_IA2V_BEST_FACE_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_BEST_FACE_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_best_face.json",
    )
)
DEFAULT_IA2V_INGREDIENTS_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_INGREDIENTS_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_ingredients.json",
    )
)
DEFAULT_IA2V_INGREDIENTS_LEGACY_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_INGREDIENTS_LEGACY_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_ingredients_legacy_v2.json",
    )
)
DEFAULT_INGREDIENTS_OFFICIAL_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_INGREDIENTS_OFFICIAL_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ingredients_official_single_stage.json",
    )
)
DEFAULT_INGREDIENTS_WANGP_I2V_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_INGREDIENTS_WANGP_I2V_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ingredients_wangp_i2v_15s.json",
    )
)
DEFAULT_LTX25_IA2V_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_LTX25_IA2V_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_5_ia2v_distilled_8steps.json",
    )
)
SEEDED_WORKFLOW = COMFYUI_HOME / "user/default/workflows/video_ltx2_3_ia2v-docker.json"
SEEDED_ID_LORA_WORKFLOW = (
    COMFYUI_HOME / "user/default/workflows/video_ltx2_3_id_lora-docker.json"
)
SEEDED_IA2V_TALKVID_WORKFLOW = (
    COMFYUI_HOME / "user/default/workflows/video_ltx2_3_ia2v_talkvid-docker.json"
)
SEEDED_IA2V_BEST_FACE_WORKFLOW = (
    COMFYUI_HOME / "user/default/workflows/video_ltx2_3_ia2v_best_face-docker.json"
)
SEEDED_IA2V_INGREDIENTS_WORKFLOW = (
    COMFYUI_HOME / "user/default/workflows/video_ltx2_3_ia2v_ingredients-docker.json"
)
SEEDED_IA2V_INGREDIENTS_LEGACY_WORKFLOW = (
    COMFYUI_HOME
    / "user/default/workflows/video_ltx2_3_ia2v_ingredients_legacy_v2-docker.json"
)
SEEDED_INGREDIENTS_OFFICIAL_WORKFLOW = (
    COMFYUI_HOME
    / "user/default/workflows/video_ltx2_3_ingredients_official_single_stage-docker.json"
)
SEEDED_INGREDIENTS_WANGP_I2V_WORKFLOW = (
    COMFYUI_HOME
    / "user/default/workflows/video_ltx2_3_ingredients_wangp_i2v_15s-docker.json"
)
SEEDED_LTX25_IA2V_WORKFLOW = (
    COMFYUI_HOME
    / "user/default/workflows/video_ltx2_5_ia2v_distilled_8steps-docker.json"
)


def prepare_directories() -> None:
    for relative in (
        "input",
        "models/checkpoints",
        "models/diffusion_models",
        "models/latent_upscale_models",
        "models/loras",
        "models/text_encoders",
        "models/vae",
        "output",
        "user/default/workflows",
    ):
        (COMFYUI_HOME / relative).mkdir(parents=True, exist_ok=True)


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
    seed_one_workflow(
        DEFAULT_IA2V_BEST_FACE_WORKFLOW,
        SEEDED_IA2V_BEST_FACE_WORKFLOW,
        "LTX 2.3 IA2V + Best Face-ID",
    )
    seed_one_workflow(
        DEFAULT_IA2V_INGREDIENTS_WORKFLOW,
        SEEDED_IA2V_INGREDIENTS_WORKFLOW,
        "LTX 2.3 IA2V + IC-LoRA Ingredients",
        schema_marker="ltx23_ia2v_ingredients_schema",
        schema_version=3,
    )
    seed_one_workflow(
        DEFAULT_IA2V_INGREDIENTS_LEGACY_WORKFLOW,
        SEEDED_IA2V_INGREDIENTS_LEGACY_WORKFLOW,
        "LTX 2.3 IA2V + IC-LoRA Ingredients legado schema 2",
        schema_marker="ltx23_ia2v_ingredients_schema",
        schema_version=2,
    )
    seed_one_workflow(
        DEFAULT_INGREDIENTS_OFFICIAL_WORKFLOW,
        SEEDED_INGREDIENTS_OFFICIAL_WORKFLOW,
        "LTX 2.3 IC-LoRA Ingredients oficial single-stage",
        schema_marker="ltx23_ingredients_reference_schema",
        schema_version=1,
    )
    seed_one_workflow(
        DEFAULT_INGREDIENTS_WANGP_I2V_WORKFLOW,
        SEEDED_INGREDIENTS_WANGP_I2V_WORKFLOW,
        "LTX 2.3 IC-LoRA Ingredients WanGP I2V 15s",
        schema_marker="ltx23_ingredients_reference_schema",
        schema_version=1,
    )
    seed_one_workflow(
        DEFAULT_LTX25_IA2V_WORKFLOW,
        SEEDED_LTX25_IA2V_WORKFLOW,
        "LTX 2.5 IA2V Distilled 8 Steps",
    )


def run_downloader(*arguments: str) -> None:
    command = [sys.executable, str(SCRIPTS_HOME / "download_models.py"), *arguments]
    subprocess.run(command, check=True)


def serve() -> None:
    prepare_directories()
    if os.environ.get("DOWNLOAD_MODELS_ON_START", "1") == "1":
        if os.environ.get("DOWNLOAD_LTX25_MODELS_ON_START", "1") == "1":
            print(
                "Verificando os modelos do LTX 2.3/2.5 "
                "antes de iniciar o ComfyUI..."
            )
        else:
            print(
                "Verificando os modelos do LTX 2.3; "
                "LTX 2.5 desativado por DOWNLOAD_LTX25_MODELS_ON_START=0."
            )
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
