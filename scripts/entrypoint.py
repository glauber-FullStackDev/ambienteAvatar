#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
SCRIPTS_HOME = Path(__file__).resolve().parent
DEFAULT_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_WORKFLOW",
        "/opt/defaults/workflows/longcat-avatar1.5.json",
    )
)


def prepare_directories() -> None:
    paths = (
        "input",
        "output",
        "models/audio_encoders",
        "models/clip",
        "models/diffusion_models",
        "models/latentsync",
        "models/liveportrait",
        "models/longcat",
        "models/loras",
        "models/ultralytics",
        "models/vae",
        "user/default/workflows",
    )
    for relative in paths:
        (COMFYUI_HOME / relative).mkdir(parents=True, exist_ok=True)


def seed_workflow() -> None:
    target = COMFYUI_HOME / "user/default/workflows/longcat-avatar1.5-docker.json"
    if target.exists() or not DEFAULT_WORKFLOW.exists():
        return

    workflow = json.loads(DEFAULT_WORKFLOW.read_text(encoding="utf-8"))
    weight_mode = os.environ.get("DEFAULT_WEIGHT_MODE", "official_int8_sharded")
    for node in workflow.get("nodes", []):
        if node.get("type") != "LongCat_Video_SM_Model":
            continue
        values = node.get("widgets_values", [])
        if len(values) >= 3:
            values[0] = weight_mode
            values[1] = "sdpa"
            values[2] = False
    target.write_text(json.dumps(workflow, ensure_ascii=False), encoding="utf-8")
    print(f"Workflow inicial criado em {target}")


def run_downloader(verify_only: bool = False) -> None:
    command = [sys.executable, str(SCRIPTS_HOME / "download_models.py")]
    if verify_only:
        command.append("--verify-only")
    raise SystemExit(subprocess.call(command))


def serve(extra_args: list[str]) -> None:
    prepare_directories()
    seed_workflow()
    port = os.environ.get("COMFYUI_PORT", "8188")
    env_args = shlex.split(os.environ.get("COMFYUI_ARGS", ""))
    command = [
        sys.executable,
        str(COMFYUI_HOME / "main.py"),
        "--listen",
        "0.0.0.0",
        "--port",
        port,
        *env_args,
        *extra_args,
    ]
    os.execvp(command[0], command)


def main() -> None:
    action, *extra_args = sys.argv[1:] or ["serve"]
    if action == "serve":
        serve(extra_args)
    if action == "download-models":
        prepare_directories()
        run_downloader()
    if action == "verify":
        prepare_directories()
        run_downloader(verify_only=True)
    os.execvp(action, [action, *extra_args])


if __name__ == "__main__":
    main()
