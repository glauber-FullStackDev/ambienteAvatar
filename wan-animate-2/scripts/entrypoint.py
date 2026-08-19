#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
MODEL_ROOT = Path(os.environ.get("COMFYUI_MODELS", "/models"))
SCRIPTS_HOME = Path(__file__).resolve().parent
DEFAULT_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_WORKFLOW", "/opt/defaults/workflows/wan-animate-2-int8.json"
    )
)
EXTRA_MODEL_PATHS = Path(
    os.environ.get(
        "EXTRA_MODEL_PATHS", "/opt/wan-animate-2/extra_model_paths.yaml"
    )
)


def prepare_directories() -> None:
    for relative in (
        "clip_vision",
        "diffusion_models",
        "loras",
        "text_encoders",
        "vae",
    ):
        (MODEL_ROOT / relative).mkdir(parents=True, exist_ok=True)
    for relative in ("input", "output", "user/default/workflows"):
        (COMFYUI_HOME / relative).mkdir(parents=True, exist_ok=True)


def disconnect_input(graph: dict, node_type: str, input_name: str) -> None:
    removed_link_ids: set[int] = set()
    for node in graph.get("nodes", []):
        if node.get("type") != node_type:
            continue
        for node_input in node.get("inputs", []):
            if node_input.get("name") == input_name and node_input.get("link") is not None:
                removed_link_ids.add(node_input["link"])
                node_input["link"] = None

    if not removed_link_ids:
        return
    graph["links"] = [
        link for link in graph.get("links", []) if link[0] not in removed_link_ids
    ]
    for node in graph.get("nodes", []):
        for output in node.get("outputs", []):
            links = output.get("links")
            if isinstance(links, list):
                output["links"] = [
                    link for link in links if link not in removed_link_ids
                ]


def customize_workflow(workflow: dict) -> None:
    # O audio sera produzido pelo InfiniteTalk. Este fluxo entrega somente a
    # base visual e nao copia a faixa do video-guia. O subgrafo opcional de
    # comparacao permanece intacto e desativado abaixo.
    disconnect_input(workflow, "CreateVideo", "audio")

    for node in workflow.get("nodes", []):
        widget_inputs = [
            item for item in node.get("inputs", []) if item.get("widget") is not None
        ]
        values = node.get("widgets_values")
        if isinstance(values, list):
            for index, item in enumerate(widget_inputs):
                if item.get("label") == "cache_device" and index < len(values):
                    # Mantem o cache INT8 em RAM para poupar VRAM. Isso troca
                    # velocidade por capacidade sem alterar a precisao do DiT.
                    values[index] = "cpu"
        if node.get("id") == 246 and node.get("type") == "SaveVideo":
            if isinstance(values, list) and values:
                values[0] = "video/wan-animate-2-int8"
        if node.get("id") in {291, 292}:
            # A comparacao lado a lado do template oficial continua no grafo,
            # mas fica desativada para nao renderizar um segundo video por fila.
            node["mode"] = 2


def seed_workflow() -> None:
    target = COMFYUI_HOME / "user/default/workflows/wan-animate-2-int8-docker.json"
    if target.exists():
        return
    workflow = json.loads(DEFAULT_WORKFLOW.read_text(encoding="utf-8"))
    customize_workflow(workflow)
    target.write_text(
        json.dumps(workflow, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Workflow inicial criado em {target}")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_downloader(verify_only: bool = False) -> int:
    command = [sys.executable, str(SCRIPTS_HOME / "download_models.py")]
    if verify_only:
        command.append("--verify-only")
    return subprocess.call(command)


def serve(extra_args: list[str]) -> None:
    prepare_directories()
    seed_workflow()
    if env_flag("DOWNLOAD_MODELS_ON_START"):
        result = run_downloader()
        if result != 0:
            raise SystemExit(result)
    port = os.environ.get("COMFYUI_PORT", "8188")
    env_args = shlex.split(os.environ.get("COMFYUI_ARGS", ""))
    command = [
        sys.executable,
        str(COMFYUI_HOME / "main.py"),
        "--listen",
        "0.0.0.0",
        "--port",
        port,
        "--extra-model-paths-config",
        str(EXTRA_MODEL_PATHS),
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
        raise SystemExit(run_downloader())
    if action == "verify":
        prepare_directories()
        raise SystemExit(run_downloader(verify_only=True))
    os.execvp(action, [action, *extra_args])


if __name__ == "__main__":
    main()
