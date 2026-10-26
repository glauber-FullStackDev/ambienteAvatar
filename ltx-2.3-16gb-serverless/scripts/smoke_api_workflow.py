#!/usr/bin/env python3
"""Falha a build se qualquer class_type do workflow API nao carregar no ComfyUI.

Usa a mesma tecnica do smoke_core_nodes.py da imagem 16 GB (import do ComfyUI
em modo CPU com stub do PromptServer), mas deriva a lista de nodes exigidos do
proprio workflow API compilado — assim qualquer node novo que entre no
workflow pessoal e verificado na build, nao apenas no boot do worker.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
WORKFLOW_PATH = Path(
    os.environ.get(
        "WORKFLOW_PATH",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_16gb_api.json",
    )
)


def main() -> None:
    template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    entries = template["prompt"].values() if "prompt" in template else template.values()
    required = {
        node["class_type"]
        for node in entries
        if isinstance(node, dict) and "class_type" in node
    }

    os.chdir(COMFYUI_HOME)
    sys.path.insert(0, str(COMFYUI_HOME))
    # Docker builders do not expose a GPU. This only affects the import smoke
    # test; the final image starts ComfyUI in normal CUDA mode.
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import server

    if getattr(server.PromptServer, "instance", None) is None:
        prompt_queue = SimpleNamespace(currently_running={}, put=lambda _item: None)
        server.PromptServer.instance = SimpleNamespace(
            routes=server.web.RouteTableDef(),
            prompt_queue=prompt_queue,
            number=0,
            last_node_id=None,
            client_id=None,
            send_sync=lambda *_args, **_kwargs: None,
            # Custom nodes (ex.: KJNodes) tocam app.router.frozen/add_routes
            # durante o import; no smoke test nao existe servidor aiohttp.
            app=SimpleNamespace(
                router=SimpleNamespace(frozen=True),
                add_routes=lambda *_args, **_kwargs: None,
            ),
        )
    import nodes as comfy_nodes

    asyncio.run(
        comfy_nodes.init_extra_nodes(
            init_custom_nodes=True,
            init_api_nodes=True,
        )
    )
    missing = sorted(required - set(comfy_nodes.NODE_CLASS_MAPPINGS))
    if missing:
        raise SystemExit(
            "Nodes exigidos pelo workflow serverless 16 GB nao carregaram: "
            + ", ".join(missing)
        )
    print(f"Smoke da API workflow OK ({len(required)} class_types).")


if __name__ == "__main__":
    main()
