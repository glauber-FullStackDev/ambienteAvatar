#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import find_spec
import json
from pathlib import Path


required_modules = (
    "aiohttp",
    "av",
    "huggingface_hub",
    "safetensors",
    "torch",
    "torchvision",
)
missing_modules = [name for name in required_modules if find_spec(name) is None]
if missing_modules:
    raise SystemExit(f"Dependencias Python ausentes: {', '.join(missing_modules)}")

comfyui_home = Path("/opt/ComfyUI")
wan_nodes = comfyui_home / "comfy_extras/nodes_wan.py"
workflow_path = Path("/opt/defaults/workflows/wan-animate-2-int8.json")
required_paths = (
    comfyui_home / "main.py",
    wan_nodes,
    workflow_path,
    Path("/opt/wan-animate-2/extra_model_paths.yaml"),
    Path("/opt/wan-animate-2/scripts/download_models.py"),
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise SystemExit(f"Arquivos da imagem ausentes: {', '.join(missing_paths)}")

node_source = wan_nodes.read_text(encoding="utf-8")
for native_node in ("WanAnimate2ToVideo", "WanAnimate2Cache"):
    if native_node not in node_source:
        raise SystemExit(f"Node nativo {native_node} ausente no ComfyUI fixado")

workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
workflow_text = json.dumps(workflow)
required_workflow_values = (
    "wan_animate_2_int8_convrot.safetensors",
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "WanAnimate2ToVideo",
    "WanAnimate2Cache",
)
for value in required_workflow_values:
    if value not in workflow_text:
        raise SystemExit(f"Workflow oficial nao contem {value}")

print("OK: ComfyUI nativo, workflow oficial e Wan-Animate-2 INT8 presentes")
