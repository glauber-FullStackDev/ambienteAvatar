#!/usr/bin/env python3
"""Valida os workflows GGUF 16 GB gerados no build da imagem."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from gguf_convert import (
    GGUF_SCHEMA_MARKER,
    GGUF_SCHEMA_VERSION,
    PERSONAL_LORA_FILE,
    validate_gguf,
)


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
IA2V_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_PERSONAL_LORA_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_16gb.json",
    )
)
A2V_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_A2V_PERSONAL_LORA_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_a2v_personal_lora_16gb.json",
    )
)
PERSONAL_LORA_ASSET = Path("/opt/ltx23-assets/glauberavatar.safetensors")


def validate_structure(workflow: dict, label: str, *, expect_a2v: bool) -> None:
    if workflow.get("version") != 0.4:
        raise SystemExit(f"Workflow {label} fora do schema 0.4")
    extra = workflow.get("extra", {})
    if extra.get(GGUF_SCHEMA_MARKER) != GGUF_SCHEMA_VERSION:
        raise SystemExit(f"Workflow {label} sem marcador GGUF 16 GB")
    if expect_a2v and extra.get("ltx23_a2v_personal_lora_schema") is None:
        raise SystemExit(f"Workflow {label} sem marcador A2V")
    if not expect_a2v and extra.get("ltx23_ia2v_personal_lora_schema") is None:
        raise SystemExit(f"Workflow {label} sem marcador IA2V pessoal")

    subgraph = workflow["definitions"]["subgraphs"][0]
    node_types = {node.get("type") for node in subgraph["nodes"]}
    missing = {"UnetLoaderGGUF", "DualCLIPLoaderGGUF", "VAELoaderKJ"} - node_types
    if missing:
        raise SystemExit(f"Nodes GGUF ausentes em {label}: {sorted(missing)}")
    forbidden = {
        "CheckpointLoaderSimple",
        "LTXAVTextEncoderLoader",
        "LTXVAudioVAELoader",
    }
    if forbidden & node_types:
        raise SystemExit(
            f"Loaders do path 48 GB em {label}: {sorted(forbidden & node_types)}"
        )

    personal_loaders = [
        node
        for node in subgraph["nodes"]
        if node.get("type") == "LoraLoaderModelOnly"
        and PERSONAL_LORA_FILE in (node.get("widgets_values") or [])
    ]
    if len(personal_loaders) != 1:
        raise SystemExit(f"LoRA pessoal ausente/duplicado em {label}")
    if len([n for n in subgraph["nodes"] if n.get("type") == "LastFrameFromBatch"]) != 1:
        raise SystemExit(f"Extrator de ultimo frame ausente em {label}")
    if len([n for n in workflow["nodes"] if n.get("type") == "SaveImage"]) != 1:
        raise SystemExit(f"SaveImage do ultimo frame ausente em {label}")
    print(f"Workflow {label} OK: {path}")


def main() -> None:
    if not IA2V_WORKFLOW.is_file():
        raise SystemExit(f"Workflow IA2V 16 GB ausente: {IA2V_WORKFLOW}")
    if not A2V_WORKFLOW.is_file():
        raise SystemExit(f"Workflow A2V 16 GB ausente: {A2V_WORKFLOW}")
    if not PERSONAL_LORA_ASSET.is_file():
        # No build da imagem o LoRA esta em /opt/ltx23-assets; localmente
        # (validacao fora do container) o arquivo fica em assets/.
        local = Path(__file__).resolve().parent.parent / "assets/glauberavatar.safetensors"
        if not (PERSONAL_LORA_ASSET.is_file() or local.is_file()):
            raise SystemExit("LoRA pessoal incorporado ausente")

    ia2v = json.loads(IA2V_WORKFLOW.read_text(encoding="utf-8"))
    validate_gguf(ia2v)
    validate_workflow(IA2V_WORKFLOW, "IA2V GGUF 16 GB", expect_a2v=False)

    a2v = json.loads(A2V_WORKFLOW.read_text(encoding="utf-8"))
    validate_gguf(a2v)
    validate_workflow(A2V_WORKFLOW, "A2V GGUF 16 GB", expect_a2v=True)
    if any(node.get("type") == "LoadImage" for node in a2v["nodes"]):
        raise SystemExit("Workflow A2V expoe LoadImage")
    print("check_dependencies (16 GB) concluido.")


def validate_workflow(path: Path, label: str, *, expect_a2v: bool) -> None:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    validate_gguf(workflow)
    subgraph = workflow["definitions"]["subgraphs"][0]
    node_types = {node.get("type") for node in subgraph["nodes"]}
    if not expect_a2v:
        if "LTXVImgToVideoInplace" not in node_types:
            raise SystemExit(f"Workflow {label} sem branch I2V")
    else:
        if "LTXVImgToVideoInplace" in node_types:
            raise SystemExit(f"Workflow {label} nao deveria ter branch I2V")
    print(f"Estrutura {label} conferida")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
