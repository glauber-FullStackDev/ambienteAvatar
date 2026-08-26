#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v.json",
    )
)
WORKFLOW_SHA256 = "7823a703f472d9c5e6f82c462235ff89a0fa14752ec1fd947c4422cf53e47685"
EXPECTED_MODELS = {
    "ltx-2.3-22b-dev-fp8.safetensors",
    "gemma_3_12B_it_fp4_mixed.safetensors",
    "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
    "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}
EXPECTED_NODE_TYPES = {
    "CheckpointLoaderSimple",
    "CreateVideo",
    "EmptyLTXVLatentVideo",
    "LTXAVTextEncoderLoader",
    "LTXVAudioVAEDecode",
    "LTXVAudioVAEEncode",
    "LTXVAudioVAELoader",
    "LTXVConcatAVLatent",
    "LTXVConditioning",
    "LTXVCropGuides",
    "LTXVImgToVideoInplace",
    "LTXVLatentUpsampler",
    "LTXVPreprocess",
    "LTXVSeparateAVLatent",
    "LatentUpscaleModelLoader",
    "SaveVideo",
    "TextGenerateLTX2Prompt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def nested_nodes(workflow: dict) -> list[dict]:
    nodes = list(workflow.get("nodes", []))
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        nodes.extend(subgraph.get("nodes", []))
    return nodes


def all_strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found.update(all_strings(item))
        return found
    if isinstance(value, dict):
        found = set()
        for item in value.values():
            found.update(all_strings(item))
        return found
    return set()


def main() -> None:
    failures: list[str] = []
    for required in (COMFYUI_HOME / "main.py", COMFYUI_HOME / "requirements.txt"):
        if not required.is_file():
            failures.append(f"arquivo do ComfyUI ausente: {required}")
    if not WORKFLOW.is_file():
        failures.append(f"workflow ausente: {WORKFLOW}")
    else:
        if sha256(WORKFLOW) != WORKFLOW_SHA256:
            failures.append(f"checksum inesperado para {WORKFLOW}")
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        if workflow.get("version") != 0.4:
            failures.append("versao do workflow oficial diferente de 0.4")
        node_types = {node.get("type") for node in nested_nodes(workflow)}
        missing_nodes = EXPECTED_NODE_TYPES - node_types
        if missing_nodes:
            failures.append(
                "nodes LTX ausentes no workflow: " + ", ".join(sorted(missing_nodes))
            )
        strings = all_strings(workflow)
        missing_models = EXPECTED_MODELS - strings
        if missing_models:
            failures.append(
                "modelos ausentes no workflow: " + ", ".join(sorted(missing_models))
            )

    try:
        import huggingface_hub  # noqa: F401
        import torch
    except Exception as error:
        failures.append(f"dependencia Python nao importou: {error}")
    else:
        print(f"PyTorch: {torch.__version__}; CUDA da build: {torch.version.cuda}")

    if failures:
        for failure in failures:
            print(f"[FALHA] {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("Dependencias e workflow oficial do LTX 2.3 validados.")


if __name__ == "__main__":
    main()
