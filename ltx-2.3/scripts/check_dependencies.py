#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from build_ia2v_talkvid_workflow import validate_hybrid
from build_ia2v_best_face_workflow import validate_best_face
from build_ltx25_ia2v_workflow import (
    REQUIRED_MODELS as LTX25_REQUIRED_MODELS,
    validate_ia2v as validate_ltx25_ia2v,
)


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
IA2V_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v.json",
    )
)
ID_LORA_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_ID_LORA_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_id_lora.json",
    )
)
IA2V_TALKVID_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_TALKVID_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_talkvid.json",
    )
)
IA2V_BEST_FACE_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_BEST_FACE_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_best_face.json",
    )
)
LTX25_IA2V_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_LTX25_IA2V_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_5_ia2v_distilled_8steps.json",
    )
)
IA2V_WORKFLOW_SHA256 = (
    "7823a703f472d9c5e6f82c462235ff89a0fa14752ec1fd947c4422cf53e47685"
)
ID_LORA_WORKFLOW_SHA256 = (
    "fcffe421129bac16b4f0655e54130d633280cdaf6949e145221e7090be42151f"
)
COMMON_MODELS = {
    "ltx-2.3-22b-dev-fp8.safetensors",
    "gemma_3_12B_it_fp4_mixed.safetensors",
    "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}
IA2V_MODELS = COMMON_MODELS | {
    "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
}
ID_LORA_MODELS = COMMON_MODELS | {
    "ltx-2.3-id-lora-talkvid-3k.safetensors",
}
IA2V_TALKVID_MODELS = IA2V_MODELS | ID_LORA_MODELS
IA2V_BEST_FACE_MODELS = IA2V_MODELS | {
    "Best_FaceID_v1.0_LoRA.safetensors",
}
COMMON_NODE_TYPES = {
    "CheckpointLoaderSimple",
    "CreateVideo",
    "EmptyLTXVLatentVideo",
    "LTXAVTextEncoderLoader",
    "LTXVAudioVAEDecode",
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
}
IA2V_NODE_TYPES = COMMON_NODE_TYPES | {
    "LTXVAudioVAEEncode",
    "TextGenerateLTX2Prompt",
}
ID_LORA_NODE_TYPES = COMMON_NODE_TYPES | {
    "LTXVEmptyLatentAudio",
    "LTXVReferenceAudio",
}
IA2V_TALKVID_NODE_TYPES = IA2V_NODE_TYPES | {
    "LTXVReferenceAudio",
}
IA2V_BEST_FACE_NODE_TYPES = IA2V_NODE_TYPES | {
    "LTXIdentityOverlapConditioning",
}
LTX25_IA2V_NODE_TYPES = {
    "CLIPLoader",
    "CLIPTextEncode",
    "CreateVideo",
    "EmptyLTXVLatentVideo",
    "LTXVAudioVAEEncode",
    "LTXVConcatAVLatent",
    "LTXVConditioning",
    "LTXVDualCFGGuider",
    "LTXVImgToVideoInplace",
    "LTXVLatentUpsampler",
    "LTXVPreprocess",
    "LTXVSeparateAVLatent",
    "LatentUpscaleModelLoader",
    "ManualSigmas",
    "SetLatentNoiseMask",
    "SolidMask",
    "TrimAudioDuration",
    "UNETLoader",
    "VAEDecodeTiled",
    "VAELoader",
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


def validate_workflow(
    label: str,
    path: Path,
    expected_sha256: str | None,
    expected_models: set[str],
    expected_node_types: set[str],
    failures: list[str],
) -> None:
    if not path.is_file():
        failures.append(f"workflow {label} ausente: {path}")
        return
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        failures.append(f"checksum inesperado para o workflow {label}: {path}")
    workflow = json.loads(path.read_text(encoding="utf-8"))
    if workflow.get("version") != 0.4:
        failures.append(f"versao do workflow {label} diferente de 0.4")
    node_types = {node.get("type") for node in nested_nodes(workflow)}
    missing_nodes = expected_node_types - node_types
    if missing_nodes:
        failures.append(
            f"nodes ausentes no workflow {label}: "
            + ", ".join(sorted(missing_nodes))
        )
    strings = all_strings(workflow)
    missing_models = expected_models - strings
    if missing_models:
        failures.append(
            f"modelos ausentes no workflow {label}: "
            + ", ".join(sorted(missing_models))
        )


def main() -> None:
    failures: list[str] = []
    for required in (COMFYUI_HOME / "main.py", COMFYUI_HOME / "requirements.txt"):
        if not required.is_file():
            failures.append(f"arquivo do ComfyUI ausente: {required}")
    bfs_node = COMFYUI_HOME / "custom_nodes/ComfyUI-BFSNodes/__init__.py"
    if not bfs_node.is_file():
        failures.append(f"custom node BFS ausente: {bfs_node}")
    validate_workflow(
        "IA2V",
        IA2V_WORKFLOW,
        IA2V_WORKFLOW_SHA256,
        IA2V_MODELS,
        IA2V_NODE_TYPES,
        failures,
    )
    validate_workflow(
        "ID-LoRA",
        ID_LORA_WORKFLOW,
        ID_LORA_WORKFLOW_SHA256,
        ID_LORA_MODELS,
        ID_LORA_NODE_TYPES,
        failures,
    )
    validate_workflow(
        "IA2V + TalkVid",
        IA2V_TALKVID_WORKFLOW,
        None,
        IA2V_TALKVID_MODELS,
        IA2V_TALKVID_NODE_TYPES,
        failures,
    )
    if IA2V_TALKVID_WORKFLOW.is_file():
        try:
            validate_hybrid(
                json.loads(IA2V_TALKVID_WORKFLOW.read_text(encoding="utf-8"))
            )
        except Exception as error:
            failures.append(f"workflow IA2V + TalkVid invalido: {error}")
    validate_workflow(
        "IA2V + Best Face-ID",
        IA2V_BEST_FACE_WORKFLOW,
        None,
        IA2V_BEST_FACE_MODELS,
        IA2V_BEST_FACE_NODE_TYPES,
        failures,
    )
    if IA2V_BEST_FACE_WORKFLOW.is_file():
        try:
            validate_best_face(
                json.loads(IA2V_BEST_FACE_WORKFLOW.read_text(encoding="utf-8"))
            )
        except Exception as error:
            failures.append(f"workflow IA2V + Best Face-ID invalido: {error}")
    validate_workflow(
        "LTX-2.5 IA2V Distilled 8 Steps",
        LTX25_IA2V_WORKFLOW,
        None,
        LTX25_REQUIRED_MODELS,
        LTX25_IA2V_NODE_TYPES,
        failures,
    )
    if LTX25_IA2V_WORKFLOW.is_file():
        try:
            validate_ltx25_ia2v(
                json.loads(LTX25_IA2V_WORKFLOW.read_text(encoding="utf-8"))
            )
        except Exception as error:
            failures.append(f"workflow LTX-2.5 IA2V invalido: {error}")

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
    print(
        "Dependencias e workflows "
        "LTX-2.3 e LTX-2.5 IA2V validados."
    )


if __name__ == "__main__":
    main()
