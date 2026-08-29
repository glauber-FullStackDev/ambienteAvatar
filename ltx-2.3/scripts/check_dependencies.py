#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from build_ia2v_talkvid_workflow import validate_hybrid
from build_ia2v_best_face_workflow import validate_best_face
from build_ia2v_ingredients_workflow import (
    INGREDIENTS_NAME,
    validate_ingredients,
)
from build_ingredients_reference_workflows import validate_profile
from build_ltx25_ia2v_workflow import (
    REQUIRED_MODELS as LTX25_REQUIRED_MODELS,
    validate_ia2v as validate_ltx25_ia2v,
)
from build_ia2v_personal_lora_workflow import (
    PERSONAL_LORA_NAME,
    validate_personal_lora,
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
IA2V_PERSONAL_LORA_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_PERSONAL_LORA_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora.json",
    )
)
IA2V_INGREDIENTS_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_INGREDIENTS_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_ingredients.json",
    )
)
IA2V_INGREDIENTS_LEGACY_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_IA2V_INGREDIENTS_LEGACY_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_ingredients_legacy_v2.json",
    )
)
INGREDIENTS_OFFICIAL_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_INGREDIENTS_OFFICIAL_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ingredients_official_single_stage.json",
    )
)
INGREDIENTS_WANGP_I2V_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_INGREDIENTS_WANGP_I2V_WORKFLOW",
        "/opt/defaults/workflows/video_ltx2_3_ingredients_wangp_i2v_15s.json",
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
IA2V_PERSONAL_LORA_MODELS = IA2V_MODELS | {PERSONAL_LORA_NAME}
IA2V_INGREDIENTS_MODELS = IA2V_MODELS | {
    INGREDIENTS_NAME,
}
INGREDIENTS_REFERENCE_MODELS = {
    "ltx-2.3-22b-dev-fp8.safetensors",
    "gemma_3_12B_it_fp4_mixed.safetensors",
    "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
    INGREDIENTS_NAME,
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
IA2V_PERSONAL_LORA_NODE_TYPES = IA2V_NODE_TYPES | {
    "LastFrameFromBatch",
    "SaveImage",
}
IA2V_INGREDIENTS_NODE_TYPES = IA2V_NODE_TYPES | {
    "LTXAddVideoICLoRAGuide",
    "LTXICLoRALoaderModelOnly",
    "RepeatImageBatch",
    "ResizeImageMaskNode",
}
INGREDIENTS_REFERENCE_NODE_TYPES = {
    "CheckpointLoaderSimple",
    "CreateVideo",
    "EmptyLTXVLatentVideo",
    "GetImageSize",
    "GetVideoComponents",
    "GemmaAPITextEncode",
    "LTXAddVideoICLoRAGuide",
    "LTXAVTextEncoderLoader",
    "LTXICLoRALoaderModelOnly",
    "LTXFloatToInt",
    "LTXVAudioVAEDecode",
    "LTXVAudioVAELoader",
    "LTXVConcatAVLatent",
    "LTXVConditioning",
    "LTXVCropGuides",
    "LTXVTiledVAEDecode",
    "RepeatImageBatch",
    "ResizeImageMaskNode",
    "SaveVideo",
}
INGREDIENTS_WANGP_I2V_NODE_TYPES = INGREDIENTS_REFERENCE_NODE_TYPES | {
    "LTXVImgToVideoConditionOnly",
    "LTXVPreprocess",
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
    ltxvideo_node = COMFYUI_HOME / "custom_nodes/ComfyUI-LTXVideo/__init__.py"
    if not ltxvideo_node.is_file():
        failures.append(f"custom node ComfyUI-LTXVideo ausente: {ltxvideo_node}")
    else:
        ltxvideo_source = ltxvideo_node.read_text(encoding="utf-8")
        for node_type in ("LTXAddVideoICLoRAGuide", "LTXICLoRALoaderModelOnly"):
            if node_type not in ltxvideo_source:
                failures.append(
                    f"custom node ComfyUI-LTXVideo sem mapeamento {node_type}"
                )
    ltxvideo_pyramid = (
        COMFYUI_HOME / "custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py"
    )
    if not ltxvideo_pyramid.is_file():
        failures.append(f"arquivo ComfyUI-LTXVideo ausente: {ltxvideo_pyramid}")
    else:
        ltxvideo_pyramid_source = ltxvideo_pyramid.read_text(encoding="utf-8")
        if "pad = F.pad" not in ltxvideo_pyramid_source:
            failures.append("shim Kornia pad ausente no ComfyUI-LTXVideo")
        if "    pad,\n" in ltxvideo_pyramid_source:
            failures.append("import quebrado de pad ainda existe no ComfyUI-LTXVideo")
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
        "IA2V + LoRA pessoal",
        IA2V_PERSONAL_LORA_WORKFLOW,
        None,
        IA2V_PERSONAL_LORA_MODELS,
        IA2V_PERSONAL_LORA_NODE_TYPES,
        failures,
    )
    if IA2V_PERSONAL_LORA_WORKFLOW.is_file():
        try:
            validate_personal_lora(
                json.loads(IA2V_PERSONAL_LORA_WORKFLOW.read_text(encoding="utf-8"))
            )
        except Exception as error:
            failures.append(f"workflow IA2V + LoRA pessoal invalido: {error}")
    validate_workflow(
        "IA2V + IC-LoRA Ingredients",
        IA2V_INGREDIENTS_WORKFLOW,
        None,
        IA2V_INGREDIENTS_MODELS,
        IA2V_INGREDIENTS_NODE_TYPES,
        failures,
    )
    if IA2V_INGREDIENTS_WORKFLOW.is_file():
        try:
            validate_ingredients(
                json.loads(IA2V_INGREDIENTS_WORKFLOW.read_text(encoding="utf-8"))
            )
        except Exception as error:
            failures.append(f"workflow IA2V + IC-LoRA Ingredients invalido: {error}")
    validate_workflow(
        "IA2V + IC-LoRA Ingredients legado schema 2",
        IA2V_INGREDIENTS_LEGACY_WORKFLOW,
        None,
        IA2V_INGREDIENTS_MODELS,
        IA2V_INGREDIENTS_NODE_TYPES,
        failures,
    )
    if IA2V_INGREDIENTS_LEGACY_WORKFLOW.is_file():
        try:
            validate_ingredients(
                json.loads(
                    IA2V_INGREDIENTS_LEGACY_WORKFLOW.read_text(encoding="utf-8")
                ),
                "legacy-v2",
            )
        except Exception as error:
            failures.append(
                f"workflow IA2V + IC-LoRA Ingredients legado invalido: {error}"
            )
    validate_workflow(
        "IC-LoRA Ingredients oficial single-stage",
        INGREDIENTS_OFFICIAL_WORKFLOW,
        None,
        INGREDIENTS_REFERENCE_MODELS,
        INGREDIENTS_REFERENCE_NODE_TYPES,
        failures,
    )
    if INGREDIENTS_OFFICIAL_WORKFLOW.is_file():
        try:
            validate_profile(
                json.loads(INGREDIENTS_OFFICIAL_WORKFLOW.read_text(encoding="utf-8")),
                "official",
            )
        except Exception as error:
            failures.append(f"workflow Ingredients oficial invalido: {error}")
    validate_workflow(
        "IC-LoRA Ingredients WanGP I2V 15s",
        INGREDIENTS_WANGP_I2V_WORKFLOW,
        None,
        INGREDIENTS_REFERENCE_MODELS,
        INGREDIENTS_WANGP_I2V_NODE_TYPES,
        failures,
    )
    if INGREDIENTS_WANGP_I2V_WORKFLOW.is_file():
        try:
            validate_profile(
                json.loads(
                    INGREDIENTS_WANGP_I2V_WORKFLOW.read_text(encoding="utf-8")
                ),
                "wangp-i2v-15s",
            )
        except Exception as error:
            failures.append(f"workflow Ingredients WanGP I2V invalido: {error}")
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
        import torchaudio
        import torchvision
    except Exception as error:
        failures.append(f"dependencia Python nao importou: {error}")
    else:
        print(f"PyTorch: {torch.__version__}; CUDA da build: {torch.version.cuda}")
        print(f"TorchVision: {torchvision.__version__}")
        print(f"TorchAudio: {torchaudio.__version__}")

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
