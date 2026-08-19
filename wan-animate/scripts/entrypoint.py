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
        "DEFAULT_WORKFLOW", "/opt/defaults/workflows/wan-animate-preprocess.json"
    )
)

MODEL_FILES = {
    "e4m3fn": "WanVideo/2_2/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
    "e5m2": "WanVideo/2_2/Wan2_2-Animate-14B_fp8_e5m2_scaled_KJ.safetensors",
}


def model_variant() -> str:
    value = os.environ.get("MODEL_VARIANT", "e4m3fn").strip().lower()
    if value not in MODEL_FILES:
        choices = ", ".join(MODEL_FILES)
        raise SystemExit(f"MODEL_VARIANT invalido: {value!r}; use {choices}")
    return value


def prepare_directories() -> None:
    paths = (
        "input",
        "output",
        "models/clip_vision",
        "models/detection/onnx",
        "models/diffusion_models/WanVideo/2_2",
        "models/loras/WanVideo/Lightx2v",
        "models/sam2",
        "models/text_encoders",
        "models/vae/wanvideo",
        "user/default/workflows",
    )
    for relative in paths:
        (COMFYUI_HOME / relative).mkdir(parents=True, exist_ok=True)


def set_widget_values(workflow: dict, node_type: str, desired: list[object]) -> None:
    for node in workflow.get("nodes", []):
        if node.get("type") != node_type:
            continue
        values = node.get("widgets_values")
        if not isinstance(values, list):
            continue
        for index, value in enumerate(desired):
            if value is not None and index < len(values):
                values[index] = value


def disconnect_inputs(workflow: dict, node_type: str, input_names: set[str]) -> None:
    removed_link_ids = set()
    for node in workflow.get("nodes", []):
        if node.get("type") != node_type:
            continue
        for node_input in node.get("inputs", []):
            if node_input.get("name") in input_names and node_input.get("link") is not None:
                removed_link_ids.add(node_input["link"])
                node_input["link"] = None

    if not removed_link_ids:
        return
    workflow["links"] = [
        link for link in workflow.get("links", []) if link[0] not in removed_link_ids
    ]
    for node in workflow.get("nodes", []):
        for output in node.get("outputs", []):
            links = output.get("links")
            if isinstance(links, list):
                output["links"] = [link for link in links if link not in removed_link_ids]


def configure_output(workflow: dict) -> None:
    for node in workflow.get("nodes", []):
        if node.get("type") != "VHS_VideoCombine":
            continue
        values = node.get("widgets_values")
        if not isinstance(values, dict) or not values.get("save_output"):
            continue
        values["filename_prefix"] = "wan-animate-base"
        values["trim_to_audio"] = False


def seed_workflow() -> None:
    target = COMFYUI_HOME / "user/default/workflows/wan-animate-base-docker.json"
    if target.exists() or not DEFAULT_WORKFLOW.exists():
        return

    workflow = json.loads(DEFAULT_WORKFLOW.read_text(encoding="utf-8"))
    set_widget_values(
        workflow,
        "WanVideoModelLoader",
        [MODEL_FILES[model_variant()], None, None, None, "sdpa"],
    )
    set_widget_values(
        workflow,
        "WanVideoLoraSelectMulti",
        [
            "WanVideo/WanAnimate_relight_lora_fp16.safetensors",
            None,
            "WanVideo/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        ],
    )
    set_widget_values(
        workflow, "WanVideoVAELoader", ["wanvideo/Wan2_1_VAE_bf16.safetensors"]
    )
    set_widget_values(workflow, "CLIPVisionLoader", ["clip_vision_h.safetensors"])
    set_widget_values(
        workflow, "WanVideoTextEncodeCached", ["umt5-xxl-enc-bf16.safetensors"]
    )
    set_widget_values(
        workflow,
        "OnnxDetectionModelLoader",
        ["vitpose-l-wholebody.onnx", "onnx/yolov10m.onnx", "CUDAExecutionProvider"],
    )

    # Animation-only: imagem define personagem, roupa e fundo; o video fornece
    # apenas pose, gestos e expressoes. Com bg_images/mask conectados, o exemplo
    # muda para o modo de substituicao e preserva o fundo do video original.
    disconnect_inputs(workflow, "WanVideoAnimateEmbeds", {"bg_images", "mask"})
    # O audio final pertence ao InfiniteTalk; esta etapa entrega apenas a base
    # visual, sem copiar a faixa do video-guia para o MP4 gerado.
    disconnect_inputs(workflow, "VHS_VideoCombine", {"audio"})
    configure_output(workflow)

    target.write_text(json.dumps(workflow, ensure_ascii=False), encoding="utf-8")
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
