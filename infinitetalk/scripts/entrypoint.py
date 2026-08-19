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
    os.environ.get("DEFAULT_WORKFLOW", "/opt/defaults/workflows/infinitetalk-i2v.json")
)
DEFAULT_V2V_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_V2V_WORKFLOW", "/opt/defaults/workflows/infinitetalk-v2v.json"
    )
)

MODEL_FILES = {
    "q4_k_m": (
        "WanVideo/wan2.1-i2v-14b-480p-Q4_K_M.gguf",
        "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf",
    ),
    "q6_k": (
        "WanVideo/wan2.1-i2v-14b-480p-Q6_K.gguf",
        "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q6_K.gguf",
    ),
    "q8": (
        "WanVideo/wan2.1-i2v-14b-480p-Q8_0.gguf",
        "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf",
    ),
}


def model_quantization() -> str:
    value = os.environ.get("MODEL_QUANTIZATION", "q4_k_m").strip().lower()
    if value not in MODEL_FILES:
        choices = ", ".join(MODEL_FILES)
        raise SystemExit(f"MODEL_QUANTIZATION invalido: {value!r}; use {choices}")
    return value


def prepare_directories() -> None:
    paths = (
        "input",
        "output",
        "models/clip_vision",
        "models/diffusion_models/MelBandRoFormer",
        "models/diffusion_models/WanVideo/InfiniteTalk",
        "models/loras/WanVideo/Lightx2v",
        "models/text_encoders",
        "models/vae/wanvideo",
        "models/wav2vec2",
        "user/default/workflows",
    )
    for relative in paths:
        (COMFYUI_HOME / relative).mkdir(parents=True, exist_ok=True)


def replace_wav2vec_download_node(workflow: dict) -> None:
    nodes = workflow.get("nodes", [])
    download_node = next(
        (node for node in nodes if node.get("type") == "DownloadAndLoadWav2VecModel"),
        None,
    )
    local_node = next(
        (node for node in nodes if node.get("type") == "Wav2VecModelLoader"),
        None,
    )
    if not download_node or not local_node:
        return

    download_id = download_node.get("id")
    local_id = local_node.get("id")
    moved_links = []
    for link in workflow.get("links", []):
        if len(link) >= 2 and link[1] == download_id:
            link[1] = local_id
            moved_links.append(link[0])

    if not moved_links:
        return
    for output in download_node.get("outputs", []):
        output["links"] = None
    for output in local_node.get("outputs", []):
        if output.get("type") == "WAV2VECMODEL":
            output["links"] = moved_links


def patch_workflow(workflow: dict, output_prefix: str | None = None) -> None:
    base_model, infinitetalk_model = MODEL_FILES[model_quantization()]
    replacements = {
        "WanVideoModelLoader": [base_model, None, None, None, "sdpa"],
        "MultiTalkModelLoader": [infinitetalk_model],
        "WanVideoVAELoader": ["wanvideo/Wan2_1_VAE_bf16.safetensors"],
        "CLIPVisionLoader": ["clip_vision_h.safetensors"],
        "Wav2VecModelLoader": ["wav2vec2-chinese-base_fp16.safetensors"],
        "MelBandRoFormerModelLoader": [
            "MelBandRoFormer/MelBandRoformer_fp16.safetensors"
        ],
        "WanVideoLoraSelect": [
            "WanVideo/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
        ],
        "WanVideoTextEncodeCached": ["umt5-xxl-enc-bf16.safetensors"],
    }
    for node in workflow.get("nodes", []):
        desired = replacements.get(node.get("type"))
        values = node.get("widgets_values")
        if desired and isinstance(values, list):
            for index, value in enumerate(desired):
                if value is not None and index < len(values):
                    values[index] = value

        if (
            output_prefix
            and node.get("type") == "VHS_VideoCombine"
            and isinstance(values, dict)
        ):
            values["filename_prefix"] = output_prefix
            values["format"] = "video/h264-mp4"
            values["pix_fmt"] = "yuv420p"
            values["save_output"] = True

    replace_wav2vec_download_node(workflow)


def seed_workflow(source: Path, target_name: str, output_prefix: str | None = None) -> None:
    target = COMFYUI_HOME / "user/default/workflows" / target_name
    if target.exists() or not source.exists():
        return

    workflow = json.loads(source.read_text(encoding="utf-8"))
    patch_workflow(workflow, output_prefix)
    target.write_text(json.dumps(workflow, ensure_ascii=False), encoding="utf-8")
    print(f"Workflow inicial criado em {target}")


def seed_workflows() -> None:
    seed_workflow(DEFAULT_WORKFLOW, "infinitetalk-i2v-docker.json")
    seed_workflow(
        DEFAULT_V2V_WORKFLOW,
        "infinitetalk-v2v-docker.json",
        "InfiniteTalk_V2V",
    )


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
    seed_workflows()
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
