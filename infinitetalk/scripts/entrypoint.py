#!/usr/bin/env python3
from __future__ import annotations

import copy
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
        "DEFAULT_V2V_WORKFLOW",
        "/opt/defaults/workflows/infinitetalk-v2v-docker.json",
    )
)
DEFAULT_V2V_LATENTSYNC_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_V2V_LATENTSYNC_WORKFLOW",
        "/opt/defaults/workflows/infinitetalk-v2v-latentsync16-docker.json",
    )
)
DEFAULT_V2V_LATENTSYNC_STABLE_WORKFLOW = Path(
    os.environ.get(
        "DEFAULT_V2V_LATENTSYNC_STABLE_WORKFLOW",
        "/opt/defaults/workflows/infinitetalk-v2v-latentsync16-stable-docker.json",
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
    value = os.environ.get("MODEL_QUANTIZATION", "q8").strip().lower()
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
        "models/FlashVSR",
        "models/loras/WanVideo/Lightx2v",
        "models/latentsync/vae",
        "models/latentsync/whisper",
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


def keep_generated_video_only(workflow: dict) -> bool:
    """Bypass the V2V before/after concat in the saved video output."""
    nodes = {
        node.get("id"): node
        for node in workflow.get("nodes", [])
        if node.get("id") is not None
    }
    links = {
        link[0]: link
        for link in workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 5
    }

    def has_upstream_type(node_id: object, node_type: str, seen: set[object]) -> bool:
        if node_id in seen:
            return False
        seen.add(node_id)
        node = nodes.get(node_id)
        if not node:
            return False
        if node.get("type") == node_type:
            return True
        for node_input in node.get("inputs", []):
            link = links.get(node_input.get("link"))
            if link and has_upstream_type(link[1], node_type, seen):
                return True
        return False

    changed = False
    for output_node in workflow.get("nodes", []):
        if output_node.get("type") != "VHS_VideoCombine":
            continue
        image_input = next(
            (
                node_input
                for node_input in output_node.get("inputs", [])
                if node_input.get("name") == "images"
            ),
            None,
        )
        output_link = links.get(image_input.get("link")) if image_input else None
        concat_node = nodes.get(output_link[1]) if output_link else None
        if not concat_node or concat_node.get("type") != "ImageConcatMulti":
            continue

        generated_link = None
        for concat_input in concat_node.get("inputs", []):
            candidate = links.get(concat_input.get("link"))
            if candidate and has_upstream_type(candidate[1], "WanVideoDecode", set()):
                generated_link = candidate
                break
        if not generated_link:
            continue

        old_source = nodes.get(output_link[1])
        if old_source:
            for output in old_source.get("outputs", []):
                output_links = output.get("links")
                if isinstance(output_links, list) and output_link[0] in output_links:
                    output["links"] = [
                        link_id for link_id in output_links if link_id != output_link[0]
                    ]

        output_link[1] = generated_link[1]
        output_link[2] = generated_link[2]
        generated_source = nodes.get(generated_link[1])
        if generated_source:
            source_output = generated_source.get("outputs", [])[generated_link[2]]
            source_links = source_output.get("links")
            if not isinstance(source_links, list):
                source_links = []
            if output_link[0] not in source_links:
                source_output["links"] = [*source_links, output_link[0]]
        concat_node["mode"] = 2
        changed = True

    return changed


def env_number(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise SystemExit(f"{name} invalido: {raw_value!r}; informe um numero") from error
    if not minimum <= value <= maximum:
        raise SystemExit(
            f"{name} invalido: {value}; use um valor entre {minimum} e {maximum}"
        )
    return value


def tune_v2v_quality(workflow: dict) -> bool:
    """Apply conservative, configurable V2V quality defaults."""
    audio_scale = env_number("INFINITETALK_AUDIO_SCALE", 1.5, 0.0, 4.0)
    audio_cfg_scale = env_number("INFINITETALK_AUDIO_CFG_SCALE", 1.2, 1.0, 4.0)
    output_crf = round(env_number("INFINITETALK_OUTPUT_CRF", 16, 0, 51))
    changed = False

    for node in workflow.get("nodes", []):
        values = node.get("widgets_values")
        if node.get("type") == "MultiTalkWav2VecEmbeds" and isinstance(values, list):
            # Widget order from the pinned WanVideoWrapper: normalize, frames,
            # fps, audio strength, audio CFG, multi-audio mode.
            for index, desired in ((3, audio_scale), (4, audio_cfg_scale)):
                if index < len(values) and values[index] != desired:
                    values[index] = desired
                    changed = True
        if node.get("type") == "VHS_VideoCombine" and isinstance(values, dict):
            if values.get("crf") != output_crf:
                values["crf"] = output_crf
                changed = True

    return changed


def route_original_audio_to_output(workflow: dict) -> bool:
    """Send the untouched input audio to the final LatentSync video mux."""
    nodes = {
        node.get("id"): node
        for node in workflow.get("nodes", [])
        if node.get("id") is not None
    }
    links = {
        link[0]: link
        for link in workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 6
    }
    input_audio_node = next(
        (
            node
            for node in workflow.get("nodes", [])
            if node.get("type") == "GetNode"
            and isinstance(node.get("widgets_values"), list)
            and node["widgets_values"]
            and node["widgets_values"][0] == "input_audio"
        ),
        None,
    )
    if not input_audio_node:
        return False

    input_audio_outputs = input_audio_node.get("outputs", [])
    if not input_audio_outputs:
        return False
    input_audio_output = input_audio_outputs[0]
    changed = False

    for combine in workflow.get("nodes", []):
        if combine.get("type") != "VHS_VideoCombine":
            continue
        audio_input = next(
            (
                node_input
                for node_input in combine.get("inputs", [])
                if node_input.get("name") == "audio"
            ),
            None,
        )
        audio_link = links.get(audio_input.get("link")) if audio_input else None
        source_node = nodes.get(audio_link[1]) if audio_link else None
        if not source_node or source_node.get("type") not in {
            "LatentSyncNode",
            "LatentSyncStableNode",
        }:
            continue

        for output in source_node.get("outputs", []):
            output_links = output.get("links")
            if isinstance(output_links, list) and audio_link[0] in output_links:
                remaining = [
                    link_id for link_id in output_links if link_id != audio_link[0]
                ]
                output["links"] = remaining or None

        audio_link[1] = input_audio_node["id"]
        audio_link[2] = 0
        output_links = input_audio_output.get("links")
        if not isinstance(output_links, list):
            output_links = []
        if audio_link[0] not in output_links:
            input_audio_output["links"] = [*output_links, audio_link[0]]
        changed = True

    return changed


def patch_workflow(
    workflow: dict,
    output_prefix: str | None = None,
    generated_only: bool = False,
) -> None:
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
    if generated_only:
        keep_generated_video_only(workflow)
        tune_v2v_quality(workflow)


def seed_workflow(
    source: Path,
    target_name: str,
    output_prefix: str | None = None,
    generated_only: bool = False,
    preserve_source: bool = False,
) -> None:
    target = COMFYUI_HOME / "user/default/workflows" / target_name
    if preserve_source:
        if target.exists():
            return
        if not source.exists():
            return
        source_text = source.read_text(encoding="utf-8")
        json.loads(source_text)
        target.write_text(source_text, encoding="utf-8")
        print(f"Workflow inicial preservado em {target}")
        return

    if target.exists():
        if not generated_only:
            return
        workflow = json.loads(target.read_text(encoding="utf-8"))
        output_changed = keep_generated_video_only(workflow)
        quality_changed = tune_v2v_quality(workflow)
        if output_changed or quality_changed:
            target.write_text(json.dumps(workflow, ensure_ascii=False), encoding="utf-8")
            print(f"Workflow V2V atualizado em {target}")
        return
    if not source.exists():
        return

    workflow = json.loads(source.read_text(encoding="utf-8"))
    patch_workflow(workflow, output_prefix, generated_only)
    target.write_text(json.dumps(workflow, ensure_ascii=False), encoding="utf-8")
    print(f"Workflow inicial criado em {target}")


def inject_missing_prompt_defaults(source: Path, target_name: str) -> None:
    """Fill only empty prompt fields in an existing seeded workflow."""
    target = COMFYUI_HOME / "user/default/workflows" / target_name
    if not source.exists() or not target.exists():
        return

    source_workflow = json.loads(source.read_text(encoding="utf-8"))
    target_workflow = json.loads(target.read_text(encoding="utf-8"))

    def prompt_node(workflow: dict) -> dict | None:
        return next(
            (
                node
                for node in workflow.get("nodes", [])
                if node.get("type") == "WanVideoTextEncodeCached"
            ),
            None,
        )

    source_node = prompt_node(source_workflow)
    target_node = prompt_node(target_workflow)
    if not source_node or not target_node:
        return

    source_values = source_node.get("widgets_values")
    target_values = target_node.get("widgets_values")
    source_named = source_node.get("widgets_values_named", {})
    target_named = target_node.setdefault("widgets_values_named", {})
    if not isinstance(source_values, list) or not isinstance(target_values, list):
        return

    changed = False
    for index, field in ((2, "positive_prompt"), (3, "negative_prompt")):
        source_prompt = source_named.get(field) or source_values[index]
        current_prompt = target_named.get(field) or target_values[index]
        if source_prompt and not str(current_prompt).strip():
            target_values[index] = source_prompt
            target_named[field] = source_prompt
            changed = True

    if changed:
        target.write_text(
            json.dumps(target_workflow, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Prompts InfiniteTalk + LatentSync adicionados em {target}")


def upgrade_stable_workflow(source: Path, target_name: str) -> None:
    """Upgrade the versioned Stable preset parameters without replacing its graph."""
    target = COMFYUI_HOME / "user/default/workflows" / target_name
    if not source.exists() or not target.exists():
        return

    source_workflow = json.loads(source.read_text(encoding="utf-8"))
    target_workflow = json.loads(target.read_text(encoding="utf-8"))

    def stable_node(workflow: dict) -> dict | None:
        return next(
            (
                node
                for node in workflow.get("nodes", [])
                if node.get("type") == "LatentSyncStableNode"
            ),
            None,
        )

    source_node = stable_node(source_workflow)
    target_node = stable_node(target_workflow)
    if not source_node or not target_node:
        return

    source_properties = source_node.get("properties", {})
    target_properties = target_node.setdefault("properties", {})
    source_schema = int(source_properties.get("infinitetalk_stable_schema", 1))
    target_schema = int(target_properties.get("infinitetalk_stable_schema", 0))
    if target_schema >= source_schema:
        return

    target_node["widgets_values"] = list(source_node.get("widgets_values", []))
    source_named = source_node.get("widgets_values_named")
    if isinstance(source_named, dict):
        target_node["widgets_values_named"] = dict(source_named)
    target_properties["infinitetalk_stable_schema"] = source_schema

    source_multitalk = next(
        (
            node
            for node in source_workflow.get("nodes", [])
            if node.get("type") == "MultiTalkWav2VecEmbeds"
        ),
        None,
    )
    target_multitalk = next(
        (
            node
            for node in target_workflow.get("nodes", [])
            if node.get("type") == "MultiTalkWav2VecEmbeds"
        ),
        None,
    )
    if source_multitalk and target_multitalk:
        target_multitalk["widgets_values"] = list(
            source_multitalk.get("widgets_values", [])
        )
        source_multitalk_named = source_multitalk.get("widgets_values_named")
        if isinstance(source_multitalk_named, dict):
            target_multitalk["widgets_values_named"] = dict(
                source_multitalk_named
            )

    target.write_text(
        json.dumps(target_workflow, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "LatentSync Stable atualizado para schema "
        f"{source_schema} em {target}"
    )


def upgrade_latentsync_audio_route(target_name: str) -> None:
    """Migrate persisted presets without replacing other user changes."""
    target = COMFYUI_HOME / "user/default/workflows" / target_name
    if not target.exists():
        return

    workflow = json.loads(target.read_text(encoding="utf-8"))
    if not route_original_audio_to_output(workflow):
        return

    target.write_text(json.dumps(workflow, ensure_ascii=False), encoding="utf-8")
    print(f"Audio original conectado a saida LatentSync em {target}")


def upgrade_stable_flashvsr(source: Path, target_name: str) -> None:
    """Append the versioned FlashVSR branch without replacing user changes."""
    target = COMFYUI_HOME / "user/default/workflows" / target_name
    if not source.exists() or not target.exists():
        return

    source_workflow = json.loads(source.read_text(encoding="utf-8"))
    target_workflow = json.loads(target.read_text(encoding="utf-8"))
    source_schema = int(
        source_workflow.get("extra", {}).get("infinitetalk_flashvsr_schema", 0)
    )
    target_extra = target_workflow.setdefault("extra", {})
    target_schema = int(target_extra.get("infinitetalk_flashvsr_schema", 0))
    if source_schema <= 0 or target_schema >= source_schema:
        return

    fullhd_prefix = "InfiniteTalk_V2V_LatentSync16_Stable_FullHD"
    base_prefix = "InfiniteTalk_V2V_LatentSync16_Stable"

    def combine_by_prefix(workflow: dict, prefix: str) -> dict | None:
        return next(
            (
                node
                for node in workflow.get("nodes", [])
                if node.get("type") == "VHS_VideoCombine"
                and node.get("widgets_values", {}).get("filename_prefix")
                == prefix
            ),
            None,
        )

    if combine_by_prefix(target_workflow, fullhd_prefix):
        target_extra["infinitetalk_flashvsr_schema"] = source_schema
        target.write_text(
            json.dumps(target_workflow, ensure_ascii=False), encoding="utf-8"
        )
        return

    source_fullhd = combine_by_prefix(source_workflow, fullhd_prefix)
    target_base = combine_by_prefix(target_workflow, base_prefix)
    if not source_fullhd or not target_base:
        print(f"AVISO: nao foi possivel adicionar FlashVSR em {target}")
        return

    source_nodes = {
        node.get("id"): node for node in source_workflow.get("nodes", [])
    }
    source_links = {
        link[0]: link
        for link in source_workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 6
    }
    target_nodes = {
        node.get("id"): node for node in target_workflow.get("nodes", [])
    }
    target_links = {
        link[0]: link
        for link in target_workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 6
    }

    try:
        target_image_link = target_links[target_base["inputs"][0]["link"]]
        target_audio_link = target_links[target_base["inputs"][1]["link"]]
        source_resize_link = source_links[source_fullhd["inputs"][0]["link"]]
        source_resize = source_nodes[source_resize_link[1]]
        source_flash_link = source_links[source_resize["inputs"][0]["link"]]
        source_flash = source_nodes[source_flash_link[1]]
    except (KeyError, IndexError, TypeError):
        print(f"AVISO: grafo Stable incompativel com a migracao FlashVSR em {target}")
        return

    existing_node_ids = [
        int(node["id"])
        for node in target_workflow.get("nodes", [])
        if isinstance(node.get("id"), int)
    ]
    existing_link_ids = [
        int(link[0])
        for link in target_workflow.get("links", [])
        if isinstance(link, list) and link and isinstance(link[0], int)
    ]
    next_node = max(existing_node_ids, default=0) + 1
    next_link = max(existing_link_ids, default=0) + 1
    flash_id, resize_id, combine_id = range(next_node, next_node + 3)
    image_in, flash_out, resize_out, audio_in = range(next_link, next_link + 4)
    max_order = max(
        (
            int(node.get("order", 0))
            for node in target_workflow.get("nodes", [])
            if isinstance(node.get("order", 0), int)
        ),
        default=0,
    )

    flash = copy.deepcopy(source_flash)
    resize = copy.deepcopy(source_resize)
    combine = copy.deepcopy(source_fullhd)
    flash.update({"id": flash_id, "order": max_order + 1})
    resize.update({"id": resize_id, "order": max_order + 2})
    combine.update({"id": combine_id, "order": max_order + 3})
    flash["inputs"][0]["link"] = image_in
    flash["inputs"][1]["link"] = None
    flash["outputs"][0]["links"] = [flash_out]
    flash["outputs"][1]["links"] = None
    resize["inputs"][0]["link"] = flash_out
    resize["inputs"][1]["link"] = None
    resize["outputs"][0]["links"] = [resize_out]
    for output in resize["outputs"][1:]:
        output["links"] = None
    combine["inputs"][0]["link"] = resize_out
    combine["inputs"][1]["link"] = audio_in
    for output in combine.get("outputs", []):
        output["links"] = None

    image_source_id, image_source_slot = target_image_link[1], target_image_link[2]
    audio_source_id, audio_source_slot = target_audio_link[1], target_audio_link[2]
    try:
        image_output = target_nodes[image_source_id]["outputs"][image_source_slot]
        audio_output = target_nodes[audio_source_id]["outputs"][audio_source_slot]
    except (KeyError, IndexError, TypeError):
        print(f"AVISO: saidas Stable incompativeis com FlashVSR em {target}")
        return
    if not isinstance(image_output.get("links"), list):
        image_output["links"] = []
    if not isinstance(audio_output.get("links"), list):
        audio_output["links"] = []
    image_output["links"].append(image_in)
    audio_output["links"].append(audio_in)

    target_workflow["nodes"].extend((flash, resize, combine))
    target_workflow["links"].extend(
        (
            [image_in, image_source_id, image_source_slot, flash_id, 0, "IMAGE"],
            [flash_out, flash_id, 0, resize_id, 0, "IMAGE"],
            [resize_out, resize_id, 0, combine_id, 0, "IMAGE"],
            [audio_in, audio_source_id, audio_source_slot, combine_id, 1, "AUDIO"],
        )
    )
    target_workflow["last_node_id"] = max(
        int(target_workflow.get("last_node_id", 0)), combine_id
    )
    target_workflow["last_link_id"] = max(
        int(target_workflow.get("last_link_id", 0)), audio_in
    )
    target_extra["infinitetalk_flashvsr_schema"] = source_schema
    target.write_text(
        json.dumps(target_workflow, ensure_ascii=False), encoding="utf-8"
    )
    print(f"FlashVSR FullHD schema {source_schema} adicionado em {target}")


def seed_workflows() -> None:
    seed_workflow(DEFAULT_WORKFLOW, "infinitetalk-i2v-docker.json")
    seed_workflow(
        DEFAULT_V2V_WORKFLOW,
        "infinitetalk-v2v-docker.json",
        "InfiniteTalk_V2V",
        generated_only=True,
        preserve_source=True,
    )
    seed_workflow(
        DEFAULT_V2V_LATENTSYNC_WORKFLOW,
        "infinitetalk-v2v-latentsync16-docker.json",
        preserve_source=True,
    )
    seed_workflow(
        DEFAULT_V2V_LATENTSYNC_STABLE_WORKFLOW,
        "infinitetalk-v2v-latentsync16-stable-docker.json",
        preserve_source=True,
    )
    upgrade_stable_workflow(
        DEFAULT_V2V_LATENTSYNC_STABLE_WORKFLOW,
        "infinitetalk-v2v-latentsync16-stable-docker.json",
    )
    upgrade_latentsync_audio_route(
        "infinitetalk-v2v-latentsync16-docker.json"
    )
    upgrade_latentsync_audio_route(
        "infinitetalk-v2v-latentsync16-stable-docker.json"
    )
    upgrade_stable_flashvsr(
        DEFAULT_V2V_LATENTSYNC_STABLE_WORKFLOW,
        "infinitetalk-v2v-latentsync16-stable-docker.json",
    )
    inject_missing_prompt_defaults(
        DEFAULT_V2V_LATENTSYNC_WORKFLOW,
        "infinitetalk-v2v-latentsync16-docker.json",
    )
    inject_missing_prompt_defaults(
        DEFAULT_V2V_LATENTSYNC_STABLE_WORKFLOW,
        "infinitetalk-v2v-latentsync16-stable-docker.json",
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
