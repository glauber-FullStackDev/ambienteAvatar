#!/usr/bin/env python3
"""Compile the checked-in Personal LoRA UI workflow to ComfyUI API format."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

LOAD_IMAGE_ID = "269"
LOAD_AUDIO_ID = "276"
PROMPT_ID = "319"
PROMPT_ENHANCE_ID = "349"
WIDTH_ID = "330"
HEIGHT_ID = "324"
DURATION_ID = "331"
FPS_ID = "323"
AUDIO_START_ID = "332"
SEED_ID = "286"
PERSONAL_LORA_ID = "350"
PROMPT_ENHANCE_ID = "349"
IMAGE_STRENGTH_ID = "325"
VIDEO_SAVE_ID = "9001"
LAST_FRAME_ID = "9002"
IMAGE_SAVE_ID = "9003"

REQUIRED_NODE_TYPES = {
    LOAD_IMAGE_ID: "LoadImage",
    LOAD_AUDIO_ID: "LoadAudio",
    PROMPT_ID: "PrimitiveStringMultiline",
    PERSONAL_LORA_ID: "LoraLoaderModelOnly",
    IMAGE_STRENGTH_ID: "LTXVImgToVideoInplace",
    "312": "CreateVideo",
    "316": "VAEDecodeTiled",
}
EXCLUDED_WIDGETS = {"fixed", "upload"}


def _top_link_map(workflow: dict[str, Any]) -> dict[int, tuple[str, int]]:
    return {int(link[0]): (str(link[1]), int(link[2])) for link in workflow["links"]}


def _node_values(node: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in node.get("widgets_values_named", {}).items()
        if name not in EXCLUDED_WIDGETS
    }


def _is_workflow_link(value: Any, prompt: dict[str, Any]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and value[0] in prompt
        and isinstance(value[1], int)
    )


def _remove_reroutes(prompt: dict[str, Any]) -> None:
    """Remove UI-only Reroute nodes and reconnect their consumers for /prompt."""
    reroutes = {
        node_id: node
        for node_id, node in prompt.items()
        if node["class_type"] == "Reroute"
    }

    def upstream(link: list[Any]) -> list[Any]:
        node_id = link[0]
        if node_id not in reroutes:
            return link
        source = next(
            (
                value
                for value in reroutes[node_id]["inputs"].values()
                if _is_workflow_link(value, prompt)
            ),
            None,
        )
        if source is None:
            raise ValueError(f"Reroute {node_id} sem entrada conectada")
        return upstream(source)

    for node in prompt.values():
        for name, value in node["inputs"].items():
            if _is_workflow_link(value, prompt):
                node["inputs"][name] = upstream(value)
    for node_id in reroutes:
        del prompt[node_id]


def compile_workflow(source: dict[str, Any]) -> dict[str, Any]:
    """Flatten the one subgraph used by the existing IA2V Personal LoRA workflow."""
    top_nodes = {int(node["id"]): node for node in source["nodes"]}
    group = top_nodes[340]
    subgraph = source["definitions"]["subgraphs"][0]
    sub_nodes = {int(node["id"]): node for node in subgraph["nodes"]}
    sub_links = {int(link["id"]): link for link in subgraph["links"]}
    top_links = _top_link_map(source)

    for node_id, expected_type in REQUIRED_NODE_TYPES.items():
        node = sub_nodes.get(int(node_id)) if int(node_id) >= 280 else top_nodes.get(int(node_id))
        if not node or node.get("type") != expected_type:
            raise ValueError(f"workflow pessoal incompatível: nó {node_id} não é {expected_type}")

    group_inputs: dict[int, Any] = {}
    for slot, input_def in enumerate(group["inputs"]):
        if input_def.get("link") is not None:
            group_inputs[slot] = list(top_links[int(input_def["link"])])
        elif slot >= 2:
            # The parent group stores its widget values positionally.  Its
            # displayed widget names are shifted for COMBO fields, so relying
            # on widgets_values_named would wire the wrong model files.
            group_inputs[slot] = group["widgets_values"][slot - 2]

    # The source workflow exposes the last two model inputs only inside the
    # subgraph; their values still occupy positions 11 and 12 in the parent's
    # widget array.
    for slot in range(len(subgraph["inputs"])):
        if slot not in group_inputs and slot >= 2:
            group_inputs[slot] = group["widgets_values"][slot - 2]

    prompt: dict[str, Any] = {}
    for node_id, node in sub_nodes.items():
        inputs = _node_values(node)
        for input_def in node.get("inputs", []):
            name = input_def["name"]
            link_id = input_def.get("link")
            if link_id is None:
                continue
            link = sub_links[int(link_id)]
            if int(link["origin_id"]) == -10:
                inputs[name] = group_inputs[int(link["origin_slot"])]
            else:
                inputs[name] = [str(link["origin_id"]), int(link["origin_slot"])]
        prompt[str(node_id)] = {
            "class_type": node["type"],
            "inputs": inputs,
            "_meta": {"title": node.get("title") or node["type"]},
        }

    for node_id in (269, 276):
        node = top_nodes[node_id]
        prompt[str(node_id)] = {
            "class_type": node["type"],
            "inputs": _node_values(node),
            "_meta": {"title": node.get("title") or node["type"]},
        }

    prompt[VIDEO_SAVE_ID] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["312", 0],
            "filename_prefix": "video/jobs/__JOB_ID__/LTX_2.3_ia2v_personal_lora",
            # SaveVideo on the pinned ComfyUI release requires this input even
            # when it uses its automatic video/container choice.
            "format": "auto",
        },
        "_meta": {"title": "Serverless video output"},
    }
    prompt[LAST_FRAME_ID] = {
        "class_type": "LastFrameFromBatch",
        "inputs": {"images": ["316", 0]},
        "_meta": {"title": "Serverless last-frame extractor"},
    }
    prompt[IMAGE_SAVE_ID] = {
        "class_type": "SaveImage",
        "inputs": {"images": [LAST_FRAME_ID, 0], "filename_prefix": "images/last_frame/jobs/__JOB_ID__/LTX_2.3_ia2v_personal_lora"},
        "_meta": {"title": "Serverless last-frame output"},
    }
    _remove_reroutes(prompt)
    return prompt


def build_job_workflow(template: dict[str, Any], values: dict[str, Any], job_id: str) -> dict[str, Any]:
    if template.get("_meta", {}).get("template") == "ltx25_iclora":
        return build_job_workflow_ltx25(template, values, job_id)
    return build_job_workflow_ltx23(template, values, job_id)


def build_job_workflow_ltx25(template: dict[str, Any], values: dict[str, Any], job_id: str) -> dict[str, Any]:
    workflow = deepcopy(template)
    prompt = workflow["prompt"]
    updates = {
        ("100", "image"): values["image_filename"],
        ("101", "audio"): values["audio_filename"],
        ("104", "value"): values["width"],
        ("105", "value"): values["height"],
        ("106", "value"): values["duration_seconds"],
        ("107", "value"): values["fps"],
        ("108", "value"): values["audio_start_seconds"],
        ("109", "value"): values["seed"],
        ("110", "value"): values["first_frame_strength"],
        ("111", "value"): values["guiding_strength"],
        ("131", "strength_model"): values["iclora_strength"],
        ("132", "strength_model"): values["lora_strength"],
        ("163", "cfg"): values["cfg"],
        ("176", "cfg"): values["cfg"],
    }
    if values.get("prompt"):
        updates[("102", "value")] = values["prompt"]
    if values.get("negative_prompt"):
        updates[("103", "value")] = values["negative_prompt"]
    if values.get("enable_prompt_enhance") is not None:
        updates[("144", "value")] = bool(values["enable_prompt_enhance"])
    if values.get("base_sigmas"):
        updates[("160", "sigmas")] = values["base_sigmas"]
    if values.get("refine_sigmas"):
        updates[("173", "sigmas")] = values["refine_sigmas"]
    for (node_id, name), value in updates.items():
        prompt[node_id]["inputs"][name] = value

    reference_filename = values.get("reference_image_filename")
    if reference_filename:
        prompt["115"]["inputs"]["image"] = reference_filename
        prompt["118"]["inputs"]["value"] = values.get("reference_frame_idx", -1)
        prompt["119"]["inputs"]["value"] = values.get("reference_guiding_strength", values.get("guiding_strength", 0.8))
    else:
        # Bypass the optional reference guide entirely: rewire consumers back
        # to the persistent guide and drop the unused nodes.
        prompt["163"]["inputs"]["positive"] = ["154", 0]
        prompt["163"]["inputs"]["negative"] = ["154", 1]
        prompt["164"]["inputs"]["video_latent"] = ["154", 2]
        prompt["167"]["inputs"]["positive"] = ["154", 0]
        prompt["167"]["inputs"]["negative"] = ["154", 1]
        for node_id in ("115", "116", "117", "118", "119", "156"):
            del prompt[node_id]

    prompt["183"]["inputs"]["filename_prefix"] = f"video/jobs/{job_id}/LTX_2.5_ia2v_iclora"
    prompt["185"]["inputs"]["filename_prefix"] = f"images/last_frame/jobs/{job_id}/LTX_2.5_ia2v_iclora"
    return prompt


def build_job_workflow_ltx23(template: dict[str, Any], values: dict[str, Any], job_id: str) -> dict[str, Any]:
    workflow = deepcopy(template)
    updates = {
        (LOAD_IMAGE_ID, "image"): values["image_filename"],
        (LOAD_AUDIO_ID, "audio"): values["audio_filename"],
        (PROMPT_ID, "value"): values["prompt"],
        (WIDTH_ID, "value"): values["width"],
        (HEIGHT_ID, "value"): values["height"],
        (DURATION_ID, "value"): values["duration_seconds"],
        (FPS_ID, "value"): values["fps"],
        (AUDIO_START_ID, "start_index"): values["audio_start_seconds"],
        (SEED_ID, "noise_seed"): values["seed"],
        (PERSONAL_LORA_ID, "strength_model"): values["lora_strength"],
        (PROMPT_ENHANCE_ID, "value"): values["enable_prompt_enhance"],
        (IMAGE_STRENGTH_ID, "strength"): values["image_strength"],
    }
    for (node_id, name), value in updates.items():
        workflow[node_id]["inputs"][name] = value
    workflow[VIDEO_SAVE_ID]["inputs"]["filename_prefix"] = f"video/jobs/{job_id}/LTX_2.3_ia2v_personal_lora"
    workflow[IMAGE_SAVE_ID]["inputs"]["filename_prefix"] = f"images/last_frame/jobs/{job_id}/LTX_2.3_ia2v_personal_lora"
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiled = compile_workflow(json.loads(args.source.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compiled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
