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
        "inputs": {"video": ["312", 0], "filename_prefix": "video/jobs/__JOB_ID__/LTX_2.3_ia2v_personal_lora"},
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
    return prompt


def build_job_workflow(template: dict[str, Any], values: dict[str, Any], job_id: str) -> dict[str, Any]:
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
