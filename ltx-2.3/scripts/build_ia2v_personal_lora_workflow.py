#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from build_ia2v_talkvid_workflow import add_link, link_map, load_json, one, replace_output_links


PERSONAL_LORA_NAME = "glauberavatar.safetensors"
SCHEMA_MARKER = "ltx23_ia2v_personal_lora_schema"
SCHEMA_VERSION = 2


def _next_id(items: list[dict]) -> int:
    return max(item["id"] for item in items if isinstance(item.get("id"), int)) + 1


def add_last_frame_export(source: dict) -> dict:
    """Preserve the user-supplied personal workflow and append only its PNG export."""
    workflow = deepcopy(source)
    if workflow.get("version") != 0.4:
        raise ValueError("o workflow pessoal precisa usar o schema 0.4")

    subgraph = one(workflow["definitions"]["subgraphs"])
    nodes = subgraph["nodes"]
    links = link_map(subgraph)
    decoded = one(nodes, node_type="VAEDecodeTiled")
    create_video = one(nodes, node_type="CreateVideo")
    if any(node.get("type") == "LastFrameFromBatch" for node in nodes):
        raise ValueError("o workflow pessoal ja possui um extrator de ultimo frame")

    extractor_id = _next_id(nodes)
    first_link_id = max(links) + 1
    output_link_id = first_link_id + 1
    extractor = {
        "id": extractor_id,
        "type": "LastFrameFromBatch",
        "pos": [5300, 3440],
        "size": [280, 80],
        "flags": {},
        "order": max(node.get("order", 0) for node in nodes) + 1,
        "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": first_link_id}],
        "outputs": [
            {"name": "last_frame", "type": "IMAGE", "links": [output_link_id]}
        ],
        "properties": {"Node name for S&R": "LastFrameFromBatch"},
        "widgets_values": [],
    }
    replace_output_links(
        decoded,
        "IMAGE",
        one([output for output in decoded["outputs"] if output["name"] == "IMAGE"])[
            "links"
        ]
        + [first_link_id],
    )
    add_link(subgraph, first_link_id, decoded["id"], 0, extractor_id, 0, "IMAGE")
    output_slot = len(subgraph["outputs"])
    add_link(
        subgraph,
        output_link_id,
        extractor_id,
        0,
        subgraph["outputNode"]["id"],
        output_slot,
        "IMAGE",
    )
    subgraph["outputs"].append(
        {
            "id": "d96a145e-7daa-4e78-91c9-b45ae5f8c3da",
            "name": "LAST_FRAME",
            "type": "IMAGE",
            "linkIds": [output_link_id],
            "localized_name": "LAST_FRAME",
            "pos": [5344, 4080],
        }
    )
    subgraph["outputNode"]["bounding"][3] += 25
    subgraph["state"]["lastNodeId"] = extractor_id
    subgraph["state"]["lastLinkId"] = output_link_id
    nodes.append(extractor)

    top = one(workflow["nodes"], node_id=340)
    top_output_slot = len(top["outputs"])
    outer_link_id = workflow["last_link_id"] + 1
    top["outputs"].append(
        {"name": "LAST_FRAME", "type": "IMAGE", "links": [outer_link_id]}
    )
    save_video = one(workflow["nodes"], node_id=341, node_type="SaveVideo")
    save_image_id = _next_id(workflow["nodes"])
    save_image = {
        "id": save_image_id,
        "type": "SaveImage",
        "pos": [save_video["pos"][0] + save_video["size"][0] + 30, save_video["pos"][1]],
        "size": [420, 310],
        "flags": {},
        "order": max(node.get("order", 0) for node in workflow["nodes"]) + 1,
        "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": outer_link_id}],
        "outputs": [],
        "title": "Last frame PNG (next-video input)",
        "properties": {"Node name for S&R": "SaveImage"},
        "widgets_values": ["images/last_frame/LTX_2.3_ia2v_personal_lora"],
    }
    workflow["links"].append(
        [outer_link_id, top["id"], top_output_slot, save_image_id, 0, "IMAGE"]
    )
    workflow["nodes"].append(save_image)
    workflow["last_node_id"] = save_image_id
    workflow["last_link_id"] = outer_link_id
    workflow.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION
    return workflow


def validate_personal_lora(workflow: dict) -> None:
    failures: list[str] = []
    if workflow.get("extra", {}).get(SCHEMA_MARKER) != SCHEMA_VERSION:
        failures.append("marcador de schema ausente")
    subgraph = workflow["definitions"]["subgraphs"][0]
    nodes = subgraph["nodes"]
    links = link_map(subgraph)
    loaders = [
        node
        for node in nodes
        if node.get("type") == "LoraLoaderModelOnly"
        and PERSONAL_LORA_NAME in node.get("widgets_values", [])
    ]
    if len(loaders) != 1:
        failures.append("LoRA pessoal ausente")
    if len([node for node in nodes if node.get("type") == "LastFrameFromBatch"]) != 1:
        failures.append("extrator do ultimo frame ausente")
    save_images = [node for node in workflow["nodes"] if node.get("type") == "SaveImage"]
    if len(save_images) != 1:
        failures.append("SaveImage do ultimo frame ausente")
    if len(links) != len(subgraph["links"]):
        failures.append("IDs de links internos duplicados")
    if failures:
        raise ValueError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adiciona exportacao PNG do ultimo frame ao workflow pessoal IA2V"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workflow = add_last_frame_export(load_json(args.source))
    validate_personal_lora(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Workflow pessoal + ultimo frame criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
