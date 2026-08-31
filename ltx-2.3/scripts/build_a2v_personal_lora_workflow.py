#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from build_ia2v_personal_lora_workflow import (
    PERSONAL_LORA_NAME,
    add_last_frame_export,
)
from build_ia2v_talkvid_workflow import link_map, load_json, one


SCHEMA_MARKER = "ltx23_a2v_personal_lora_schema"
SCHEMA_VERSION = 1


def _output(node: dict, name: str) -> dict:
    return one([item for item in node["outputs"] if item.get("name") == name])


def _input(node: dict, name: str) -> dict:
    return one([item for item in node["inputs"] if item.get("name") == name])


def build_a2v(source: dict) -> dict:
    """Create pure A2V from the personal IA2V source without any image input."""
    workflow = add_last_frame_export(source)
    subgraph = one(workflow["definitions"]["subgraphs"])
    nodes = subgraph["nodes"]
    links = link_map(subgraph)

    # Do not merely bypass the I2V nodes: remove their image-only branch so no
    # image input or image-dependent execution remains in the A2V preset.
    removed_ids = {294, 296, 297, 325, 334}
    empty_video = one(nodes, node_id=302, node_type="EmptyLTXVLatentVideo")
    concat = one(nodes, node_id=326, node_type="LTXVConcatAVLatent")
    upscaler = one(nodes, node_id=295, node_type="LTXVLatentUpsampler")
    second_concat = one(nodes, node_id=287, node_type="LTXVConcatAVLatent")
    t2v = one(nodes, node_id=305, node_type="PrimitiveBoolean")

    # Reuse the existing link IDs after redirecting around the two I2V stages.
    links[677].update({"origin_id": 302, "origin_slot": 0, "target_id": 326, "target_slot": 0})
    _output(empty_video, "LATENT")["links"] = [677]
    _input(concat, "video_latent")["link"] = 677
    links[642].update({"origin_id": 295, "origin_slot": 0, "target_id": 287, "target_slot": 0})
    _output(upscaler, "LATENT")["links"] = [642]
    _input(second_concat, "video_latent")["link"] = 642

    subgraph["links"] = [
        link
        for link in subgraph["links"]
        if not (
            (link["origin_id"] in removed_ids or link["target_id"] in removed_ids)
            and link["id"] not in {642, 677}
        )
    ]
    nodes[:] = [node for node in nodes if node["id"] not in removed_ids]
    remaining_link_ids = {link["id"] for link in subgraph["links"]}
    for node in nodes:
        for output in node.get("outputs", []):
            if output.get("links") is not None:
                output["links"] = [
                    link_id
                    for link_id in output["links"]
                    if link_id in remaining_link_ids
                ]

    # Remove the first-frame interface and renumber the remaining subgraph inputs.
    image_input = subgraph["inputs"].pop(0)
    image_link_ids = set(image_input["linkIds"])
    subgraph["links"] = [
        link for link in subgraph["links"] if link["id"] not in image_link_ids
    ]
    for link in subgraph["links"]:
        if link["origin_id"] == subgraph["inputNode"]["id"]:
            link["origin_slot"] -= 1
    subgraph["inputNode"]["bounding"][3] -= 20
    subgraph["name"] = "Audio to Video (LTX-2.3 + Glauber Avatar)"

    # Keep the semantic T2V setting visible and immutable in the preset.
    t2v["title"] = "A2V mode (always enabled)"
    t2v["widgets_values"] = [True]
    t2v["widgets_values_named"] = {"value": True}
    _output(t2v, "BOOLEAN")["links"] = []

    # Remove the visible Load Image node and its corresponding first-frame slot.
    top = one(workflow["nodes"], node_id=340)
    load_image = one(workflow["nodes"], node_id=269, node_type="LoadImage")
    top["inputs"].pop(0)
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] != 269]
    workflow["links"] = [
        link
        for link in workflow["links"]
        if link[1] != load_image["id"] and link[2] != load_image["id"]
    ]
    for link in workflow["links"]:
        if link[3] == top["id"]:
            link[4] -= 1
    top["title"] = "LTX 2.3 A2V + Glauber Avatar"
    top["size"][1] -= 30

    save_video = one(workflow["nodes"], node_id=341, node_type="SaveVideo")
    save_video["widgets_values"][0] = "video/LTX_2.3_a2v_personal_lora"
    save_image = one(workflow["nodes"], node_type="SaveImage")
    save_image["widgets_values"][0] = "images/last_frame/LTX_2.3_a2v_personal_lora"
    workflow.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION
    return workflow


def validate_a2v(workflow: dict) -> None:
    failures: list[str] = []
    if workflow.get("extra", {}).get(SCHEMA_MARKER) != SCHEMA_VERSION:
        failures.append("marcador de schema A2V ausente")
    if any(node.get("type") == "LoadImage" for node in workflow["nodes"]):
        failures.append("A2V nao pode expor LoadImage")

    subgraph = one(workflow["definitions"]["subgraphs"])
    if any("IMAGE" in item.get("type", "") for item in subgraph["inputs"]):
        failures.append("A2V nao pode ter entrada de imagem")
    nodes = subgraph["nodes"]
    if any(node.get("type") == "LTXVImgToVideoInplace" for node in nodes):
        failures.append("nodes I2V nao devem permanecer no A2V")
    t2v = one(nodes, node_id=305, node_type="PrimitiveBoolean")
    if t2v.get("widgets_values") != [True]:
        failures.append("modo A2V/T2V precisa permanecer ligado")
    loaders = [
        node for node in nodes
        if node.get("type") == "LoraLoaderModelOnly"
        and PERSONAL_LORA_NAME in node.get("widgets_values", [])
    ]
    if len(loaders) != 1:
        failures.append("LoRA Glauber Avatar ausente")
    if len([node for node in nodes if node.get("type") == "LastFrameFromBatch"]) != 1:
        failures.append("extrator do ultimo frame ausente")
    if len([node for node in workflow["nodes"] if node.get("type") == "SaveImage"]) != 1:
        failures.append("SaveImage externo ausente")

    links = link_map(subgraph)
    if len(links) != len(subgraph["links"]):
        failures.append("IDs de links internos duplicados")
    for node in nodes:
        for slot, item in enumerate(node.get("inputs", [])):
            link_id = item.get("link")
            if link_id is not None and (
                link_id not in links
                or links[link_id]["target_id"] != node["id"]
                or links[link_id]["target_slot"] != slot
            ):
                failures.append(f"entrada inconsistente: {node['id']}:{slot}")
    if failures:
        raise ValueError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria o workflow A2V pessoal LTX 2.3")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workflow = build_a2v(load_json(args.source))
    validate_a2v(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Workflow A2V + LoRA pessoal criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
