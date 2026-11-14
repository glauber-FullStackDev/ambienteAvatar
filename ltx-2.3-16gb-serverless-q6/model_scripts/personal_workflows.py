#!/usr/bin/env python3
"""Transformacoes dos workflows pessoais do GlauberAvatar (IA2V e A2V).

Logicas copiadas do path 48 GB (`ltx-2.3/scripts/build_ia2v_personal_lora_workflow.py`
e `build_a2v_personal_lora_workflow.py`), preservando IDs de nodes e links.
A conversao GGUF 16 GB acontece depois, em `gguf_convert.py`.
"""
from __future__ import annotations

from copy import deepcopy

from workflow_utils import (
    add_link,
    link_map,
    next_node_id,
    one,
    replace_output_links,
)


PERSONAL_LORA_NAME = "glauberavatar.safetensors"
IA2V_SCHEMA_MARKER = "ltx23_ia2v_personal_lora_schema"
IA2V_SCHEMA_VERSION = 2
A2V_SCHEMA_MARKER = "ltx23_a2v_personal_lora_schema"
A2V_SCHEMA_VERSION = 1


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

    extractor_id = next_node_id(nodes)
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
    save_image_id = next_node_id(workflow["nodes"])
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
    workflow.setdefault("extra", {})[IA2V_SCHEMA_MARKER] = IA2V_SCHEMA_VERSION
    return workflow


def validate_personal_lora(workflow: dict) -> None:
    failures: list[str] = []
    if workflow.get("extra", {}).get(IA2V_SCHEMA_MARKER) != IA2V_SCHEMA_VERSION:
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
    subgraph["name"] = "Audio to Video (LTX-2.3 GGUF + Glauber Avatar)"

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
    top["title"] = "LTX 2.3 A2V GGUF + Glauber Avatar"
    top["size"][1] -= 30

    save_video = one(workflow["nodes"], node_id=341, node_type="SaveVideo")
    save_video["widgets_values"][0] = "video/LTX_2.3_a2v_personal_lora"
    save_image = one(workflow["nodes"], node_type="SaveImage")
    save_image["widgets_values"][0] = "images/last_frame/LTX_2.3_a2v_personal_lora"
    workflow.setdefault("extra", {})[A2V_SCHEMA_MARKER] = A2V_SCHEMA_VERSION
    return workflow


def validate_a2v(workflow: dict) -> None:
    failures: list[str] = []
    if workflow.get("extra", {}).get(A2V_SCHEMA_MARKER) != A2V_SCHEMA_VERSION:
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
