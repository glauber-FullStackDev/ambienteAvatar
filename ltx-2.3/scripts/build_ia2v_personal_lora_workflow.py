#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from build_ia2v_talkvid_workflow import add_link, link_map, load_json, one, replace_input_link, replace_output_links


PERSONAL_LORA_NAME = "glauberavatar.safetensors"
SCHEMA_MARKER = "ltx23_ia2v_personal_lora_schema"
SCHEMA_VERSION = 1


def build_personal_lora(ia2v: dict) -> dict:
    workflow = deepcopy(ia2v)
    if workflow.get("version") != 0.4:
        raise ValueError("o template IA2V oficial precisa usar o schema 0.4")

    subgraph = one(workflow["definitions"]["subgraphs"])
    nodes = subgraph["nodes"]
    links = link_map(subgraph)
    distilled = one(nodes, node_id=293, node_type="LoraLoaderModelOnly")
    decoded = one(nodes, node_id=316, node_type="VAEDecodeTiled")

    personal_loader = deepcopy(distilled)
    personal_loader.update({"id": 351, "pos": [2100, 3430], "order": 50, "title": "Personal LoRA (glauberavatar)"})
    replace_input_link(personal_loader, "model", 766)
    replace_input_link(personal_loader, "lora_name", None)
    personal_loader["widgets_values"] = [PERSONAL_LORA_NAME, 1.0]
    personal_loader["properties"]["models"] = [{"name": PERSONAL_LORA_NAME, "directory": "loras"}]
    replace_output_links(personal_loader, "MODEL", [644, 757])

    last_frame = {
        "id": 350, "type": "LastFrameFromBatch", "pos": [5300, 3440], "size": [260, 80],
        "flags": {}, "order": 49, "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": 763}],
        "outputs": [{"name": "last_frame", "type": "IMAGE", "links": [764]}],
        "properties": {"Node name for S&R": "LastFrameFromBatch"}, "widgets_values": [],
    }
    replace_output_links(distilled, "MODEL", [766])
    replace_output_links(decoded, "IMAGE", [695, 763])
    links[644].update({"origin_id": 351, "origin_slot": 0})
    links[757].update({"origin_id": 351, "origin_slot": 0})
    add_link(subgraph, 763, 316, 0, 350, 0, "IMAGE")
    add_link(subgraph, 764, 350, 0, subgraph["outputNode"]["id"], 1, "IMAGE")
    add_link(subgraph, 766, 293, 0, 351, 0, "MODEL")
    subgraph["outputs"].append({
        "id": "d96a145e-7daa-4e78-91c9-b45ae5f8c3da", "name": "LAST_FRAME", "type": "IMAGE",
        "linkIds": [764], "localized_name": "LAST_FRAME", "pos": [5344, 4080],
    })
    subgraph["outputNode"]["bounding"][3] += 25
    subgraph["state"]["lastNodeId"] = 351
    subgraph["state"]["lastLinkId"] = 766
    nodes.extend([last_frame, personal_loader])
    subgraph["name"] = "Video Generation (LTX-2.3 IA2V + Personal LoRA)"

    top = one(workflow["nodes"], node_id=340)
    top["title"] = "LTX 2.3 IA2V + Personal LoRA"
    top["outputs"].append({"name": "LAST_FRAME", "type": "IMAGE", "links": [767]})
    top["size"][1] += 30
    save_video = one(workflow["nodes"], node_id=341, node_type="SaveVideo")
    save_video["widgets_values"][0] = "video/LTX_2.3_ia2v_personal_lora"
    save_image = {
        "id": 350, "type": "SaveImage", "pos": [430, 4150], "size": [580, 280],
        "flags": {}, "order": 6, "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": 767}], "outputs": [],
        "title": "Last frame PNG (next-video input)",
        "properties": {"Node name for S&R": "SaveImage"},
        "widgets_values": ["images/last_frame/LTX_2.3_ia2v_personal_lora"],
    }
    workflow["nodes"].append(save_image)
    workflow["links"].append([767, 340, 1, 350, 0, "IMAGE"])
    workflow["last_node_id"] = 350
    workflow["last_link_id"] = 767
    workflow.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION
    return workflow


def validate_personal_lora(workflow: dict) -> None:
    failures: list[str] = []
    if workflow.get("extra", {}).get(SCHEMA_MARKER) != SCHEMA_VERSION:
        failures.append("marcador de schema ausente")
    subgraph = workflow["definitions"]["subgraphs"][0]
    nodes = subgraph["nodes"]
    links = link_map(subgraph)
    loaders = [node for node in nodes if node.get("type") == "LoraLoaderModelOnly" and PERSONAL_LORA_NAME in node.get("widgets_values", [])]
    if len(loaders) != 1 or loaders[0].get("widgets_values", [None, None])[1] != 1.0:
        failures.append("LoRA pessoal precisa ter strength=1.0")
    extractor = [node for node in nodes if node.get("type") == "LastFrameFromBatch"]
    if len(extractor) != 1:
        failures.append("extrator do último frame ausente")
    if not any(node.get("type") == "SaveImage" for node in workflow["nodes"]):
        failures.append("SaveImage do último frame ausente")
    if len(links) != len(subgraph["links"]):
        failures.append("IDs de links internos duplicados")
    if failures:
        raise ValueError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria o IA2V LTX 2.3 com LoRA pessoal e PNG final")
    parser.add_argument("--ia2v", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workflow = build_personal_lora(load_json(args.ia2v))
    validate_personal_lora(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Workflow IA2V + LoRA pessoal criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
