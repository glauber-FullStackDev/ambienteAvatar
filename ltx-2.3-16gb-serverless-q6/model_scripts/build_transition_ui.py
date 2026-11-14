#!/usr/bin/env python3
"""Create an editable ComfyUI first/last-frame + audio workflow from IA2V."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path


def build(source: dict) -> dict:
    workflow = deepcopy(source)
    graph = workflow["definitions"]["subgraphs"][0]
    nodes = {n["id"]: n for n in graph["nodes"]}
    group = next(n for n in workflow["nodes"] if n["id"] == 340)
    first = next(n for n in workflow["nodes"] if n["id"] == 269)
    last = deepcopy(first)
    last.update(id=10001, pos=[-880, 3530], title="Imagem final (referência)")
    last["outputs"][0]["links"] = [10001]
    workflow["nodes"].append(last)
    # Insert the new image port before widgets, retaining their widget order.
    group["inputs"].insert(2, {"name": "end_image", "label": "last_frame", "type": "IMAGE,MASK", "link": 10001})
    workflow["links"] = [[*link[:4], link[4] + 1, *link[5:]] if link[3] == 340 and link[4] >= 2 else link for link in workflow["links"]]
    workflow["links"].append([10001, 10001, 0, 340, 2, "IMAGE"])
    for link in graph["links"]:
        if link["origin_id"] == -10 and link["origin_slot"] >= 2:
            link["origin_slot"] += 1
    graph["inputs"].insert(2, {"id": "d9988dbe-d77e-4a97-9591-3545039b8901", "name": "end_image", "label": "last_frame", "type": "IMAGE,MASK", "linkIds": [10002], "pos": [0, 0]})

    def clone(original: int, new_id: int, title: str) -> dict:
        node = deepcopy(nodes[original]); node["id"] = new_id; node["title"] = title
        for slot in node.get("inputs", []): slot["link"] = None
        for slot in node.get("outputs", []): slot["links"] = []
        graph["nodes"].append(node); nodes[new_id] = node
        return node

    clone(297, 10002, "Ajustar imagem final")["pos"] = [890, 4900]
    clone(294, 10003, "Redimensionar imagem final")["pos"] = [1220, 4900]
    clone(334, 10004, "Preparar imagem final")["pos"] = [1550, 4900]
    guide = {"id": 10005, "type": "LTXVAddGuide", "pos": [2500, 4500], "size": [280, 240], "flags": {}, "order": 50, "mode": 0,
             "title": "Guia do frame final", "properties": {"Node name for S&R": "LTXVAddGuide"},
             "inputs": [{"name": n, "type": t, "link": None} for n,t in [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"), ("latent", "LATENT"), ("image", "IMAGE")]],
             "outputs": [{"name": n, "type": t, "links": []} for n,t in [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")]],
             "widgets_values": [-1, 0.7], "widgets_values_named": {"frame_idx": -1, "strength": 0.7}}
    graph["nodes"].append(guide); nodes[10005] = guide
    links = {l["id"]: l for l in graph["links"]}
    def redirect(link_id: int, origin: int, slot: int) -> None:
        link = links[link_id]; old = nodes[link["origin_id"]]["outputs"][link["origin_slot"]]["links"]
        old.remove(link_id); link["origin_id"] = origin; link["origin_slot"] = slot
        nodes[origin]["outputs"][slot]["links"].append(link_id)
    for link_id,slot in [(648,0),(655,0),(649,1),(656,1),(685,2)]: redirect(link_id,10005,slot)
    redirect(660,292,2)
    def add(link_id: int, origin: int, oslot: int, target: int, tslot: int, typ: str) -> None:
        graph["links"].append({"id": link_id, "origin_id": origin, "origin_slot": oslot, "target_id": target, "target_slot": tslot, "type": typ})
        nodes[target]["inputs"][tslot]["link"] = link_id
        if origin != -10:
            output = nodes[origin]["outputs"][oslot]
            if output.get("links") is None: output["links"] = []
            output["links"].append(link_id)
    for x in [(10002,-10,2,10002,0,"IMAGE,MASK"),(10003,330,0,10002,1,"INT"),(10004,324,0,10002,2,"INT"),
              (10005,10002,0,10003,0,"IMAGE,MASK"),(10006,10003,0,10004,0,"IMAGE"),
              (10007,307,0,10005,0,"CONDITIONING"),(10008,307,1,10005,1,"CONDITIONING"),
              (10009,352,0,10005,2,"VAE"),(10010,325,0,10005,3,"LATENT"),(10011,10004,0,10005,4,"IMAGE")]: add(*x)
    graph["state"]["lastNodeId"] = 10005; graph["state"]["lastLinkId"] = 10011
    workflow["last_node_id"] = 10001; workflow["last_link_id"] = 10001
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); workflow = build(json.loads(args.source.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
