#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


CHECKPOINT = "ltx-2.3-22b-dev-fp8.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
DISTILLED_LORA = (
    "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
)
INGREDIENTS_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
SCHEMA_MARKER = "ltx23_ingredients_reference_schema"
SCHEMA_VERSION = 1

MODEL_REPLACEMENTS = {
    "ltx-2.3-22b-dev.safetensors": CHECKPOINT,
    "comfy_gemma_3_12B_it.safetensors": TEXT_ENCODER,
    "ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors": DISTILLED_LORA,
    "ltxv/ltx2/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors": INGREDIENTS_LORA,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def one(nodes: list[dict], *, node_id: int | None = None, node_type: str | None = None) -> dict:
    matches = [
        node
        for node in nodes
        if (node_id is None or node.get("id") == node_id)
        and (node_type is None or node.get("type") == node_type)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"esperado um node id={node_id!r} type={node_type!r}; encontrados={len(matches)}"
        )
    return matches[0]


def replace_strings(value: object) -> object:
    if isinstance(value, str):
        return MODEL_REPLACEMENTS.get(value, value)
    if isinstance(value, list):
        return [replace_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item) for key, item in value.items()}
    return value


def append_link(
    workflow: dict,
    origin: dict,
    origin_slot: int,
    target: dict,
    target_slot: int,
    link_type: str,
) -> int:
    link_id = workflow["last_link_id"] + 1
    workflow["last_link_id"] = link_id
    workflow["links"].append(
        [link_id, origin["id"], origin_slot, target["id"], target_slot, link_type]
    )
    origin["outputs"][origin_slot].setdefault("links", []).append(link_id)
    target["inputs"][target_slot]["link"] = link_id
    return link_id


def adapt_official(source: dict) -> dict:
    if source.get("version") != 0.4:
        raise ValueError("o workflow Ingredients oficial precisa usar schema 0.4")
    workflow = replace_strings(deepcopy(source))
    assert isinstance(workflow, dict)
    nodes = workflow["nodes"]

    one(nodes, node_type="CheckpointLoaderSimple")["widgets_values"][0] = CHECKPOINT
    one(nodes, node_type="LTXVAudioVAELoader")["widgets_values"][0] = CHECKPOINT
    text_loader = one(nodes, node_type="LTXAVTextEncoderLoader")
    text_loader["widgets_values"][0] = TEXT_ENCODER
    text_loader["widgets_values"][1] = CHECKPOINT
    one(nodes, node_type="LoraLoaderModelOnly")["widgets_values"][0] = DISTILLED_LORA
    one(nodes, node_type="LTXICLoRALoaderModelOnly")["widgets_values"] = [
        INGREDIENTS_LORA,
        1,
    ]

    reference = one(nodes, node_id=2004, node_type="LoadImage")
    reference["title"] = "Ingredients Reference Sheet (official format)"
    one(nodes, node_type="SaveVideo")["widgets_values"][0] = (
        "video/LTX_2.3_ingredients_official_10s"
    )
    workflow.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION
    workflow["extra"]["ltx23_ingredients_profile"] = "official-single-stage-10s"
    workflow["extra"]["source_workflow"] = (
        "LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json"
    )
    return workflow


def add_wangp_i2v_profile(official: dict) -> dict:
    workflow = deepcopy(official)
    nodes = workflow["nodes"]
    links = {link[0]: link for link in workflow["links"]}

    # The Reddit/WanGP path repeats the Ingredients sheet for every generated
    # frame and optionally conditions the first frame. The official ComfyUI
    # Ingredients graph already does the full-length repeat; add the native I2V
    # latent conditioner before the IC-LoRA guide and use 15 s at 24 fps.
    video_length = one(nodes, node_id=5072, node_type="PrimitiveInt")
    video_length["widgets_values"] = [361, "fixed"]
    video_length["title"] = "Video Length (15s at 24fps; 8n+1)"
    one(nodes, node_type="SaveVideo")["widgets_values"][0] = (
        "video/LTX_2.3_ingredients_wangp_i2v_15s"
    )

    workflow["last_node_id"] = max(workflow["last_node_id"], 5114)
    start_image = {
        "id": 5110,
        "type": "LoadImage",
        "pos": [-4300, 4320],
        "size": [270, 314],
        "flags": {},
        "order": 9,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "MASK", "type": "MASK", "links": []},
        ],
        "title": "Optional I2V Start Image",
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.56",
            "Node name for S&R": "LoadImage",
        },
        "widgets_values": ["start_image.png", "image"],
    }
    resize = {
        "id": 5111,
        "type": "ResizeImageMaskNode",
        "pos": [-3970, 4420],
        "size": [270, 106],
        "flags": {},
        "order": 22,
        "mode": 0,
        "inputs": [{"name": "input", "type": "IMAGE,MASK", "link": None}],
        "outputs": [{"name": "resized", "type": "IMAGE", "links": []}],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.16.0",
            "Node name for S&R": "ResizeImageMaskNode",
        },
        "widgets_values": ["scale longer dimension", 1536, "lanczos"],
    }
    preprocess = {
        "id": 5112,
        "type": "LTXVPreprocess",
        "pos": [-3640, 4420],
        "size": [270, 74],
        "flags": {},
        "order": 23,
        "mode": 0,
        "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
        "outputs": [{"name": "output_image", "type": "IMAGE", "links": []}],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.60",
            "Node name for S&R": "LTXVPreprocess",
        },
        "widgets_values": [18],
    }
    bypass = {
        "id": 5113,
        "type": "PrimitiveBoolean",
        "pos": [-3970, 4560],
        "size": [270, 58],
        "flags": {},
        "order": 10,
        "mode": 0,
        "inputs": [],
        "outputs": [{"name": "BOOLEAN", "type": "BOOLEAN", "links": []}],
        "title": "bypass_i2v (False = use start image)",
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.16.0",
            "Node name for S&R": "PrimitiveBoolean",
        },
        "widgets_values": [False],
    }
    i2v = {
        "id": 5114,
        "type": "LTXVImgToVideoConditionOnly",
        "pos": [-2220, 3530],
        "size": [315, 143],
        "flags": {},
        "order": 26,
        "mode": 0,
        "inputs": [
            {"name": "vae", "type": "VAE", "link": None},
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "latent", "type": "LATENT", "link": None},
            {
                "name": "bypass",
                "shape": 7,
                "type": "BOOLEAN",
                "widget": {"name": "bypass"},
                "link": None,
            },
        ],
        "outputs": [{"name": "latent", "type": "LATENT", "links": []}],
        "properties": {"Node name for S&R": "LTXVImgToVideoConditionOnly"},
        "widgets_values": [0.7, False],
    }
    nodes.extend([start_image, resize, preprocess, bypass, i2v])

    checkpoint = one(nodes, node_type="CheckpointLoaderSimple")
    empty_video = one(nodes, node_type="EmptyLTXVLatentVideo")
    guide = one(nodes, node_type="LTXAddVideoICLoRAGuide")
    old_latent_link = guide["inputs"][3]["link"]
    links[old_latent_link][3] = i2v["id"]
    links[old_latent_link][4] = 2
    i2v["inputs"][2]["link"] = old_latent_link
    guide["inputs"][3]["link"] = None

    append_link(workflow, start_image, 0, resize, 0, "IMAGE,MASK")
    append_link(workflow, resize, 0, preprocess, 0, "IMAGE")
    append_link(workflow, preprocess, 0, i2v, 1, "IMAGE")
    append_link(workflow, bypass, 0, i2v, 3, "BOOLEAN")
    append_link(workflow, checkpoint, 2, i2v, 0, "VAE")
    append_link(workflow, i2v, 0, guide, 3, "LATENT")

    assert empty_video["outputs"][0]["links"] == [old_latent_link]
    workflow["extra"]["ltx23_ingredients_profile"] = "wangp-inspired-i2v-15s"
    return workflow


def validate_links(workflow: dict) -> list[str]:
    failures: list[str] = []
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    if len(links) != len(workflow["links"]):
        failures.append("IDs de links duplicados")
    for node in nodes.values():
        for slot, node_input in enumerate(node.get("inputs", [])):
            link_id = node_input.get("link")
            if link_id is None:
                continue
            link = links.get(link_id)
            if link is None or link[3:5] != [node["id"], slot]:
                failures.append(f"entrada inconsistente: node {node['id']} slot {slot}")
        for slot, node_output in enumerate(node.get("outputs", [])):
            for link_id in node_output.get("links") or []:
                link = links.get(link_id)
                if link is None or link[1:3] != [node["id"], slot]:
                    failures.append(f"saida inconsistente: node {node['id']} slot {slot}")
    return failures


def validate_profile(workflow: dict, profile: str) -> None:
    failures = validate_links(workflow)
    nodes = workflow["nodes"]
    strings = json.dumps(workflow, ensure_ascii=False)
    for model in (CHECKPOINT, TEXT_ENCODER, DISTILLED_LORA, INGREDIENTS_LORA):
        if model not in strings:
            failures.append(f"modelo ausente: {model}")
    for obsolete in MODEL_REPLACEMENTS:
        if obsolete in strings:
            failures.append(f"nome de modelo nao adaptado: {obsolete}")
    if workflow.get("extra", {}).get(SCHEMA_MARKER) != SCHEMA_VERSION:
        failures.append("marcador de schema ausente")
    if one(nodes, node_type="RepeatImageBatch")["inputs"][1]["link"] is None:
        failures.append("reference sheet nao repete pelo comprimento do video")
    if one(nodes, node_id=5069, node_type="ResizeImageMaskNode")["widgets_values"][1] != 544:
        failures.append("sheet oficial precisa usar lado curto 544")
    if profile == "official":
        if one(nodes, node_id=5072)["widgets_values"][0] != 241:
            failures.append("perfil oficial precisa usar 241 frames")
        if any(node.get("type") == "LTXVImgToVideoConditionOnly" for node in nodes):
            failures.append("perfil oficial nao deve conter I2V adicional")
    elif profile == "wangp-i2v-15s":
        if one(nodes, node_id=5072)["widgets_values"][0] != 361:
            failures.append("perfil WanGP precisa usar 361 frames")
        i2v = one(nodes, node_type="LTXVImgToVideoConditionOnly")
        guide = one(nodes, node_type="LTXAddVideoICLoRAGuide")
        links = {link[0]: link for link in workflow["links"]}
        if links[guide["inputs"][3]["link"]][1] != i2v["id"]:
            failures.append("I2V precisa alimentar o guia Ingredients")
    else:
        failures.append(f"perfil desconhecido: {profile}")
    if failures:
        raise ValueError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapta o workflow Ingredients oficial para a imagem Vast"
    )
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("official", "wangp-i2v-15s"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workflow = adapt_official(load_json(args.official))
    if args.profile == "wangp-i2v-15s":
        workflow = add_wangp_i2v_profile(workflow)
    validate_profile(workflow, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Workflow Ingredients {args.profile} criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
