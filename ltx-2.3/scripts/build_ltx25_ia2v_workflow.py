#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from build_ia2v_talkvid_workflow import (
    add_link,
    link_map,
    load_json,
    one,
    replace_input_link,
    replace_output_links,
)


SCHEMA_MARKER = "ltx25_ia2v_distilled_schema"
SCHEMA_VERSION = 1
STAGE_ONE_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE_TWO_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
REQUIRED_MODELS = {
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "gemma4_e2b_it_int8_convrot.safetensors",
    "ltx-2.5-video-vae-bf16.safetensors",
    "ltx-2.5-audio-vae-bf16.safetensors",
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}


def _drop_links(subgraph: dict, *link_ids: int) -> None:
    unwanted = set(link_ids)
    subgraph["links"] = [
        link for link in subgraph["links"] if link["id"] not in unwanted
    ]


def build_ia2v(i2v: dict) -> dict:
    workflow = deepcopy(i2v)
    if workflow.get("version") != 0.4:
        raise ValueError("o template LTX-2.5 I2V oficial precisa usar o schema 0.4")

    subgraph = one(workflow["definitions"]["subgraphs"])
    nodes = subgraph["nodes"]
    links = link_map(subgraph)

    duration = one(nodes, node_id=362, node_type="PrimitiveInt")
    audio_vae = one(nodes, node_id=386, node_type="VAELoader")
    empty_audio = one(nodes, node_id=366, node_type="LTXVEmptyLatentAudio")
    frame_count = one(nodes, node_id=378, node_type="ComfyMathExpression")
    fps_number = one(nodes, node_id=359, node_type="ComfyMathExpression")
    width = one(nodes, node_id=372, node_type="PrimitiveInt")
    height = one(nodes, node_id=360, node_type="PrimitiveInt")
    first_concat = one(nodes, node_id=377, node_type="LTXVConcatAVLatent")
    decoded_audio = one(nodes, node_id=358, node_type="LTXVAudioVAEDecode")
    create_video = one(nodes, node_id=370, node_type="CreateVideo")

    # Allow fractional durations and use the same value to crop the narration.
    duration["type"] = "PrimitiveFloat"
    duration["widgets_values"] = [5.0]
    duration["inputs"][0]["type"] = "FLOAT"
    duration["outputs"][0].update(
        {
            "localized_name": "FLOAT",
            "name": "FLOAT",
            "type": "FLOAT",
            "links": [715, 788],
        }
    )
    duration["properties"]["Node name for S&R"] = "PrimitiveFloat"
    links[715]["type"] = "FLOAT"
    links[767]["type"] = "FLOAT"
    duration_input = one(
        [item for item in subgraph["inputs"] if item.get("name") == "value_2"]
    )
    duration_input["type"] = "FLOAT"

    # Remove the generated-audio source and its now-unused incoming links.
    replace_output_links(audio_vae, "VAE", [730, 790])
    replace_output_links(frame_count, "INT", [718])
    replace_output_links(fps_number, "INT", [])
    _drop_links(subgraph, 729, 717, 701)

    # Reuse node 366 as the audio encoder. The zero noise mask makes the audio
    # a frozen conditioning modality instead of something the model denoises.
    empty_audio.update(
        {
            "type": "LTXVAudioVAEEncode",
            "pos": [1410, 5940],
            "size": [300, 110],
            "inputs": [
                {"name": "audio", "type": "AUDIO", "link": 789},
                {"name": "audio_vae", "type": "VAE", "link": 790},
            ],
            "outputs": [
                {
                    "name": "Audio Latent",
                    "type": "LATENT",
                    "links": [791],
                }
            ],
            "properties": {
                "cnr_id": "comfy-core",
                "ver": "0.28.0",
                "Node name for S&R": "LTXVAudioVAEEncode",
            },
            "widgets_values": [],
        }
    )

    trim_audio = {
        "id": 405,
        "type": "TrimAudioDuration",
        "pos": [890, 5940],
        "size": [330, 150],
        "flags": {},
        "order": 48,
        "mode": 0,
        "inputs": [
            {"name": "audio", "type": "AUDIO", "link": 786},
            {
                "name": "start_index",
                "type": "FLOAT",
                "widget": {"name": "start_index"},
                "link": 787,
            },
            {
                "name": "duration",
                "type": "FLOAT",
                "widget": {"name": "duration"},
                "link": 788,
            },
        ],
        "outputs": [
            {"name": "AUDIO", "type": "AUDIO", "links": [789, 712]}
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.28.0",
            "Node name for S&R": "TrimAudioDuration",
        },
        "widgets_values": [0.0, 5.0],
    }
    solid_mask = {
        "id": 407,
        "type": "SolidMask",
        "pos": [1410, 6140],
        "size": [280, 170],
        "flags": {},
        "order": 49,
        "mode": 0,
        "inputs": [
            {
                "name": "width",
                "type": "INT",
                "widget": {"name": "width"},
                "link": 793,
            },
            {
                "name": "height",
                "type": "INT",
                "widget": {"name": "height"},
                "link": 794,
            },
        ],
        "outputs": [{"name": "MASK", "type": "MASK", "links": [792]}],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.28.0",
            "Node name for S&R": "SolidMask",
        },
        "widgets_values": [0.0, 1024, 1024],
    }
    freeze_audio = {
        "id": 408,
        "type": "SetLatentNoiseMask",
        "pos": [1760, 5940],
        "size": [340, 110],
        "flags": {},
        "order": 50,
        "mode": 0,
        "inputs": [
            {"name": "samples", "type": "LATENT", "link": 791},
            {"name": "mask", "type": "MASK", "link": 792},
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": [700]}],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.28.0",
            "Node name for S&R": "SetLatentNoiseMask",
        },
        "widgets_values": [],
    }

    replace_output_links(width, "INT", [687, 793])
    replace_output_links(height, "INT", [688, 794])
    links[700].update({"origin_id": 408, "origin_slot": 0})
    replace_output_links(decoded_audio, "Audio", [])
    replace_input_link(create_video, "audio", 712)
    links[712].update({"origin_id": 405, "origin_slot": 0})

    audio_input = {
        "id": "4a7a7218-7b8f-4756-ad04-b9a4c87914a3",
        "name": "driving_audio",
        "type": "AUDIO",
        "linkIds": [786],
        "label": "driving_audio",
        "pos": [-480.185546875, 6764],
    }
    audio_start_input = {
        "id": "e15db8aa-7aad-4c2e-a8f9-a738135b69b4",
        "name": "audio_start",
        "type": "FLOAT",
        "linkIds": [787],
        "label": "audio_start",
        "pos": [-480.185546875, 6784],
    }
    subgraph["inputs"].extend([audio_input, audio_start_input])
    audio_slot = subgraph["inputs"].index(audio_input)
    audio_start_slot = subgraph["inputs"].index(audio_start_input)
    add_link(subgraph, 786, subgraph["inputNode"]["id"], audio_slot, 405, 0, "AUDIO")
    add_link(
        subgraph,
        787,
        subgraph["inputNode"]["id"],
        audio_start_slot,
        405,
        1,
        "FLOAT",
    )
    add_link(subgraph, 788, 362, 0, 405, 2, "FLOAT")
    add_link(subgraph, 789, 405, 0, 366, 0, "AUDIO")
    add_link(subgraph, 790, 386, 0, 366, 1, "VAE")
    add_link(subgraph, 791, 366, 0, 408, 0, "LATENT")
    add_link(subgraph, 792, 407, 0, 408, 1, "MASK")
    add_link(subgraph, 793, 372, 0, 407, 0, "INT")
    add_link(subgraph, 794, 360, 0, 407, 1, "INT")

    nodes.remove(decoded_audio)
    nodes.extend([trim_audio, solid_mask, freeze_audio])
    subgraph["name"] = "Image + Audio to Video (LTX-2.5 Distilled 8 Steps)"
    subgraph["state"]["lastNodeId"] = 408
    subgraph["state"]["lastLinkId"] = 794
    subgraph["inputNode"]["bounding"][3] += 40
    subgraph.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION

    top_subgraph = one(workflow["nodes"], node_id=398)
    top_subgraph["title"] = "LTX 2.5 IA2V Distilled (8 steps)"
    top_duration = one(
        [item for item in top_subgraph["inputs"] if item.get("name") == "value_2"]
    )
    top_duration["type"] = "FLOAT"
    top_audio_slot = len(top_subgraph["inputs"])
    top_subgraph["inputs"].append(
        {
            "label": "driving_audio",
            "name": "driving_audio",
            "type": "AUDIO",
            "link": 785,
        }
    )
    top_subgraph["inputs"].append(
        {
            "label": "audio_start",
            "name": "audio_start",
            "type": "FLOAT",
            "widget": {"name": "audio_start"},
            "link": None,
        }
    )
    top_subgraph["widgets_values"][0] = (
        "A cinematic close-up of the person from the reference image speaking "
        "naturally and expressively. Precise lip articulation follows the provided "
        "voice, with subtle head motion, stable facial identity, realistic teeth, "
        "natural skin texture, steady lighting, and no subtitles or text."
    )
    top_subgraph["widgets_values"][1] = False
    top_subgraph["widgets_values"][2] = 5.0
    top_subgraph["widgets_values"].append(0.0)
    top_subgraph["size"] = [520, 760]

    load_image = one(workflow["nodes"], node_id=395, node_type="LoadImage")
    load_image["title"] = "Identity / First Frame"
    load_audio = {
        "id": 409,
        "type": "LoadAudio",
        "pos": [1450, 6830],
        "size": [360, 190],
        "flags": {},
        "order": 1,
        "mode": 0,
        "inputs": [],
        "outputs": [{"name": "AUDIO", "type": "AUDIO", "links": [785]}],
        "title": "Driving Audio (original narration)",
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.28.0",
            "Node name for S&R": "LoadAudio",
        },
        "widgets_values": ["audio.wav", None, None],
    }
    workflow["links"].append([785, 409, 0, 398, top_audio_slot, "AUDIO"])

    save_video = one(workflow["nodes"], node_id=75, node_type="SaveVideo")
    save_video["widgets_values"][0] = "video/LTX_2.5_IA2V_Distilled_8Steps"

    about_note = one(workflow["nodes"], node_id=399, node_type="MarkdownNote")
    about_note["title"] = "LTX-2.5 IA2V adaptation"
    about_note["widgets_values"][0] = (
        "# IA2V adaptation\n\n"
        "This Docker workflow adds a frozen driving-audio path to the official "
        "LTX-2.5 I2V graph. The original narration conditions video generation "
        "and is preserved in the output MP4.\n\n"
        + about_note["widgets_values"][0]
    )
    model_note = one(workflow["nodes"], node_id=401, node_type="MarkdownNote")
    model_note["widgets_values"][0] = (
        "**Docker preset:** all LTX-2.5 weights listed here are downloaded when "
        "`DOWNLOAD_LTX25_MODELS_ON_START=1`. Prompt enhancement still starts "
        "disabled for faster first tests.\n\n"
        + model_note["widgets_values"][0]
    )

    usage_note = {
        "id": 410,
        "type": "MarkdownNote",
        "pos": [1350, 7100],
        "size": [710, 520],
        "flags": {},
        "order": 8,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": "LTX-2.5 IA2V distilled",
        "properties": {},
        "widgets_values": [
            "## Image + narracao -> video\n\n"
            "- Adaptacao do I2V oficial LTX-2.5 para audio congelado.\n"
            "- O audio recortado e codificado pelo Audio VAE com noise mask `0`; "
            "ele condiciona as duas etapas e nao e regenerado.\n"
            "- O MP4 recebe diretamente o audio original recortado.\n"
            "- Agenda destilada oficial: 8 passos na primeira etapa e 3 na etapa "
            "de upscale; CFG de video/audio em `1`.\n"
            "- Prompt enhancer fica desligado por padrao, mas seu Gemma 4 E2B "
            "e baixado quando os modelos 2.5 estao habilitados.\n"
            "- Comece com 5 segundos e uma foto frontal bem iluminada."
        ],
    }
    workflow["nodes"].extend([load_audio, usage_note])
    workflow["last_node_id"] = 410
    workflow["last_link_id"] = 794
    workflow.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION
    return workflow


def validate_ia2v(workflow: dict) -> None:
    failures: list[str] = []
    if workflow.get("extra", {}).get(SCHEMA_MARKER) != SCHEMA_VERSION:
        failures.append("marcador de schema ausente")

    subgraph = one(workflow["definitions"]["subgraphs"])
    nodes = subgraph["nodes"]
    links = link_map(subgraph)
    strings = set()

    def collect(value: object) -> None:
        if isinstance(value, str):
            strings.add(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(workflow)
    if REQUIRED_MODELS - strings:
        failures.append("modelos LTX-2.5 obrigatorios ausentes")

    stage_one = one(nodes, node_id=397, node_type="ManualSigmas")
    stage_two = one(nodes, node_id=396, node_type="ManualSigmas")
    if stage_one["widgets_values"] != [STAGE_ONE_SIGMAS]:
        failures.append("agenda destilada de 8 passos alterada")
    if stage_two["widgets_values"] != [STAGE_TWO_SIGMAS]:
        failures.append("agenda destilada da segunda etapa alterada")
    for node_id in (388, 391):
        if one(nodes, node_id=node_id, node_type="LTXVDualCFGGuider")[
            "widgets_values"
        ] != [1, 1]:
            failures.append(f"CFG destilado inesperado no node {node_id}")

    encoded = one(nodes, node_id=366, node_type="LTXVAudioVAEEncode")
    frozen = one(nodes, node_id=408, node_type="SetLatentNoiseMask")
    mask = one(nodes, node_id=407, node_type="SolidMask")
    trim = one(nodes, node_id=405, node_type="TrimAudioDuration")
    create = one(nodes, node_id=370, node_type="CreateVideo")
    if mask["widgets_values"][0] != 0.0:
        failures.append("noise mask do audio precisa ser zero")
    if links[encoded["inputs"][0]["link"]]["origin_id"] != 405:
        failures.append("audio recortado nao alimenta o Audio VAE Encode")
    if links[frozen["outputs"][0]["links"][0]]["target_id"] != 377:
        failures.append("audio congelado nao alimenta o primeiro concat AV")
    create_audio = one(
        [item for item in create["inputs"] if item.get("name") == "audio"]
    )["link"]
    if links[create_audio]["origin_id"] != 405:
        failures.append("CreateVideo nao recebe o audio original recortado")
    if any(node.get("type") == "LTXVAudioVAEDecode" for node in nodes):
        failures.append("Audio VAE Decode nao deve participar da saida")

    top = one(workflow["nodes"], node_id=398)
    if top["widgets_values"][1] is not False:
        failures.append("prompt enhancer opcional precisa iniciar desligado")
    top_audio = one(
        [item for item in top["inputs"] if item.get("name") == "driving_audio"]
    )
    outer_links = {item[0]: item for item in workflow["links"]}
    if top_audio.get("link") != 785 or outer_links[785][1] != 409:
        failures.append("LoadAudio nao esta ligado ao subgrafo")

    if len(links) != len(subgraph["links"]):
        failures.append("IDs de links internos duplicados")
    for node in nodes:
        for slot, node_input in enumerate(node.get("inputs", [])):
            link_id = node_input.get("link")
            if link_id is None:
                continue
            link = links.get(link_id)
            if link is None or link["target_id"] != node["id"] or link["target_slot"] != slot:
                failures.append(f"entrada inconsistente: node {node['id']} slot {slot}")
        for slot, node_output in enumerate(node.get("outputs", [])):
            for link_id in node_output.get("links") or []:
                link = links.get(link_id)
                if link is None or link["origin_id"] != node["id"] or link["origin_slot"] != slot:
                    failures.append(f"saida inconsistente: node {node['id']} slot {slot}")
    for slot, subgraph_input in enumerate(subgraph.get("inputs", [])):
        for link_id in subgraph_input.get("linkIds", []):
            link = links.get(link_id)
            if (
                link is None
                or link["origin_id"] != subgraph["inputNode"]["id"]
                or link["origin_slot"] != slot
            ):
                failures.append(f"entrada do subgrafo inconsistente: {subgraph_input['name']}")

    if failures:
        raise ValueError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cria o workflow LTX-2.5 IA2V destilado de 8 passos"
    )
    parser.add_argument("--i2v", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workflow = build_ia2v(load_json(args.i2v))
    validate_ia2v(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Workflow LTX-2.5 IA2V destilado criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
