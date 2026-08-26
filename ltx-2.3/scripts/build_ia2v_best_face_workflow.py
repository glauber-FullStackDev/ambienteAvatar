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


BEST_FACE_NAME = "Best_FaceID_v1.0_LoRA.safetensors"
SCHEMA_MARKER = "ltx23_ia2v_best_face_schema"
SCHEMA_VERSION = 1


def build_best_face(ia2v: dict) -> dict:
    workflow = deepcopy(ia2v)
    if workflow.get("version") != 0.4:
        raise ValueError("o template IA2V oficial precisa usar o schema 0.4")

    subgraph = one(workflow["definitions"]["subgraphs"])
    nodes = subgraph["nodes"]
    distilled = one(nodes, node_id=293, node_type="LoraLoaderModelOnly")
    stage_one_guider = one(nodes, node_id=315, node_type="CFGGuider")
    conditioning = one(nodes, node_id=307, node_type="LTXVConditioning")
    concat_latent = one(nodes, node_id=326, node_type="LTXVConcatAVLatent")
    video_vae = one(nodes, node_id=300, node_type="Reroute")
    driving_trim = one(nodes, node_id=332, node_type="TrimAudioDuration")
    decoded_audio = one(nodes, node_id=303, node_type="LTXVAudioVAEDecode")
    create_video = one(nodes, node_id=312, node_type="CreateVideo")

    best_face_loader = deepcopy(distilled)
    best_face_loader.update({"id": 350, "pos": [1370, 4240], "order": 50})
    replace_input_link(best_face_loader, "model", 763)
    lora_name_input = one(
        [item for item in best_face_loader["inputs"] if item.get("name") == "lora_name"]
    )
    lora_name_input["link"] = None
    best_face_loader["widgets_values"] = [BEST_FACE_NAME, 1.0]
    best_face_loader["properties"]["models"] = [
        {
            "name": BEST_FACE_NAME,
            "url": (
                "https://huggingface.co/Alissonerdx/LTX-Best-Face-ID/resolve/"
                "dac8cc2dd6e3cec350810ff1336d04fc120e9561/"
                "Best_FaceID_v1.0_LoRA.safetensors"
            ),
            "directory": "loras",
        }
    ]
    replace_output_links(best_face_loader, "MODEL", [764])

    identity_node = {
        "id": 351,
        "type": "LTXIdentityOverlapConditioning",
        "pos": [2000, 3330],
        "size": [470, 390],
        "flags": {},
        "order": 51,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": 764},
            {"name": "positive", "type": "CONDITIONING", "link": 765},
            {"name": "negative", "type": "CONDITIONING", "link": 766},
            {"name": "vae", "type": "VAE", "link": 768},
            {"name": "latent", "type": "LATENT", "link": 767},
            {"name": "reference_image", "type": "IMAGE", "link": 769},
        ],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": [647]},
            {
                "name": "positive",
                "type": "CONDITIONING",
                "links": [648, 655],
            },
            {
                "name": "negative",
                "type": "CONDITIONING",
                "links": [649, 656],
            },
            {"name": "latent", "type": "LATENT", "links": [654]},
            {"name": "debug", "type": "STRING", "links": None},
            {"name": "ref_preview", "type": "IMAGE", "links": None},
            {"name": "crop_overlay", "type": "IMAGE", "links": None},
        ],
        "properties": {
            "cnr_id": "bfsnodes",
            "ver": "0a2553869254eef4f3f735fdd9fea04614c3dd7e",
            "Node name for S&R": "LTXIdentityOverlapConditioning",
            "ue_properties": {
                "widget_ue_connectable": {},
                "input_ue_unconnectable": {},
                "version": "7.7",
            },
        },
        "widgets_values": [
            2.0,
            1.0,
            "match_target",
            False,
            "center",
            "overlap",
            1.0,
            0,
        ],
    }

    # The Best Face-ID recipe uses the image as separate reference tokens, not
    # as a rendered first frame. Keep both I2V stages in bypass mode.
    t2v_switch = one(nodes, node_id=305, node_type="PrimitiveBoolean")
    t2v_switch["title"] = "Best Face-ID reference mode (keep enabled)"
    t2v_switch["widgets_values"] = [True]
    distilled["widgets_values"][1] = 0.6

    replace_output_links(distilled, "MODEL", [644, 757, 763])
    replace_input_link(stage_one_guider, "model", 647)
    replace_output_links(conditioning, "positive", [765])
    replace_output_links(conditioning, "negative", [766])
    replace_output_links(concat_latent, "latent", [767])
    replace_output_links(video_vae, "", [662, 663, 694, 768])
    replace_output_links(driving_trim, "AUDIO", [708, 696])
    replace_output_links(decoded_audio, "Audio", [])
    replace_input_link(create_video, "audio", 696)

    links = link_map(subgraph)
    links[647].update({"origin_id": 351, "origin_slot": 0})
    links[648].update({"origin_id": 351, "origin_slot": 1})
    links[655].update({"origin_id": 351, "origin_slot": 1})
    links[649].update({"origin_id": 351, "origin_slot": 2})
    links[656].update({"origin_id": 351, "origin_slot": 2})
    links[654].update({"origin_id": 351, "origin_slot": 3})
    links[696].update({"origin_id": 332, "origin_slot": 0})

    identity_input = {
        "id": "7f55989e-cb48-4e2a-904c-c884a7943843",
        "name": "identity_reference",
        "type": "IMAGE",
        "linkIds": [769],
        "localized_name": "identity_reference",
        "label": "identity_reference",
        "pos": [-21.01953125, 4024],
    }
    subgraph["inputs"].append(identity_input)
    add_link(subgraph, 763, 293, 0, 350, 0, "MODEL")
    add_link(subgraph, 764, 350, 0, 351, 0, "MODEL")
    add_link(subgraph, 765, 307, 0, 351, 1, "CONDITIONING")
    add_link(subgraph, 766, 307, 1, 351, 2, "CONDITIONING")
    add_link(subgraph, 767, 326, 0, 351, 4, "LATENT")
    add_link(subgraph, 768, 300, 0, 351, 3, "VAE")
    add_link(
        subgraph,
        769,
        subgraph["inputNode"]["id"],
        len(subgraph["inputs"]) - 1,
        351,
        5,
        "IMAGE",
    )

    nodes.extend([best_face_loader, identity_node])
    subgraph["name"] = "Video Generation (LTX-2.3 Best Face-ID + Audio)"
    subgraph["state"]["lastNodeId"] = 351
    subgraph["state"]["lastLinkId"] = 769
    subgraph["inputNode"]["bounding"][3] += 20
    subgraph.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION

    top_subgraph = one(workflow["nodes"], node_id=340)
    top_subgraph["title"] = "LTX 2.3 Best Face-ID + Original Audio"
    top_subgraph["inputs"][0]["label"] = "source_image"
    top_identity_slot = len(top_subgraph["inputs"])
    top_subgraph["inputs"].append(
        {
            "label": "identity_reference",
            "name": "identity_reference",
            "type": "IMAGE",
            "link": 770,
        }
    )
    top_subgraph["size"] = [440, 680]

    load_image = one(workflow["nodes"], node_id=269, node_type="LoadImage")
    load_image["title"] = "Identity Reference (frontal close-up)"
    load_image["outputs"][0]["links"] = [726, 770]
    workflow["links"].append([770, 269, 0, 340, top_identity_slot, "IMAGE"])

    load_audio = one(workflow["nodes"], node_id=276, node_type="LoadAudio")
    load_audio["title"] = "Driving Audio (original narration)"
    save_video = one(workflow["nodes"], node_id=341, node_type="SaveVideo")
    save_video["widgets_values"][0] = "video/LTX_2.3_best_face_id_audio"

    prompt = one(nodes, node_id=319, node_type="PrimitiveStringMultiline")
    if not prompt["widgets_values"][0].startswith("ref_t2v:"):
        prompt["widgets_values"][0] = "ref_t2v: " + prompt["widgets_values"][0]
    prompt_enhance = one(nodes, node_id=349, node_type="PrimitiveBoolean")
    prompt_enhance["widgets_values"] = [False]

    model_note = one(workflow["nodes"], node_id=103, node_type="MarkdownNote")
    note_text = model_note["widgets_values"][0]
    if BEST_FACE_NAME not in note_text:
        anchor = "**loras**\n\n"
        addition = (
            "- [Best_FaceID_v1.0_LoRA.safetensors]"
            "(https://huggingface.co/Alissonerdx/LTX-Best-Face-ID/resolve/"
            "dac8cc2dd6e3cec350810ff1336d04fc120e9561/"
            "Best_FaceID_v1.0_LoRA.safetensors)\n"
        )
        model_note["widgets_values"][0] = note_text.replace(anchor, anchor + addition, 1)

    usage_note = deepcopy(model_note)
    usage_note.update(
        {
            "id": 352,
            "title": "Best Face-ID + original narration",
            "pos": [1060, 3530],
            "size": [650, 590],
            "order": 6,
            "widgets_values": [
                "## Best Face-ID + IA2V audio\n\n"
                "- Use uma referencia frontal, iluminada, peito/rosto e fundo simples.\n"
                "- A mesma imagem entra como referencia separada; ela nao e forçada como "
                "primeiro frame.\n"
                "- A narracao e congelada no latent para dirigir a fala e o arquivo original "
                "vai direto ao MP4.\n"
                "- Prompt manual deve iniciar com `ref_t2v:` e descrever cabelo, olhos, "
                "barba, oculos, formato do rosto, enquadramento e acao.\n"
                "- Valores iniciais: distilled `0.6`, Best Face-ID `1.0`, source ID `2`, "
                "phase `1`, layout `overlap`, reference guidance `1`, offset `0`.\n\n"
                "Workflow experimental derivado do IA2V oficial e da receita do autor do "
                "Best Face-ID. Comece com 6-8 segundos."
            ],
        }
    )
    workflow["nodes"].append(usage_note)
    workflow["last_node_id"] = 352
    workflow["last_link_id"] = 770
    workflow.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION
    return workflow


def validate_best_face(workflow: dict) -> None:
    failures: list[str] = []
    subgraph = workflow["definitions"]["subgraphs"][0]
    nodes = subgraph["nodes"]
    links = link_map(subgraph)

    if workflow.get("extra", {}).get(SCHEMA_MARKER) != SCHEMA_VERSION:
        failures.append("marcador de schema ausente")
    identity_nodes = [
        node for node in nodes if node.get("type") == "LTXIdentityOverlapConditioning"
    ]
    if len(identity_nodes) != 1:
        failures.append("LTXIdentityOverlapConditioning precisa aparecer uma vez")
    else:
        identity = identity_nodes[0]
        if identity.get("widgets_values") != [
            2.0,
            1.0,
            "match_target",
            False,
            "center",
            "overlap",
            1.0,
            0,
        ]:
            failures.append("parametros BFS Best Face-ID inesperados")

    loaders = [
        node
        for node in nodes
        if node.get("type") == "LoraLoaderModelOnly"
        and BEST_FACE_NAME in node.get("widgets_values", [])
    ]
    if len(loaders) != 1 or loaders[0]["widgets_values"][1] != 1.0:
        failures.append("Best Face-ID precisa estar configurado com strength=1.0")
    if one(nodes, node_id=293)["widgets_values"][1] != 0.6:
        failures.append("distilled LoRA precisa estar configurado com strength=0.6")
    if one(nodes, node_id=305)["widgets_values"] != [True]:
        failures.append("I2V precisa permanecer em bypass no modo reference-to-video")

    create_video = one(nodes, node_type="CreateVideo")
    create_audio_link = one(
        [item for item in create_video["inputs"] if item.get("name") == "audio"]
    )["link"]
    if links[create_audio_link]["origin_id"] != 332:
        failures.append("CreateVideo nao recebe a narracao original recortada")
    stage_one_model = one(nodes, node_id=315)["inputs"][0]["link"]
    stage_two_model = one(nodes, node_id=290)["inputs"][0]["link"]
    if links[stage_one_model]["origin_id"] != 351:
        failures.append("Best Face-ID nao alimenta o primeiro estagio")
    if links[stage_two_model]["origin_id"] != 293:
        failures.append("segundo estagio precisa continuar no distilled LoRA")
    prompt = one(nodes, node_id=319)["widgets_values"][0]
    if not prompt.startswith("ref_t2v:"):
        failures.append("prompt padrao nao usa o prefixo ref_t2v")

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

    outer_links = {link[0]: link for link in workflow["links"]}
    top_subgraph = one(workflow["nodes"], node_id=340)
    identity_input = one(
        [item for item in top_subgraph["inputs"] if item.get("name") == "identity_reference"]
    )
    if identity_input.get("link") != 770 or outer_links.get(770, [None, None])[1] != 269:
        failures.append("imagem de identidade nao esta ligada ao subgrafo")

    if failures:
        raise ValueError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cria o workflow LTX 2.3 Best Face-ID com narracao congelada"
    )
    parser.add_argument("--ia2v", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workflow = build_best_face(load_json(args.ia2v))
    validate_best_face(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Workflow IA2V + Best Face-ID criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
