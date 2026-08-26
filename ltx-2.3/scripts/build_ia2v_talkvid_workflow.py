#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


ID_LORA_NAME = "ltx-2.3-id-lora-talkvid-3k.safetensors"
SCHEMA_MARKER = "ltx23_ia2v_talkvid_schema"
SCHEMA_VERSION = 1


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def one(items: list[dict], *, node_id: int | None = None, node_type: str | None = None) -> dict:
    matches = [
        item
        for item in items
        if (node_id is None or item.get("id") == node_id)
        and (node_type is None or item.get("type") == node_type)
    ]
    if len(matches) != 1:
        wanted = f"id={node_id}, type={node_type}"
        raise ValueError(f"esperado exatamente um node ({wanted}); encontrados {len(matches)}")
    return matches[0]


def link_map(subgraph: dict) -> dict[int, dict]:
    return {link["id"]: link for link in subgraph["links"]}


def add_link(
    subgraph: dict,
    link_id: int,
    origin_id: int | str,
    origin_slot: int,
    target_id: int | str,
    target_slot: int,
    link_type: str,
) -> None:
    subgraph["links"].append(
        {
            "id": link_id,
            "origin_id": origin_id,
            "origin_slot": origin_slot,
            "target_id": target_id,
            "target_slot": target_slot,
            "type": link_type,
        }
    )


def replace_input_link(node: dict, input_name: str, link_id: int) -> None:
    target = one([item for item in node["inputs"] if item.get("name") == input_name])
    target["link"] = link_id


def replace_output_links(node: dict, output_name: str, links: list[int]) -> None:
    target = one([item for item in node["outputs"] if item.get("name") == output_name])
    target["links"] = links


def build_hybrid(ia2v: dict, id_lora: dict) -> dict:
    workflow = deepcopy(ia2v)
    if workflow.get("version") != 0.4 or id_lora.get("version") != 0.4:
        raise ValueError("os templates oficiais precisam usar o schema 0.4")

    subgraph = one(workflow["definitions"]["subgraphs"], node_type=None)
    id_subgraph = one(id_lora["definitions"]["subgraphs"], node_type=None)
    nodes = subgraph["nodes"]
    id_nodes = id_subgraph["nodes"]

    distilled = one(nodes, node_id=293, node_type="LoraLoaderModelOnly")
    stage_one_guider = one(nodes, node_id=315, node_type="CFGGuider")
    positive_encode = one(nodes, node_id=306, node_type="CLIPTextEncode")
    negative_encode = one(nodes, node_id=314, node_type="CLIPTextEncode")
    conditioning = one(nodes, node_id=307, node_type="LTXVConditioning")
    audio_vae_loader = one(nodes, node_id=335, node_type="LTXVAudioVAELoader")
    driving_trim = one(nodes, node_id=332, node_type="TrimAudioDuration")
    decoded_audio = one(nodes, node_id=303, node_type="LTXVAudioVAEDecode")
    create_video = one(nodes, node_id=312, node_type="CreateVideo")

    id_lora_loader = deepcopy(one(id_nodes, node_id=346, node_type="LoraLoaderModelOnly"))
    reference_audio = deepcopy(one(id_nodes, node_id=349, node_type="LTXVReferenceAudio"))
    reference_trim = deepcopy(driving_trim)

    id_lora_loader.update({"id": 350, "pos": [870, 3830], "order": 50})
    replace_input_link(id_lora_loader, "model", 766)
    id_lora_name_input = one(
        [item for item in id_lora_loader["inputs"] if item.get("name") == "lora_name"]
    )
    id_lora_name_input["link"] = None
    id_lora_loader["widgets_values"] = [ID_LORA_NAME, 1.0]
    replace_output_links(id_lora_loader, "MODEL", [767])

    reference_trim.update(
        {
            "id": 351,
            "title": "Voice identity reference (first 5 seconds)",
            "pos": [410, 5010],
            "order": 51,
        }
    )
    replace_input_link(reference_trim, "audio", 763)
    replace_input_link(reference_trim, "start_index", 764)
    duration_input = one(
        [item for item in reference_trim["inputs"] if item.get("name") == "duration"]
    )
    duration_input["link"] = None
    reference_trim["widgets_values"] = [0.0, 5.0]
    replace_output_links(reference_trim, "AUDIO", [765])

    reference_audio.update({"id": 352, "pos": [2000, 3430], "order": 52})
    for input_name, link_id in (
        ("model", 767),
        ("positive", 682),
        ("negative", 683),
        ("reference_audio", 765),
        ("audio_vae", 768),
    ):
        replace_input_link(reference_audio, input_name, link_id)
    reference_audio["widgets_values"] = [3.0, 0.0, 1.0]
    reference_audio["outputs"][0]["links"] = [647]
    reference_audio["outputs"][1]["links"] = [769]
    reference_audio["outputs"][2]["links"] = [770]

    replace_output_links(distilled, "MODEL", [644, 757, 766])
    replace_input_link(stage_one_guider, "model", 647)
    replace_output_links(positive_encode, "CONDITIONING", [682])
    replace_output_links(negative_encode, "CONDITIONING", [683])
    replace_input_link(conditioning, "positive", 769)
    replace_input_link(conditioning, "negative", 770)
    replace_output_links(audio_vae_loader, "Audio VAE", [680, 718, 768])
    replace_output_links(driving_trim, "AUDIO", [708, 696])
    replace_output_links(decoded_audio, "Audio", [])
    replace_input_link(create_video, "audio", 696)

    links = link_map(subgraph)
    links[647]["origin_id"] = 352
    links[647]["origin_slot"] = 0
    links[682]["target_id"] = 352
    links[682]["target_slot"] = 1
    links[683]["target_id"] = 352
    links[683]["target_slot"] = 2
    links[696]["origin_id"] = 332
    links[696]["origin_slot"] = 0

    audio_input = one([item for item in subgraph["inputs"] if item.get("name") == "audio"])
    audio_input["label"] = "driving_audio"
    audio_input["linkIds"] = [709, 763]
    audio_start = one(
        [item for item in subgraph["inputs"] if item.get("name") == "start_index"]
    )
    audio_start["linkIds"] = [733, 764]

    add_link(
        subgraph,
        763,
        subgraph["inputNode"]["id"],
        subgraph["inputs"].index(audio_input),
        351,
        0,
        "AUDIO",
    )
    add_link(
        subgraph,
        764,
        subgraph["inputNode"]["id"],
        subgraph["inputs"].index(audio_start),
        351,
        1,
        "FLOAT",
    )
    add_link(subgraph, 765, 351, 0, 352, 3, "AUDIO")
    add_link(subgraph, 766, 293, 0, 350, 0, "MODEL")
    add_link(subgraph, 767, 350, 0, 352, 0, "MODEL")
    add_link(subgraph, 768, 335, 0, 352, 4, "VAE")
    add_link(subgraph, 769, 352, 1, 307, 0, "CONDITIONING")
    add_link(subgraph, 770, 352, 2, 307, 1, "CONDITIONING")

    nodes.extend([id_lora_loader, reference_trim, reference_audio])
    subgraph["name"] = "Video Generation (LTX-2.3 IA2V + TalkVid)"
    subgraph["state"]["lastNodeId"] = 352
    subgraph["state"]["lastLinkId"] = 770
    subgraph.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION

    top_subgraph = one(workflow["nodes"], node_id=340)
    top_subgraph["title"] = "LTX 2.3 IA2V + TalkVid 3K"
    top_audio = one([item for item in top_subgraph["inputs"] if item.get("name") == "audio"])
    top_audio["label"] = "driving_audio"
    top_subgraph["size"] = [420, 650]

    load_audio = one(workflow["nodes"], node_id=276, node_type="LoadAudio")
    load_audio["title"] = "Driving Audio (original narration)"
    save_video = one(workflow["nodes"], node_id=341, node_type="SaveVideo")
    save_video["widgets_values"][0] = "video/LTX_2.3_ia2v_talkvid"

    model_note = one(workflow["nodes"], node_id=103, node_type="MarkdownNote")
    note_text = model_note["widgets_values"][0]
    if ID_LORA_NAME not in note_text:
        anchor = "**loras**\n\n"
        addition = (
            "- [ltx-2.3-id-lora-talkvid-3k.safetensors]"
            "(https://huggingface.co/Comfy-Org/ltx-2.3/resolve/main/"
            "split_files/loras/ltx-2.3-id-lora-talkvid-3k.safetensors)\n"
        )
        model_note["widgets_values"][0] = note_text.replace(anchor, anchor + addition, 1)

    usage_note = deepcopy(model_note)
    usage_note.update(
        {
            "id": 353,
            "title": "IA2V + TalkVid (experimental)",
            "pos": [1040, 3530],
            "size": [620, 520],
            "order": 6,
            "widgets_values": [
                "## IA2V + TalkVid 3K\n\n"
                "- **Load Image:** rosto/identidade visual.\n"
                "- **Driving Audio:** narracao completa usada para dirigir o video e copiada "
                "para o MP4 final.\n"
                "- **audio_start:** tambem escolhe o inicio do trecho de identidade vocal.\n"
                "- O TalkVid recebe 5 segundos desse audio, com `strength=1.0` e "
                "`identity_guidance=3.0`.\n"
                "- O TalkVid atua somente no primeiro estagio; o segundo permanece igual ao "
                "IA2V oficial.\n\n"
                "Este arranjo combina nodes oficiais, mas a combinacao IA2V + ID-LoRA e "
                "experimental. Teste primeiro 6-8 segundos."
            ],
        }
    )
    workflow["nodes"].append(usage_note)
    workflow["last_node_id"] = 353
    workflow["last_link_id"] = max(workflow.get("last_link_id", 0), 770)
    workflow.setdefault("extra", {})[SCHEMA_MARKER] = SCHEMA_VERSION
    return workflow


def validate_hybrid(workflow: dict) -> None:
    failures: list[str] = []
    subgraph = workflow["definitions"]["subgraphs"][0]
    nodes = subgraph["nodes"]
    links = link_map(subgraph)

    if workflow.get("extra", {}).get(SCHEMA_MARKER) != SCHEMA_VERSION:
        failures.append("marcador de schema ausente")
    if sum(node.get("type") == "LTXVReferenceAudio" for node in nodes) != 1:
        failures.append("LTXVReferenceAudio precisa aparecer exatamente uma vez")
    id_loaders = [
        node
        for node in nodes
        if node.get("type") == "LoraLoaderModelOnly"
        and ID_LORA_NAME in node.get("widgets_values", [])
    ]
    if len(id_loaders) != 1 or id_loaders[0].get("widgets_values", [None, None])[1] != 1.0:
        failures.append("TalkVid 3K precisa estar configurado com strength=1.0")

    reference = one(nodes, node_type="LTXVReferenceAudio")
    if reference.get("widgets_values") != [3.0, 0.0, 1.0]:
        failures.append("identity guidance precisa ser [3.0, 0.0, 1.0]")
    reference_trim = one(nodes, node_id=351, node_type="TrimAudioDuration")
    if reference_trim.get("widgets_values", [None, None])[1] != 5.0:
        failures.append("a referencia TalkVid precisa ter 5 segundos")

    create_video = one(nodes, node_type="CreateVideo")
    create_audio_link = one(
        [item for item in create_video["inputs"] if item.get("name") == "audio"]
    )["link"]
    if links[create_audio_link]["origin_id"] != 332:
        failures.append("CreateVideo nao recebe a narracao original recortada")

    stage_one = one(nodes, node_id=315, node_type="CFGGuider")
    stage_two = one(nodes, node_id=290, node_type="CFGGuider")
    stage_one_link = one(
        [item for item in stage_one["inputs"] if item.get("name") == "model"]
    )["link"]
    stage_two_link = one(
        [item for item in stage_two["inputs"] if item.get("name") == "model"]
    )["link"]
    if links[stage_one_link]["origin_id"] != reference["id"]:
        failures.append("TalkVid/ReferenceAudio nao alimenta o primeiro estagio")
    if links[stage_two_link]["origin_id"] != 293:
        failures.append("o segundo estagio deve continuar no distilled LoRA oficial")

    if len(links) != len(subgraph["links"]):
        failures.append("IDs de links duplicados")
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
    for slot, subgraph_output in enumerate(subgraph.get("outputs", [])):
        for link_id in subgraph_output.get("linkIds", []):
            link = links.get(link_id)
            if (
                link is None
                or link["target_id"] != subgraph["outputNode"]["id"]
                or link["target_slot"] != slot
            ):
                failures.append(f"saida do subgrafo inconsistente: {subgraph_output['name']}")

    if failures:
        raise ValueError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cria o workflow LTX 2.3 IA2V + TalkVid a partir dos templates oficiais"
    )
    parser.add_argument("--ia2v", type=Path, required=True)
    parser.add_argument("--id-lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workflow = build_hybrid(load_json(args.ia2v), load_json(args.id_lora))
    validate_hybrid(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Workflow IA2V + TalkVid criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
