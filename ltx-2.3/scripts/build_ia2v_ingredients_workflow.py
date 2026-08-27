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


INGREDIENTS_NAME = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
SCHEMA_MARKER = "ltx23_ia2v_ingredients_schema"
SCHEMA_VERSION = 3
CORRECTED_PROFILE = "corrected"
LEGACY_PROFILE = "legacy-v2"
PROFILE_SCHEMA = {CORRECTED_PROFILE: SCHEMA_VERSION, LEGACY_PROFILE: 2}


def build_ingredients(ia2v: dict, profile: str = CORRECTED_PROFILE) -> dict:
    if profile not in PROFILE_SCHEMA:
        raise ValueError(f"perfil Ingredients desconhecido: {profile}")
    legacy = profile == LEGACY_PROFILE
    schema_version = PROFILE_SCHEMA[profile]
    width, height, sheet_short_side = (768, 448, 448) if legacy else (960, 544, 544)
    workflow = deepcopy(ia2v)
    if workflow.get("version") != 0.4:
        raise ValueError("o template IA2V oficial precisa usar o schema 0.4")

    subgraph = one(workflow["definitions"]["subgraphs"])
    nodes = subgraph["nodes"]
    links = link_map(subgraph)

    distilled = one(nodes, node_id=293, node_type="LoraLoaderModelOnly")
    stage_one_guider = one(nodes, node_id=315, node_type="CFGGuider")
    conditioning = one(nodes, node_id=307, node_type="LTXVConditioning")
    concat_latent = one(nodes, node_id=326, node_type="LTXVConcatAVLatent")
    crop_guides = one(nodes, node_id=292, node_type="LTXVCropGuides")
    stage_one_separate = one(nodes, node_id=309, node_type="LTXVSeparateAVLatent")
    latent_upscaler = one(nodes, node_id=295, node_type="LTXVLatentUpsampler")
    video_vae = one(nodes, node_id=300, node_type="Reroute")
    frame_count = one(nodes, node_id=329, node_type="ComfyMathExpression")
    driving_trim = one(nodes, node_id=332, node_type="TrimAudioDuration")
    image_to_video = one(nodes, node_id=325, node_type="LTXVImgToVideoInplace")
    decoded_audio = one(nodes, node_id=303, node_type="LTXVAudioVAEDecode")
    create_video = one(nodes, node_id=312, node_type="CreateVideo")

    ingredients_loader = {
        "id": 350,
        "type": "LTXICLoRALoaderModelOnly",
        "pos": [1300, 3880],
        "size": [500, 130],
        "flags": {},
        "order": 50,
        "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": 763}],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": [647]},
            {"name": "latent_downscale_factor", "type": "FLOAT", "links": [764]},
        ],
        "properties": {
            "aux_id": "Lightricks/ComfyUI-LTXVideo",
            "ver": "15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d",
            "Node name for S&R": "LTXICLoRALoaderModelOnly",
            "models": [
                {
                    "name": INGREDIENTS_NAME,
                    "url": (
                        "https://huggingface.co/Comfy-Org/ltx-2.3/resolve/"
                        "ae386fe2afb1b06c1a47afdc78f6835e3f5fcf91/"
                        "split_files/loras/"
                        "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
                    ),
                    "directory": "loras",
                }
            ],
        },
        "widgets_values": [INGREDIENTS_NAME, 1.0],
    }
    resize_sheet = {
        "id": 351,
        "type": "ResizeImageMaskNode",
        "pos": [1220, 5330],
        "size": [300, 140],
        "flags": {},
        "order": 51,
        "mode": 0,
        "inputs": [{"name": "input", "type": "IMAGE,MASK", "link": 769}],
        "outputs": [{"name": "resized", "type": "IMAGE", "links": [765]}],
        "properties": {
            "aux_id": "Lightricks/ComfyUI-LTXVideo",
            "ver": "15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d",
            "Node name for S&R": "ResizeImageMaskNode",
        },
        "widgets_values": ["scale shorter dimension", sheet_short_side, "lanczos"],
    }
    repeat_sheet = {
        "id": 352,
        "type": "RepeatImageBatch",
        "pos": [1580, 5330],
        "size": [260, 100],
        "flags": {},
        "order": 52,
        "mode": 0,
        "inputs": [
            {"name": "image", "type": "IMAGE", "link": 765},
            {
                "name": "amount",
                "type": "INT",
                "widget": {"name": "amount"},
                "link": 766,
            },
        ],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [767]}],
        "properties": {"Node name for S&R": "RepeatImageBatch"},
        "widgets_values": [1],
    }
    guide = {
        "id": 353,
        "type": "LTXAddVideoICLoRAGuide",
        "pos": [2030, 3860],
        "size": [490, 420],
        "flags": {},
        "order": 53,
        "mode": 0,
        "inputs": [
            {"name": "positive", "type": "CONDITIONING", "link": 771},
            {"name": "negative", "type": "CONDITIONING", "link": 772},
            {"name": "vae", "type": "VAE", "link": 773},
            {"name": "latent", "type": "LATENT", "link": 774},
            {"name": "image", "type": "IMAGE", "link": 767},
            {
                "name": "latent_downscale_factor",
                "type": "FLOAT",
                "widget": {"name": "latent_downscale_factor"},
                "link": 764,
            },
        ],
        "outputs": [
            {"name": "positive", "type": "CONDITIONING", "links": [648, 655]},
            {"name": "negative", "type": "CONDITIONING", "links": [649, 656]},
            {"name": "latent", "type": "LATENT", "links": [685]},
        ],
        "properties": {
            "aux_id": "Lightricks/ComfyUI-LTXVideo",
            "ver": "15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d",
            "Node name for S&R": "LTXAddVideoICLoRAGuide",
        },
        "widgets_values": [0.0, 1.0, 1.0, "disabled", False, 256, 64],
    }

    # The corrected profile uses the official spatial bucket. The legacy
    # profile intentionally preserves the previous, smaller schema-2 preset.
    one(nodes, node_id=330, node_type="PrimitiveInt")["widgets_values"] = [width, "fixed"]
    one(nodes, node_id=324, node_type="PrimitiveInt")["widgets_values"] = [height, "fixed"]
    one(nodes, node_id=331, node_type="PrimitiveFloat")["widgets_values"] = [5.0]
    one(nodes, node_id=323, node_type="PrimitiveInt")["widgets_values"] = [24, "fixed"]
    distilled["widgets_values"][1] = 0.5
    one(nodes, node_id=349, node_type="PrimitiveBoolean")["widgets_values"] = [False]
    one(nodes, node_id=315, node_type="CFGGuider")["widgets_values"] = [1]

    prompt = one(nodes, node_id=319, node_type="PrimitiveStringMultiline")
    if legacy:
        prompt["widgets_values"][0] = (
            "Reference sheet: describe each panel in the uploaded sheet: face, hair, "
            "skin tone, glasses, beard, outfit, body shape, key props, and location.\n\n"
            "Generated video: a natural close-up talking-head shot of the person from "
            "the sheet speaking to camera, with stable facial identity, realistic mouth "
            "motion following the driving audio, subtle head movement, natural teeth, "
            "consistent skin texture, and no subtitles or on-screen text."
        )
    else:
        prompt["widgets_values"][0] = (
            "### Reference Sheet Description\n"
            "Describe each panel in the uploaded sheet: face, hair, skin tone, glasses, "
            "beard, outfit, body shape, key props, and location.\n\n"
            "### Target Description\n"
            "A natural close-up talking-head shot of the person from "
            "the sheet speaking to camera, with stable facial identity, realistic mouth "
            "motion following the driving audio, subtle head movement, natural teeth, "
            "consistent skin texture, and no subtitles or on-screen text."
        )

    replace_output_links(distilled, "MODEL", [644, 757, 763])
    replace_input_link(stage_one_guider, "model", 647)
    replace_output_links(conditioning, "positive", [771])
    replace_output_links(conditioning, "negative", [772])
    replace_output_links(image_to_video, "latent", [774])
    replace_output_links(concat_latent, "latent", [654])
    # LTXAddVideoICLoRAGuide appends the reference frames to the latent tail.
    # The base IA2V graph only used LTXVCropGuides to clear conditioning metadata;
    # route its latent output into stage 2 as well so the sheet is never upscaled or
    # decoded as generated video.
    replace_output_links(stage_one_separate, "video_latent", [657])
    replace_output_links(crop_guides, "latent", [660])
    replace_input_link(latent_upscaler, "samples", 660)
    replace_output_links(video_vae, "", [662, 663, 694, 773])
    replace_output_links(frame_count, "INT", [712, 766])
    replace_output_links(driving_trim, "AUDIO", [708, 696])
    replace_output_links(decoded_audio, "Audio", [])
    replace_input_link(create_video, "audio", 696)

    links[647].update({"origin_id": 350, "origin_slot": 0})
    links[648].update({"origin_id": 353, "origin_slot": 0})
    links[655].update({"origin_id": 353, "origin_slot": 0})
    links[649].update({"origin_id": 353, "origin_slot": 1})
    links[656].update({"origin_id": 353, "origin_slot": 1})
    links[685].update({"origin_id": 353, "origin_slot": 2})
    links[660].update({"origin_id": 292, "origin_slot": 2})
    links[696].update({"origin_id": 332, "origin_slot": 0})
    add_link(subgraph, 774, 325, 0, 353, 3, "LATENT")

    sheet_input = {
        "id": "e8d8f733-5579-4452-88df-a31b6a390118",
        "name": "ingredients_reference_sheet",
        "type": "IMAGE,MASK",
        "linkIds": [769],
        "localized_name": "ingredients_reference_sheet",
        "label": "ingredients_reference_sheet",
        "pos": [-21.01953125, 4054],
    }
    subgraph["inputs"].append(sheet_input)
    add_link(subgraph, 763, 293, 0, 350, 0, "MODEL")
    add_link(subgraph, 764, 350, 1, 353, 5, "FLOAT")
    add_link(subgraph, 765, 351, 0, 352, 0, "IMAGE")
    add_link(subgraph, 766, 329, 1, 352, 1, "INT")
    add_link(subgraph, 767, 352, 0, 353, 4, "IMAGE")
    add_link(
        subgraph,
        769,
        subgraph["inputNode"]["id"],
        len(subgraph["inputs"]) - 1,
        351,
        0,
        "IMAGE,MASK",
    )
    add_link(subgraph, 771, 307, 0, 353, 0, "CONDITIONING")
    add_link(subgraph, 772, 307, 1, 353, 1, "CONDITIONING")
    add_link(subgraph, 773, 300, 0, 353, 2, "VAE")

    nodes.extend([ingredients_loader, resize_sheet, repeat_sheet, guide])
    subgraph["name"] = "Video Generation (LTX-2.3 IA2V + Ingredients)"
    subgraph["state"]["lastNodeId"] = 353
    subgraph["state"]["lastLinkId"] = 774
    subgraph["inputNode"]["bounding"][3] += 20
    subgraph.setdefault("extra", {})[SCHEMA_MARKER] = schema_version

    top_subgraph = one(workflow["nodes"], node_id=340)
    top_subgraph["title"] = "LTX 2.3 IA2V + IC-LoRA Ingredients"
    if legacy:
        top_subgraph["title"] += " (legacy schema 2)"
    top_sheet_slot = len(top_subgraph["inputs"])
    top_subgraph["inputs"].append(
        {
            "label": "ingredients_reference_sheet",
            "name": "ingredients_reference_sheet",
            "type": "IMAGE,MASK",
            "link": 770,
        }
    )
    top_subgraph["size"] = [470, 710]

    source_image = one(workflow["nodes"], node_id=269, node_type="LoadImage")
    source_image["title"] = "First Frame / Main Face"
    sheet_image = deepcopy(source_image)
    sheet_image.update(
        {
            "id": 354,
            "title": "Ingredients Reference Sheet",
            "pos": [1410, 1260],
            "order": 1,
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [770]},
                {"name": "MASK", "type": "MASK", "links": []},
            ],
            "widgets_values": ["ingredients_input.jpg", "image"],
        }
    )
    workflow["links"].append([770, 354, 0, 340, top_sheet_slot, "IMAGE,MASK"])

    load_audio = one(workflow["nodes"], node_id=276, node_type="LoadAudio")
    load_audio["title"] = "Driving Audio (original narration)"
    save_video = one(workflow["nodes"], node_id=341, node_type="SaveVideo")
    save_video["widgets_values"][0] = (
        "video/LTX_2.3_ia2v_ingredients_legacy_v2"
        if legacy
        else "video/LTX_2.3_ia2v_ingredients"
    )

    model_note = one(workflow["nodes"], node_id=103, node_type="MarkdownNote")
    note_text = model_note["widgets_values"][0]
    if INGREDIENTS_NAME not in note_text:
        anchor = "**loras**\n\n"
        addition = (
            "- [ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors]"
            "(https://huggingface.co/Comfy-Org/ltx-2.3/resolve/"
            "ae386fe2afb1b06c1a47afdc78f6835e3f5fcf91/split_files/loras/"
            "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors)\n"
        )
        model_note["widgets_values"][0] = note_text.replace(anchor, anchor + addition, 1)

    prompt_help = (
        "- Prompt legado: `Reference sheet:` seguido de `Generated video:`.\n"
        if legacy
        else "- Prompt recomendado: `### Reference Sheet Description` seguido de "
        "`### Target Description`.\n"
    )
    sheet_background_help = (
        "uma imagem composta em fundo preto com "
        if legacy
        else "uma imagem composta com fundos claros ou neutros e "
    )
    usage_note_text = (
        "## IA2V + Ingredients reference sheet\n\n"
        "- **First Frame / Main Face:** imagem inicial do video.\n"
        "- **Driving Audio:** narracao original; ela dirige a fala e vai direto ao MP4.\n"
        "- **Ingredients Reference Sheet:** "
        + sheet_background_help
        + "paineis da pessoa, roupa, props e ambiente.\n"
        "- O workflow repete a sheet como video estatico pelo numero de frames e "
        "aplica o IC-LoRA Ingredients no primeiro estagio.\n"
        + prompt_help
        + "- O `LTXVCropGuides` remove a sheet antes do upscale e do decode.\n"
        f"- Defaults: {width}x{height}, 5s, 24fps, 121 frames, "
        "IC-LoRA strength `1.0`.\n"
        "- Mantenha largura e altura divisiveis por 32 e use uma sheet com a "
        "mesma proporcao do video.\n\n"
        "Use uma sheet sem texto visivel, com paineis limpos. Para identidade, "
        "inclua close-up frontal, perfil/3/4, corpo/roupa e detalhes fixos como "
        "oculos/barba."
    )
    usage_note = deepcopy(model_note)
    usage_note.update(
        {
            "id": 355,
            "title": "IA2V + IC-LoRA Ingredients",
            "pos": [1060, 3530],
            "size": [720, 620],
            "order": 6,
            "widgets_values": [usage_note_text],
        }
    )
    workflow["nodes"].extend([sheet_image, usage_note])
    workflow["last_node_id"] = 355
    workflow["last_link_id"] = max(workflow.get("last_link_id", 0), 774)
    workflow.setdefault("extra", {})[SCHEMA_MARKER] = schema_version
    workflow["extra"]["ltx23_ia2v_ingredients_profile"] = profile
    return workflow


def validate_ingredients(workflow: dict, profile: str = CORRECTED_PROFILE) -> None:
    if profile not in PROFILE_SCHEMA:
        raise ValueError(f"perfil Ingredients desconhecido: {profile}")
    legacy = profile == LEGACY_PROFILE
    schema_version = PROFILE_SCHEMA[profile]
    width, height, sheet_short_side = (768, 448, 448) if legacy else (960, 544, 544)
    failures: list[str] = []
    subgraph = workflow["definitions"]["subgraphs"][0]
    nodes = subgraph["nodes"]
    links = link_map(subgraph)

    if workflow.get("extra", {}).get(SCHEMA_MARKER) != schema_version:
        failures.append(f"marcador de schema precisa ser {schema_version}")
    if workflow.get("extra", {}).get("ltx23_ia2v_ingredients_profile") != profile:
        failures.append(f"marcador de perfil precisa ser {profile}")
    if workflow.get("version") != 0.4:
        failures.append("versao do workflow diferente de 0.4")

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
    if INGREDIENTS_NAME not in strings:
        failures.append("modelo Ingredients ausente do workflow")

    loaders = [
        node for node in nodes if node.get("type") == "LTXICLoRALoaderModelOnly"
    ]
    if len(loaders) != 1 or loaders[0]["widgets_values"] != [INGREDIENTS_NAME, 1.0]:
        failures.append("LTXICLoRALoaderModelOnly precisa usar Ingredients strength=1.0")
    if sum(node.get("type") == "LTXAddVideoICLoRAGuide" for node in nodes) != 1:
        failures.append("LTXAddVideoICLoRAGuide precisa aparecer uma vez")
    if one(nodes, node_id=351).get("type") != "ResizeImageMaskNode":
        failures.append("ResizeImageMaskNode da sheet precisa aparecer no node 351")
    if sum(node.get("type") == "RepeatImageBatch" for node in nodes) != 1:
        failures.append("RepeatImageBatch precisa aparecer uma vez")

    if one(nodes, node_id=330)["widgets_values"][0] != width:
        failures.append(f"largura padrao precisa ser {width}")
    if one(nodes, node_id=324)["widgets_values"][0] != height:
        failures.append(f"altura padrao precisa ser {height}")
    if one(nodes, node_id=331)["widgets_values"] != [5.0]:
        failures.append("duracao padrao precisa ser 5s")
    if one(nodes, node_id=323)["widgets_values"][0] != 24:
        failures.append("fps padrao precisa ser 24")
    if one(nodes, node_id=349)["widgets_values"] != [False]:
        failures.append("prompt enhancer precisa iniciar desligado")
    if one(nodes, node_id=351)["widgets_values"] != [
        "scale shorter dimension",
        sheet_short_side,
        "lanczos",
    ]:
        failures.append(f"sheet precisa usar lado curto {sheet_short_side}")
    prompt_text = one(nodes, node_id=319)["widgets_values"][0]
    prompt_markers = (
        ("Reference sheet:", "Generated video:")
        if legacy
        else ("### Reference Sheet Description", "### Target Description")
    )
    if not all(marker in prompt_text for marker in prompt_markers):
        failures.append("prompt nao corresponde ao perfil selecionado")

    create_video = one(nodes, node_type="CreateVideo")
    create_audio_link = one(
        [item for item in create_video["inputs"] if item.get("name") == "audio"]
    )["link"]
    if links[create_audio_link]["origin_id"] != 332:
        failures.append("CreateVideo nao recebe a narracao original recortada")

    stage_one_model = one(nodes, node_id=315)["inputs"][0]["link"]
    stage_two_model = one(nodes, node_id=290)["inputs"][0]["link"]
    if links[stage_one_model]["origin_id"] != 350:
        failures.append("Ingredients nao alimenta o modelo do primeiro estagio")
    if links[stage_two_model]["origin_id"] != 293:
        failures.append("segundo estagio precisa continuar no distilled LoRA")
    guide_latent = one(nodes, node_id=353)["inputs"][3]["link"]
    if links[guide_latent]["origin_id"] != 325:
        failures.append("guia Ingredients precisa receber o latent de video antes do audio")
    concat_video = one(nodes, node_id=326)["inputs"][0]["link"]
    if links[concat_video]["origin_id"] != 353:
        failures.append("audio precisa ser concatenado depois do guia Ingredients")
    crop_node = one(nodes, node_id=292, node_type="LTXVCropGuides")
    crop_latent = one(
        [output for output in crop_node["outputs"] if output.get("name") == "latent"]
    ).get("links") or []
    upscaler_node = one(nodes, node_id=295, node_type="LTXVLatentUpsampler")
    upscaler_samples = one(
        [item for item in upscaler_node["inputs"] if item.get("name") == "samples"]
    )["link"]
    if crop_latent != [upscaler_samples]:
        failures.append("saida latent do LTXVCropGuides precisa alimentar o upscaler")
    if (
        links[upscaler_samples]["origin_id"] != 292
        or links[upscaler_samples]["origin_slot"] != 2
    ):
        failures.append("upscaler precisa receber o latent recortado pelo LTXVCropGuides")
    separated_node = one(nodes, node_id=309, node_type="LTXVSeparateAVLatent")
    separated_video_links = one(
        [
            output
            for output in separated_node["outputs"]
            if output.get("name") == "video_latent"
        ]
    ).get("links") or []
    if upscaler_samples in separated_video_links:
        failures.append("latent bruto do primeiro estagio nao pode contornar o CropGuides")
    repeat_amount = one(nodes, node_id=352)["inputs"][1]["link"]
    if links[repeat_amount]["origin_id"] != 329:
        failures.append("sheet estatica precisa repetir pelo numero de frames")

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
    sheet_input = one(
        [
            item
            for item in top_subgraph["inputs"]
            if item.get("name") == "ingredients_reference_sheet"
        ]
    )
    if sheet_input.get("link") != 770 or outer_links.get(770, [None, None])[1] != 354:
        failures.append("sheet Ingredients nao esta ligada ao subgrafo")

    if failures:
        raise ValueError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cria o workflow LTX 2.3 IA2V + IC-LoRA Ingredients"
    )
    parser.add_argument("--ia2v", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(CORRECTED_PROFILE, LEGACY_PROFILE),
        default=CORRECTED_PROFILE,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workflow = build_ingredients(load_json(args.ia2v), args.profile)
    validate_ingredients(workflow, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Workflow IA2V + Ingredients ({args.profile}) criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
