#!/usr/bin/env python3
"""Converte o workflow pessoal LTX 2.3 (checkpoint fp8) para a variante GGUF 16 GB.

Estrategia de memoria para GPUs de 16 GB (alvo: RTX 5060 Ti):

- DiT: `UnetLoaderGGUF` com o checkpoint distilled 1.1 quantizado em GGUF
  (models/diffusion_models), no lugar do checkpoint fp8 de 27 GiB;
- Text encoder: `DualCLIPLoaderGGUF` com Gemma 3 12B QAT em GGUF + arquivo
  standalone de embeddings connectors (o node nativo `LTXAVTextEncoderLoader`
  le o checkpoint inteiro para extrair o connector, o que nao cabe aqui);
- VAEs: `VAELoaderKJ` (KJNodes) com video VAE e audio VAE standalone do
  repositorio unsloth/LTX-2.3-GGUF, no lugar dos VAEs internos do checkpoint;
- LoRA distilled dinamico (2,6 GiB) removido do grafo: o DiT distilled GGUF ja
  embute a destilacao (agenda de 8 passos, CFG 1);
- LoRA pessoal `glauberavatar.safetensors` e o LoRA de prompt-enhancement do
  Gemma permanecem intactos, aplicados em runtime sobre os pesos GGUF;
- sigmas, samplers, two-stage (latent upscale x2) e o restante do grafo nao
  mudam: mesmas seeds e mesmos sigmas do path de 48 GB.
"""
from __future__ import annotations

from copy import deepcopy

from workflow_utils import link_map, next_node_id, one


GGUF_UNET_FILE = "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
GEMMA_GGUF_FILE = "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
CONNECTORS_FILE = "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
VIDEO_VAE_FILE = "ltx-2.3-22b-distilled_video_vae.safetensors"
AUDIO_VAE_FILE = "ltx-2.3-22b-distilled_audio_vae.safetensors"
UPSCALER_FILE = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
PERSONAL_LORA_FILE = "glauberavatar.safetensors"
GEMMA_LORA_FILE = "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
DISTILLED_LORA_FILE = (
    "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
)

GGUF_SCHEMA_MARKER = "ltx23_16gb_gguf_schema"
GGUF_SCHEMA_VERSION = 1

CHECKPOINT_NODE_ID = 317
TEXT_ENCODER_NODE_ID = 318
AUDIO_VAE_NODE_ID = 335
DISTILLED_LORA_NODE_ID = 293
TOP_NODE_ID = 340

REQUIRED_FILES = (
    GGUF_UNET_FILE,
    GEMMA_GGUF_FILE,
    CONNECTORS_FILE,
    VIDEO_VAE_FILE,
    AUDIO_VAE_FILE,
    UPSCALER_FILE,
    PERSONAL_LORA_FILE,
    GEMMA_LORA_FILE,
)

# Widgets expostos no subgrafo que ficam sem consumidores apos a conversao.
# Localizados por nome porque o A2V remove o slot 0 e desloca os indices.
ORPHANED_SUBGRAPH_NAMES = ("ckpt_name", "lora_name", "text_encoder")

# Ligacoes de widget (COMBO) que saem do subgrafo junto com os loaders antigos.
ORPHANED_BINDING_LINKS = {735, 736, 737, 738, 739}


def _widget_input(name: str) -> dict:
    """Widget input sem ligacao (o valor vem de widgets_values do node)."""
    return {
        "localized_name": name,
        "name": name,
        "type": "COMBO",
        "widget": {"name": name},
        "link": None,
    }


def convert_to_gguf(workflow: dict) -> dict:
    """Aplica a conversao GGUF 16 GB ao workflow pessoal (IA2V ou A2V)."""
    if workflow.get("version") != 0.4:
        raise ValueError("o workflow pessoal precisa usar o schema 0.4")
    workflow = deepcopy(workflow)
    subgraph = one(workflow["definitions"]["subgraphs"])
    nodes = subgraph["nodes"]
    links = link_map(subgraph)

    checkpoint = one(
        nodes, node_id=CHECKPOINT_NODE_ID, node_type="CheckpointLoaderSimple"
    )
    text_encoder = one(
        nodes, node_id=TEXT_ENCODER_NODE_ID, node_type="LTXAVTextEncoderLoader"
    )
    audio_loader = one(nodes, node_id=AUDIO_VAE_NODE_ID, node_type="LTXVAudioVAELoader")
    distilled = one(
        nodes,
        node_id=DISTILLED_LORA_NODE_ID,
        node_type="LoraLoaderModelOnly",
    )
    top = one(workflow["nodes"], node_id=TOP_NODE_ID)

    # --- 1. DiT: CheckpointLoaderSimple -> UnetLoaderGGUF -------------------
    # O modelo GGUF assume os links de saida do LoRA distilled (removido do
    # grafo): guider do estagio 1, loader do prompt enhancement e o loader do
    # LoRA pessoal. O link MODEL do checkpoint para o node 293 e apagado.
    distilled_model_links = list(
        one([o for o in distilled["outputs"] if o.get("name") == "MODEL"])["links"]
    )
    checkpoint_model_links = list(
        one([o for o in checkpoint["outputs"] if o.get("name") == "MODEL"])["links"]
    )
    for link_id in distilled_model_links:
        links[link_id]["origin_id"] = CHECKPOINT_NODE_ID
        links[link_id]["origin_slot"] = 0
    subgraph["links"] = [
        link for link in subgraph["links"] if link["id"] not in checkpoint_model_links
    ]
    nodes.remove(distilled)

    checkpoint["type"] = "UnetLoaderGGUF"
    checkpoint["title"] = "LTX 2.3 distilled 1.1 DiT (GGUF)"
    checkpoint["inputs"] = [_widget_input("unet_name")]
    checkpoint["outputs"] = [
        {
            "localized_name": "MODEL",
            "name": "MODEL",
            "type": "MODEL",
            "links": distilled_model_links,
        }
    ]
    checkpoint["widgets_values"] = [GGUF_UNET_FILE]
    checkpoint["widgets_values_named"] = {"unet_name": GGUF_UNET_FILE}
    checkpoint["properties"] = {
        "Node name for S&R": "UnetLoaderGGUF",
        "models": [
            {
                "name": GGUF_UNET_FILE,
                "url": (
                    "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/"
                    f"distilled-1.1/{GGUF_UNET_FILE}"
                ),
                "directory": "diffusion_models",
            }
        ],
    }

    # --- 2. Text encoder: Gemma QAT GGUF + connectors standalone ------------
    text_encoder["type"] = "DualCLIPLoaderGGUF"
    text_encoder["title"] = "Gemma 3 12B QAT (GGUF) + connectors"
    text_encoder["inputs"] = [
        _widget_input("clip_name1"),
        _widget_input("clip_name2"),
        _widget_input("type"),
    ]
    text_encoder["widgets_values"] = [GEMMA_GGUF_FILE, CONNECTORS_FILE, "ltxv"]
    text_encoder["widgets_values_named"] = {
        "clip_name1": GEMMA_GGUF_FILE,
        "clip_name2": CONNECTORS_FILE,
        "type": "ltxv",
    }
    text_encoder["properties"] = {
        "Node name for S&R": "DualCLIPLoaderGGUF",
        "models": [
            {
                "name": GEMMA_GGUF_FILE,
                "url": (
                    "https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF/"
                    f"resolve/main/{GEMMA_GGUF_FILE}"
                ),
                "directory": "text_encoders",
            },
            {
                "name": CONNECTORS_FILE,
                "url": (
                    "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/"
                    f"text_encoders/{CONNECTORS_FILE}"
                ),
                "directory": "text_encoders",
            },
        ],
    }

    # --- 3. Audio VAE standalone via VAELoaderKJ -----------------------------
    audio_loader["type"] = "VAELoaderKJ"
    audio_loader["title"] = "LTX 2.3 Audio VAE (standalone)"
    audio_loader["inputs"] = [
        _widget_input("vae_name"),
        _widget_input("device"),
        _widget_input("weight_dtype"),
    ]
    audio_loader["widgets_values"] = [AUDIO_VAE_FILE, "main_device", "bf16"]
    audio_loader["widgets_values_named"] = {
        "vae_name": AUDIO_VAE_FILE,
        "device": "main_device",
        "weight_dtype": "bf16",
    }
    audio_loader["properties"] = {
        "Node name for S&R": "VAELoaderKJ",
        "models": [
            {
                "name": AUDIO_VAE_FILE,
                "url": (
                    "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/"
                    f"vae/{AUDIO_VAE_FILE}"
                ),
                "directory": "vae",
            }
        ],
    }

    # --- 4. Video VAE standalone para os consumers do VAE do checkpoint -----
    vae_consumer_link_ids = [
        link["id"]
        for link in subgraph["links"]
        if link["origin_id"] == CHECKPOINT_NODE_ID and link["origin_slot"] == 2
    ]
    video_vae_node_id = next_node_id(nodes)
    for link_id in vae_consumer_link_ids:
        links[link_id]["origin_id"] = video_vae_node_id
        links[link_id]["origin_slot"] = 0
    nodes.append(
        {
            "id": video_vae_node_id,
            "type": "VAELoaderKJ",
            "title": "LTX 2.3 Video VAE (standalone)",
            "pos": list(distilled["pos"]),
            "size": [430, 140],
            "flags": {},
            "order": distilled.get("order", 0),
            "mode": 0,
            "inputs": [
                _widget_input("vae_name"),
                _widget_input("device"),
                _widget_input("weight_dtype"),
            ],
            "outputs": [
                {
                    "localized_name": "VAE",
                    "name": "VAE",
                    "type": "VAE",
                    "links": list(vae_consumer_link_ids),
                }
            ],
            "properties": {
                "Node name for S&R": "VAELoaderKJ",
                "models": [
                    {
                        "name": VIDEO_VAE_FILE,
                        "url": (
                            "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/"
                            f"vae/{VIDEO_VAE_FILE}"
                        ),
                        "directory": "vae",
                    }
                ],
            },
            "widgets_values": [VIDEO_VAE_FILE, "main_device", "bf16"],
            "widgets_values_named": {
                "vae_name": VIDEO_VAE_FILE,
                "device": "main_device",
                "weight_dtype": "bf16",
            },
        }
    )
    # o loader antigo passa a expor apenas a saida MODEL
    checkpoint["outputs"] = checkpoint["outputs"][:1]

    # --- 6. Hint de modelo do loader do LoRA pessoal -------------------------
    personal_loader = one(
        [
            node
            for node in nodes
            if node.get("type") == "LoraLoaderModelOnly"
            and PERSONAL_LORA_FILE in (node.get("widgets_values") or [])
        ]
    )
    personal_loader["properties"]["models"] = [
        {"name": PERSONAL_LORA_FILE, "directory": "loras"}
    ]

    # --- 7. Ligacoes de widget orfas -----------------------------------------
    dropped_links = {735, 736, 737, 738, 739}
    subgraph["links"] = [
        link for link in subgraph["links"] if link["id"] not in dropped_links
    ]
    for sub_input in subgraph["inputs"]:
        if sub_input.get("name") in ORPHANED_SUBGRAPH_NAMES:
            sub_input["linkIds"] = []
    named = top.setdefault("widgets_values_named", {})
    named["ckpt_name"] = GGUF_UNET_FILE
    # O slot 11 (distilled LoRA) nao tem mais consumers; o valor exposto fica
    # com um nome de arquivo real para nao quebrar validacao de combo.
    named["lora_name"] = PERSONAL_LORA_FILE
    named["text_encoder"] = GEMMA_GGUF_FILE
    if isinstance(top.get("widgets_values"), list):
        # widgets posicionais do espelho de widgets nomeados do subgrafo
        for index, key in enumerate(named):
            if index < len(top["widgets_values"]) and isinstance(named[key], str):
                top["widgets_values"][index] = named[key]

    workflow.setdefault("extra", {})[GGUF_SCHEMA_MARKER] = GGUF_SCHEMA_VERSION
    return workflow


def validate_gguf(workflow: dict) -> None:
    """Verifica integridade estrutural e semantica do workflow convertido."""
    failures: list[str] = []
    if workflow.get("extra", {}).get(GGUF_SCHEMA_MARKER) != GGUF_SCHEMA_VERSION:
        failures.append("marcador de schema GGUF ausente")
    subgraph = one(workflow["definitions"]["subgraphs"])
    nodes = subgraph["nodes"]
    links = link_map(subgraph)
    if len(links) != len(subgraph["links"]):
        failures.append("IDs de links internos duplicados")

    def all_strings(value: object) -> set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, list):
            found: set[str] = set()
            for item in value:
                found.update(all_strings(item))
            return found
        if isinstance(value, dict):
            found: set[str] = set()
            for item in value.values():
                found.update(all_strings(item))
            return found
        return set()

    # Varre apenas widgets e hints de modelos dos nodes; MarkdownNotes sao
    # documentacao e podem citar arquivos do path 48 GB.
    strings: set[str] = set()
    for node in [*workflow.get("nodes", []), *nodes]:
        if node.get("type") == "MarkdownNote":
            continue
        strings.update(all_strings(node.get("widgets_values")))
        strings.update(all_strings(node.get("widgets_values_named")))
        models = node.get("properties", {}).get("models")
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    strings.add(entry["name"])
    for filename in REQUIRED_FILES:
        if filename not in strings:
            failures.append(f"arquivo de modelo ausente no grafo: {filename}")
    if DISTILLED_LORA_FILE in strings:
        failures.append("LoRA distilled dinamico deveria ter sido removido")
    if "ltx-2.3-22b-dev-fp8.safetensors" in strings:
        failures.append("checkpoint fp8 ainda referenciado no grafo")

    node_types = [node.get("type") for node in nodes]
    forbidden_types = {"CheckpointLoaderSimple", "LTXAVTextEncoderLoader", "LTXVAudioVAELoader"}
    if forbidden_types & set(node_types):
        failures.append(
            "loaders do path 48 GB ainda presentes: "
            + ", ".join(sorted(forbidden_types & set(node_types)))
        )
    all_node_types = [node.get("type") for node in workflow.get("nodes", [])] + node_types
    expected_counts = {
        "UnetLoaderGGUF": 1,
        "DualCLIPLoaderGGUF": 1,
        "VAELoaderKJ": 2,
        "LastFrameFromBatch": 1,
        "CreateVideo": 1,
        "SaveVideo": 1,
        "SaveImage": 1,
    }
    for node_type, expected in expected_counts.items():
        found = sum(1 for item in all_node_types if item == node_type)
        if found != expected:
            failures.append(f"esperado {expected}x {node_type}; encontrados {found}")

    unet_loader = one([n for n in nodes if n.get("type") == "UnetLoaderGGUF"])
    if unet_loader.get("widgets_values") != [GGUF_UNET_FILE]:
        failures.append("UnetLoaderGGUF com widgets incorretos")
    clip_loader = one([n for n in nodes if n.get("type") == "DualCLIPLoaderGGUF"])
    if sorted(str(item) for item in clip_loader.get("widgets_values", [])) != sorted(
        [GEMMA_GGUF_FILE, CONNECTORS_FILE, "ltxv"]
    ):
        failures.append("DualCLIPLoaderGGUF com widgets incorretos")
    vae_loaders = [n for n in nodes if n.get("type") == "VAELoaderKJ"]
    vae_files = sorted((n.get("widgets_values") or [None])[0] for n in vae_loaders)
    if vae_files != sorted([AUDIO_VAE_FILE, VIDEO_VAE_FILE]):
        failures.append(f"VAELoaderKJ com arquivos incorretos: {vae_files}")

    personal_loaders = [
        node
        for node in nodes
        if node.get("type") == "LoraLoaderModelOnly"
        and PERSONAL_LORA_FILE in (node.get("widgets_values") or [])
    ]
    if len(personal_loaders) != 1:
        failures.append("LoRA pessoal ausente ou duplicado")
    else:
        model_input = one(
            [i for i in personal_loaders[0]["inputs"] if i.get("name") == "model"]
        )
        link_id = model_input.get("link")
        if link_id not in links or links[link_id]["origin_id"] != CHECKPOINT_NODE_ID:
            failures.append("LoRA pessoal nao esta ligado ao DiT GGUF")

    # Integridade estrutural de links dentro do subgrafo, incluindo os
    # pseudo-nodes -10 (entradas) e -20 (saidas).
    node_by_id = {node["id"]: node for node in nodes}
    input_node_id = subgraph["inputNode"]["id"]
    output_node_id = subgraph["outputNode"]["id"]
    for link in subgraph["links"]:
        if link["origin_id"] == input_node_id:
            slots = subgraph["inputs"]
            if link["origin_slot"] >= len(slots):
                failures.append(f"link {link['id']} com slot de entrada invalido")
            elif link["id"] not in (slots[link["origin_slot"]].get("linkIds") or []):
                failures.append(
                    f"link {link['id']} nao listado no slot de entrada {link['origin_slot']}"
                )
        elif link["origin_id"] not in node_by_id:
            failures.append(f"link {link['id']} aponta para node inexistente")
            continue
        else:
            origin = node_by_id[link["origin_id"]]
            outputs = origin.get("outputs", [])
            if link["origin_slot"] >= len(outputs):
                failures.append(f"link {link['id']} com origin_slot invalido")
            elif link["id"] not in (outputs[link["origin_slot"]].get("links") or []):
                failures.append(
                    f"link {link['id']} nao listado na saida "
                    f"{link['origin_id']}:{link['origin_slot']}"
                )
        if link["target_id"] == output_node_id:
            outs = subgraph["outputs"]
            if link["target_slot"] >= len(outs):
                failures.append(f"link {link['id']} com target_slot invalido")
            elif link["id"] not in (outs[link["target_slot"]].get("linkIds") or []):
                failures.append(f"link {link['id']} nao listado na saida do subgrafo")
            continue
        if link["target_id"] not in node_by_id:
            failures.append(f"link {link['id']} aponta para node inexistente")
            continue
        target = node_by_id[link["target_id"]]
        inputs = target.get("inputs", [])
        if link["target_slot"] >= len(inputs):
            failures.append(f"link {link['id']} com target_slot invalido")
        elif inputs[link["target_slot"]].get("link") != link["id"]:
            failures.append(
                f"entrada {link['target_id']}:{link['target_slot']} inconsistente"
            )
    orphan_slots = [
        index
        for index, sub_input in enumerate(subgraph["inputs"])
        if sub_input.get("name") in ORPHANED_SUBGRAPH_NAMES
    ]
    for slot in orphan_slots:
        if subgraph["inputs"][slot].get("linkIds"):
            failures.append(f"slot de subgrafo {slot} deveria estar sem ligacoes")

    if failures:
        raise ValueError("; ".join(failures))
