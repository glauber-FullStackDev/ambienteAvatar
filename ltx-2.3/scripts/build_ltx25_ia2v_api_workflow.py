#!/usr/bin/env python3
"""Build the flat ComfyUI API workflow for LTX 2.5 IA2V (official A2V two-stage
pattern) with the personal identity LoRA.

Architecture (all nodes are flat API nodes; groups are conveyed via _meta titles):

  [INPUT]         LoadImage / LoadAudio / prompt prims / seed / strengths / decode tile
  [PREPROCESS]    ResizeImageMaskNode -> ResizeImagesByLongerEdge -> LTXVPreprocess
  [AUDIO]         TrimAudioDuration -> LTXVAudioVAEEncode -> SolidMask(0) freeze
  [MODELS]        UNETLoader(2.5 distilled) -> personal LoRA
  [PROMPT]        CLIPTextEncode +/- with optional TextGenerateLTX2Prompt switch
  [ANCHOR]        LTXVImgToVideoInplace pins frame 0 (first_frame_strength)
  [GENERATION 1]  base sampler (8-step distilled sigmas, CFG 1)
  [GENERATION 2]  LatentUpsampler x2 -> re-pin -> 3-step refine sigmas
  [OUTPUT]        VAEDecodeTiled + AudioVAEDecode -> CreateVideo -> SaveVideo/SaveImage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

TEMPLATE_MARKER = "ltx25_ia2v"
TEMPLATE_SCHEMA = 1

UNET_NAME = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
UNET_NAME_BF16 = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER_NAME = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
ENHANCER_NAME = "gemma4_e2b_it_int8_convrot.safetensors"
VIDEO_VAE_NAME = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE_NAME = "ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER_NAME = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
PERSONAL_LORA_NAME = "glauberavatar.safetensors"

BASE_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"

DEFAULT_PROMPT = (
    "The person remains in the exact composition and framing of the initial frame. "
    "He speaks expressively and clearly, with precise lip sync following the supplied audio. "
    "He looks directly at the camera, with natural blinking, breathing and small head movements. "
    "The camera remains completely stationary and the background stays unchanged."
)

REQUIRED_MODEL_NAMES = {
    UNET_NAME,
    TEXT_ENCODER_NAME,
    ENHANCER_NAME,
    VIDEO_VAE_NAME,
    AUDIO_VAE_NAME,
    UPSCALER_NAME,
    PERSONAL_LORA_NAME,
}

OUTPUT_PREFIX = "LTX_2.5_ia2v"


def _node(node_id: str, class_type: str, title: str, inputs: dict) -> dict:
    return {
        "class_type": class_type,
        "inputs": inputs,
        "_meta": {"title": title},
    }


def build_template(unet_name: str = UNET_NAME) -> dict:
    prompt: dict[str, dict] = {}

    # --- [INPUT] Entrada ---------------------------------------------------
    prompt["100"] = _node("100", "LoadImage", "[INPUT] First frame / identidade", {"image": "example.png"})
    prompt["101"] = _node("101", "LoadAudio", "[INPUT] Audio dirigente", {"audio": "audio.wav"})
    prompt["102"] = _node("102", "PrimitiveStringMultiline", "[INPUT] Prompt de movimento", {"value": DEFAULT_PROMPT})
    prompt["103"] = _node("103", "PrimitiveStringMultiline", "[INPUT] Negative prompt", {"value": ""})
    prompt["104"] = _node("104", "PrimitiveInt", "[INPUT] Width (saida)", {"value": 704})
    prompt["105"] = _node("105", "PrimitiveInt", "[INPUT] Height (saida)", {"value": 1280})
    prompt["106"] = _node("106", "PrimitiveFloat", "[INPUT] Duracao (s)", {"value": 5.0})
    prompt["107"] = _node("107", "PrimitiveFloat", "[INPUT] FPS", {"value": 24.0})
    prompt["108"] = _node("108", "PrimitiveFloat", "[INPUT] Audio start (s)", {"value": 0.0})
    prompt["109"] = _node("109", "PrimitiveInt", "[INPUT] Seed", {"value": 0})
    prompt["110"] = _node("110", "PrimitiveFloat", "[INPUT] first_frame_strength", {"value": 1.0})
    prompt["144"] = _node("144", "PrimitiveBoolean", "[INPUT] Prompt enhancer (off)", {"value": False})
    prompt["196"] = _node("196", "PrimitiveInt", "[INPUT] decode_tile_size", {"value": 512})

    # --- [MATH] derivados --------------------------------------------------
    prompt["190"] = _node("190", "ComfyMathExpression", "[MATH] width/2 (base)", {"expression": "a/2", "values.a": ["104", 0]})
    prompt["191"] = _node("191", "ComfyMathExpression", "[MATH] height/2 (base)", {"expression": "a/2", "values.a": ["105", 0]})
    prompt["192"] = _node(
        "192",
        "ComfyMathExpression",
        "[MATH] frames = 1+floor(dur*fps/8)*8",
        {"expression": "1 + floor(a*b/8)*8", "values.a": ["106", 0], "values.b": ["107", 0]},
    )
    prompt["193"] = _node("193", "LTXFloatToInt", "[MATH] width -> INT", {"a": ["190", 0]})
    prompt["194"] = _node("194", "LTXFloatToInt", "[MATH] height -> INT", {"a": ["191", 0]})
    prompt["195"] = _node("195", "LTXFloatToInt", "[MATH] frames -> INT", {"a": ["192", 0]})

    # --- [PREPROCESS] Imagem ------------------------------------------------
    prompt["112"] = _node(
        "112",
        "ResizeImageMaskNode",
        "[PREPROCESS] Resize para WxH",
        {
            "input": ["100", 0],
            "resize_type": "scale dimensions",
            "resize_type.width": ["104", 0],
            "resize_type.height": ["105", 0],
            "resize_type.crop": "center",
            "scale_method": "lanczos",
        },
    )
    prompt["113"] = _node("113", "ResizeImagesByLongerEdge", "[PREPROCESS] Cap 1536", {"images": ["112", 0], "longer_edge": 1536})
    prompt["114"] = _node("114", "LTXVPreprocess", "[PREPROCESS] Compressao 18", {"image": ["113", 0], "img_compression": 18})

    # --- [AUDIO / LIPSYNC] ---------------------------------------------------
    prompt["121"] = _node("121", "VAELoader", "[AUDIO] Audio VAE 2.5", {"vae_name": AUDIO_VAE_NAME})
    prompt["120"] = _node("120", "TrimAudioDuration", "[AUDIO] Recorte", {"audio": ["101", 0], "start_index": ["108", 0], "duration": ["106", 0]})
    prompt["122"] = _node("122", "LTXVAudioVAEEncode", "[AUDIO] Encode (lipsync nativo)", {"audio": ["120", 0], "audio_vae": ["121", 0]})
    prompt["123"] = _node("123", "SolidMask", "[AUDIO] Mascara zero", {"value": 0, "width": 1024, "height": 1024})
    prompt["124"] = _node("124", "SetLatentNoiseMask", "[AUDIO] Audio congelado", {"samples": ["122", 0], "mask": ["123", 0]})

    # --- [MODELS] -------------------------------------------------------------
    prompt["130"] = _node("130", "UNETLoader", "[MODELS] LTX 2.5 22B destilado", {"unet_name": unet_name, "weight_dtype": "default"})
    prompt["132"] = _node(
        "132",
        "LoraLoaderModelOnly",
        "[MODELS] LoRA pessoal (toggle)",
        {"model": ["130", 0], "lora_name": PERSONAL_LORA_NAME, "strength_model": 0.7},
    )
    prompt["133"] = _node("133", "VAELoader", "[MODELS] Video VAE 2.5", {"vae_name": VIDEO_VAE_NAME})
    prompt["134"] = _node("134", "CLIPLoader", "[MODELS] Gemma4 12B (texto)", {"clip_name": TEXT_ENCODER_NAME, "type": "ltxv", "device": "default"})
    prompt["135"] = _node("135", "CLIPLoader", "[MODELS] Gemma4 E2B (enhancer)", {"clip_name": ENHANCER_NAME, "type": "ltxv", "device": "default"})
    prompt["136"] = _node("136", "LatentUpscaleModelLoader", "[MODELS] Upscaler x2", {"model_name": UPSCALER_NAME})

    # --- [PROMPT] ---------------------------------------------------------------
    prompt["140"] = _node("140", "CLIPTextEncode", "[PROMPT] Negative", {"clip": ["134", 0], "text": ["103", 0]})
    prompt["141"] = _node(
        "141",
        "TextGenerateLTX2Prompt",
        "[PROMPT] Enhancer (opcional)",
        {
            "clip": ["135", 0],
            "prompt": ["102", 0],
            "max_length": 600,
            "sampling_mode": "on",
            "sampling_mode.temperature": 0.01,
            "sampling_mode.top_k": 0,
            "sampling_mode.top_p": 1.0,
            "sampling_mode.min_p": 0.0,
            "sampling_mode.repetition_penalty": 1.15,
            "sampling_mode.seed": 0,
            "sampling_mode.presence_penalty": 0.0,
            "thinking": False,
            "use_default_template": True,
        },
    )
    prompt["142"] = _node("142", "ComfySwitchNode", "[PROMPT] Switch enhancer", {"switch": ["144", 0], "on_false": ["102", 0], "on_true": ["141", 0]})
    prompt["143"] = _node("143", "CLIPTextEncode", "[PROMPT] Positive", {"clip": ["134", 0], "text": ["142", 0]})

    # --- [CONDITIONING + ANCHOR] -------------------------------------------------
    prompt["150"] = _node("150", "LTXVConditioning", "[COND] frame rate", {"positive": ["143", 0], "negative": ["140", 0], "frame_rate": ["107", 0]})
    prompt["152"] = _node(
        "152",
        "EmptyLTXVLatentVideo",
        "[ANCHOR] Latente base (W/2 x H/2)",
        {"width": ["193", 0], "height": ["194", 0], "length": ["195", 0], "batch_size": 1},
    )
    prompt["153"] = _node(
        "153",
        "LTXVImgToVideoInplace",
        "[ANCHOR] First frame pin",
        {"vae": ["133", 0], "image": ["114", 0], "latent": ["152", 0], "strength": ["110", 0], "bypass": False},
    )

    # --- [GENERATION 1] base ------------------------------------------------------
    prompt["160"] = _node("160", "ManualSigmas", "[GEN1] Sigmas base (8 passos)", {"sigmas": BASE_SIGMAS})
    prompt["161"] = _node("161", "KSamplerSelect", "[GEN1] Sampler", {"sampler_name": "euler_ancestral"})
    prompt["162"] = _node("162", "RandomNoise", "[GEN1] Noise (seed)", {"noise_seed": ["109", 0], "control_after_generate": "fixed"})
    prompt["163"] = _node("163", "CFGGuider", "[GEN1] CFG", {"model": ["132", 0], "positive": ["150", 0], "negative": ["150", 1], "cfg": 1.0})
    prompt["164"] = _node("164", "LTXVConcatAVLatent", "[GEN1] Concat video+audio", {"video_latent": ["153", 0], "audio_latent": ["124", 0]})
    prompt["165"] = _node(
        "165",
        "SamplerCustomAdvanced",
        "[GEN1] Sampler base",
        {"noise": ["162", 0], "guider": ["163", 0], "sampler": ["161", 0], "sigmas": ["160", 0], "latent_image": ["164", 0]},
    )
    prompt["166"] = _node("166", "LTXVSeparateAVLatent", "[GEN1] Separa AV", {"av_latent": ["165", 0]})

    # --- [GENERATION 2] refine -----------------------------------------------------
    prompt["170"] = _node("170", "LTXVLatentUpsampler", "[GEN2] Upsample x2", {"samples": ["166", 0], "upscale_model": ["136", 0], "vae": ["133", 0]})
    prompt["171"] = _node(
        "171",
        "LTXVImgToVideoInplace",
        "[GEN2] First frame pin (refine)",
        {"vae": ["133", 0], "image": ["114", 0], "latent": ["170", 0], "strength": 1.0, "bypass": False},
    )
    prompt["172"] = _node("172", "LTXVConcatAVLatent", "[GEN2] Concat video+audio", {"video_latent": ["171", 0], "audio_latent": ["124", 0]})
    prompt["173"] = _node("173", "ManualSigmas", "[GEN2] Sigmas refine (3 passos)", {"sigmas": REFINE_SIGMAS})
    prompt["174"] = _node("174", "KSamplerSelect", "[GEN2] Sampler", {"sampler_name": "euler_ancestral"})
    prompt["175"] = _node("175", "RandomNoise", "[GEN2] Noise fixo", {"noise_seed": 42, "control_after_generate": "fixed"})
    prompt["176"] = _node("176", "CFGGuider", "[GEN2] CFG", {"model": ["132", 0], "positive": ["150", 0], "negative": ["150", 1], "cfg": 1.0})
    prompt["177"] = _node(
        "177",
        "SamplerCustomAdvanced",
        "[GEN2] Sampler refine",
        {"noise": ["175", 0], "guider": ["176", 0], "sampler": ["174", 0], "sigmas": ["173", 0], "latent_image": ["172", 0]},
    )
    prompt["178"] = _node("178", "LTXVSeparateAVLatent", "[GEN2] Separa AV", {"av_latent": ["177", 0]})

    # --- [OUTPUT] -----------------------------------------------------------------
    prompt["180"] = _node(
        "180",
        "VAEDecodeTiled",
        "[OUTPUT] Decode video",
        {"samples": ["178", 0], "vae": ["133", 0], "tile_size": ["196", 0], "overlap": 64, "temporal_size": 64, "temporal_overlap": 8},
    )
    prompt["181"] = _node("181", "LTXVAudioVAEDecode", "[OUTPUT] Decode audio", {"samples": ["178", 1], "audio_vae": ["121", 0]})
    prompt["182"] = _node("182", "CreateVideo", "[OUTPUT] Mux video+audio", {"images": ["180", 0], "fps": ["107", 0], "audio": ["181", 0], "bit_depth": "auto", "color_space": "sRGB"})
    prompt["183"] = _node(
        "183",
        "SaveVideo",
        "[OUTPUT] Save video",
        {"video": ["182", 0], "filename_prefix": f"video/jobs/__JOB_ID__/{OUTPUT_PREFIX}", "format": "auto"},
    )
    prompt["184"] = _node("184", "LastFrameFromBatch", "[OUTPUT] Ultimo frame", {"images": ["180", 0]})
    prompt["185"] = _node(
        "185",
        "SaveImage",
        "[OUTPUT] Save last frame",
        {"images": ["184", 0], "filename_prefix": f"images/last_frame/jobs/__JOB_ID__/{OUTPUT_PREFIX}"},
    )

    return {
        "_meta": {"template": TEMPLATE_MARKER, "schema": TEMPLATE_SCHEMA},
        "prompt": prompt,
    }


def validate_template(template: dict, unet_name: str = UNET_NAME) -> None:
    prompt = template["prompt"]
    if template["_meta"]["template"] != TEMPLATE_MARKER:
        raise ValueError("marcador de template ausente")

    referenced_models = {
        value
        for node in prompt.values()
        for key, value in node["inputs"].items()
        if key in {"unet_name", "vae_name", "clip_name", "model_name", "lora_name"} and isinstance(value, str)
    }
    required_models = set(REQUIRED_MODEL_NAMES)
    required_models.discard(UNET_NAME)
    required_models.add(unet_name)
    missing = required_models - referenced_models
    if missing:
        raise ValueError(f"modelos obrigatorios ausentes no template: {sorted(missing)}")

    # Every link endpoint must exist with a matching output slot.
    for node_id, node in prompt.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2:
                source_id, slot = value
                if source_id not in prompt:
                    raise ValueError(f"{node_id}.{name} aponta para node inexistente {source_id}")
                # Slot count is validated implicitly by ComfyUI; check self-links.
                if source_id == node_id:
                    raise ValueError(f"{node_id}.{name} tem auto-link")

    # The anchor/sampler chain must be wired in this exact order.
    assert prompt["153"]["inputs"]["latent"] == ["152", 0]
    assert prompt["163"]["inputs"]["positive"] == ["150", 0]
    assert prompt["163"]["inputs"]["negative"] == ["150", 1]
    assert prompt["163"]["inputs"]["model"] == ["132", 0]
    assert prompt["164"]["inputs"]["video_latent"] == ["153", 0]
    assert prompt["170"]["inputs"]["samples"] == ["166", 0]
    assert prompt["176"]["inputs"]["positive"] == ["150", 0]
    assert prompt["176"]["inputs"]["model"] == ["132", 0]
    assert prompt["172"]["inputs"]["audio_latent"] == ["124", 0]
    assert prompt["180"]["inputs"]["tile_size"] == ["196", 0]
    for gone in ("111", "115", "116", "117", "118", "119", "131", "151", "154", "156", "167"):
        if gone in prompt:
            raise ValueError(f"node {gone} deveria ter sido removido")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o workflow API LTX-2.5 IA2V")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unet-name",
        default=UNET_NAME,
        choices=(UNET_NAME, UNET_NAME_BF16),
        help="checkpoint do DiT destilado usado no UNETLoader",
    )
    args = parser.parse_args()

    template = build_template(unet_name=args.unet_name)
    validate_template(template, unet_name=args.unet_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Workflow LTX-2.5 IA2V criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
