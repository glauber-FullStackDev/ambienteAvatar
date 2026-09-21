#!/usr/bin/env python3
"""Build the flat ComfyUI API workflow for LTX 2.5 IA2V with IC-LoRA persistent guide.

Architecture (all nodes are flat API nodes; groups are conveyed via _meta titles):

  [INPUT]         LoadImage / LoadAudio / prompt prims / seed / strengths
  [PREPROCESS]    ResizeImageMaskNode -> ResizeImagesByLongerEdge -> LTXVPreprocess
  [AUDIO]         TrimAudioDuration -> LTXVAudioVAEEncode -> SolidMask(0) freeze
  [MODELS]        UNETLoader(2.5 distilled) -> IC-LoRA ingredients -> personal LoRA
  [PROMPT]        CLIPTextEncode +/- with optional TextGenerateLTX2Prompt switch
  [ANCHOR]        LTXVImgToVideoInplace pins frame 0 (first_frame_strength)
  [GUIDE]         LTXAddVideoICLoRAGuide appends persistent in-context guide tokens
  [GENERATION 1]  base sampler (8-step distilled sigmas, CFG 1)
  [CROP]          LTXVCropGuides strips guide tokens between stages
  [GENERATION 2]  LatentUpsampler x2 -> re-sampler (3-step refine sigmas)
  [OUTPUT]        VAEDecodeTiled + AudioVAEDecode -> CreateVideo -> SaveVideo/SaveImage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

TEMPLATE_MARKER = "ltx25_iclora"
TEMPLATE_SCHEMA = 1

UNET_NAME = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
TEXT_ENCODER_NAME = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
ENHANCER_NAME = "gemma4_e2b_it_int8_convrot.safetensors"
VIDEO_VAE_NAME = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE_NAME = "ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER_NAME = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
ICLORA_NAME = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
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
    ICLORA_NAME,
    PERSONAL_LORA_NAME,
}

OUTPUT_PREFIX = "LTX_2.5_ia2v_iclora"


def _node(node_id: str, class_type: str, title: str, inputs: dict) -> dict:
    return {
        "class_type": class_type,
        "inputs": inputs,
        "_meta": {"title": title},
    }


def build_template() -> dict:
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
    prompt["111"] = _node("111", "PrimitiveFloat", "[INPUT] guiding_strength (0 desliga)", {"value": 0.5})
    prompt["115"] = _node("115", "LoadImage", "[INPUT] Referencia adicional (opcional)", {"image": "reference.png"})
    prompt["118"] = _node("118", "PrimitiveInt", "[INPUT] reference_frame_idx", {"value": -1})
    prompt["119"] = _node("119", "PrimitiveFloat", "[INPUT] reference_guiding_strength", {"value": 0.8})
    prompt["144"] = _node("144", "PrimitiveBoolean", "[INPUT] Prompt enhancer (off)", {"value": False})

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
    prompt["116"] = _node(
        "116",
        "ResizeImageMaskNode",
        "[PREPROCESS] Resize referencia WxH",
        {
            "input": ["115", 0],
            "resize_type": "scale dimensions",
            "resize_type.width": ["104", 0],
            "resize_type.height": ["105", 0],
            "resize_type.crop": "center",
            "scale_method": "lanczos",
        },
    )
    prompt["117"] = _node("117", "ResizeImagesByLongerEdge", "[PREPROCESS] Cap referencia 1536", {"images": ["116", 0], "longer_edge": 1536})

    # --- [AUDIO / LIPSYNC] ---------------------------------------------------
    prompt["121"] = _node("121", "VAELoader", "[AUDIO] Audio VAE 2.5", {"vae_name": AUDIO_VAE_NAME})
    prompt["120"] = _node("120", "TrimAudioDuration", "[AUDIO] Recorte", {"audio": ["101", 0], "start_index": ["108", 0], "duration": ["106", 0]})
    prompt["122"] = _node("122", "LTXVAudioVAEEncode", "[AUDIO] Encode (lipsync nativo)", {"audio": ["120", 0], "audio_vae": ["121", 0]})
    prompt["123"] = _node("123", "SolidMask", "[AUDIO] Mascara zero", {"value": 0, "width": 1024, "height": 1024})
    prompt["124"] = _node("124", "SetLatentNoiseMask", "[AUDIO] Audio congelado", {"samples": ["122", 0], "mask": ["123", 0]})

    # --- [MODELS] -------------------------------------------------------------
    prompt["130"] = _node("130", "UNETLoader", "[MODELS] LTX 2.5 22B destilado", {"unet_name": UNET_NAME, "weight_dtype": "default"})
    prompt["131"] = _node(
        "131",
        "LTXICLoRALoaderModelOnly",
        "[MODELS] IC-LoRA Ingredients (guia persistente)",
        {"model": ["130", 0], "lora_name": ICLORA_NAME, "strength_model": 0.9},
    )
    prompt["132"] = _node(
        "132",
        "LoraLoaderModelOnly",
        "[MODELS] LoRA pessoal (toggle)",
        {"model": ["131", 0], "lora_name": PERSONAL_LORA_NAME, "strength_model": 0.7},
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

    # --- [CONDITIONING + ANCHOR + GUIDE] ---------------------------------------
    prompt["150"] = _node("150", "LTXVConditioning", "[COND] frame rate", {"positive": ["143", 0], "negative": ["140", 0], "frame_rate": ["107", 0]})
    prompt["151"] = _node("151", "RepeatImageBatch", "[GUIDE] Repete first frame", {"image": ["113", 0], "amount": ["195", 0]})
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
    prompt["154"] = _node(
        "154",
        "LTXAddVideoICLoRAGuide",
        "[GUIDE] Guia persistente (video inteiro)",
        {
            "positive": ["150", 0],
            "negative": ["150", 1],
            "vae": ["133", 0],
            "latent": ["153", 0],
            "image": ["151", 0],
            "frame_idx": 0,
            "strength": ["111", 0],
            "latent_downscale_factor": 1.0,
            "crop": "disabled",
            "use_tiled_encode": False,
            "tile_size": 256,
            "tile_overlap": 64,
        },
    )
    prompt["156"] = _node(
        "156",
        "LTXAddVideoICLoRAGuide",
        "[GUIDE] Referencia adicional (keyframe)",
        {
            "positive": ["154", 0],
            "negative": ["154", 1],
            "vae": ["133", 0],
            "latent": ["154", 2],
            "image": ["117", 0],
            "frame_idx": ["118", 0],
            "strength": ["119", 0],
            "latent_downscale_factor": 1.0,
            "crop": "disabled",
            "use_tiled_encode": False,
            "tile_size": 256,
            "tile_overlap": 64,
        },
    )

    # --- [GENERATION 1] base ------------------------------------------------------
    prompt["160"] = _node("160", "ManualSigmas", "[GEN1] Sigmas base (8 passos)", {"sigmas": BASE_SIGMAS})
    prompt["161"] = _node("161", "KSamplerSelect", "[GEN1] Sampler", {"sampler_name": "euler_ancestral"})
    prompt["162"] = _node("162", "RandomNoise", "[GEN1] Noise (seed)", {"noise_seed": ["109", 0], "control_after_generate": "fixed"})
    prompt["163"] = _node("163", "CFGGuider", "[GEN1] CFG", {"model": ["132", 0], "positive": ["156", 0], "negative": ["156", 1], "cfg": 1.0})
    prompt["164"] = _node("164", "LTXVConcatAVLatent", "[GEN1] Concat video+audio", {"video_latent": ["156", 2], "audio_latent": ["124", 0]})
    prompt["165"] = _node(
        "165",
        "SamplerCustomAdvanced",
        "[GEN1] Sampler base",
        {"noise": ["162", 0], "guider": ["163", 0], "sampler": ["161", 0], "sigmas": ["160", 0], "latent_image": ["164", 0]},
    )
    prompt["166"] = _node("166", "LTXVSeparateAVLatent", "[GEN1] Separa AV", {"av_latent": ["165", 0]})
    prompt["167"] = _node(
        "167",
        "LTXVCropGuides",
        "[CROP] Remove tokens do guia",
        {"positive": ["156", 0], "negative": ["156", 1], "latent": ["166", 0]},
    )

    # --- [GENERATION 2] refine -----------------------------------------------------
    prompt["170"] = _node("170", "LTXVLatentUpsampler", "[GEN2] Upsample x2", {"samples": ["167", 2], "upscale_model": ["136", 0], "vae": ["133", 0]})
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
    prompt["176"] = _node("176", "CFGGuider", "[GEN2] CFG", {"model": ["132", 0], "positive": ["167", 0], "negative": ["167", 1], "cfg": 1.0})
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
        {"samples": ["178", 0], "vae": ["133", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8},
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


def validate_template(template: dict) -> None:
    prompt = template["prompt"]
    if template["_meta"]["template"] != TEMPLATE_MARKER:
        raise ValueError("marcador de template ausente")

    referenced_models = {
        value
        for node in prompt.values()
        for key, value in node["inputs"].items()
        if key in {"unet_name", "vae_name", "clip_name", "model_name", "lora_name"} and isinstance(value, str)
    }
    missing = REQUIRED_MODEL_NAMES - referenced_models
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

    # The anchor/guide/crop chain must be wired in this exact order.
    assert prompt["153"]["inputs"]["latent"] == ["152", 0]
    assert prompt["154"]["inputs"]["latent"] == ["153", 0]
    assert prompt["156"]["inputs"]["positive"] == ["154", 0]
    assert prompt["156"]["inputs"]["negative"] == ["154", 1]
    assert prompt["156"]["inputs"]["latent"] == ["154", 2]
    assert prompt["164"]["inputs"]["video_latent"] == ["156", 2]
    assert prompt["163"]["inputs"]["positive"] == ["156", 0]
    assert prompt["167"]["inputs"]["positive"] == ["156", 0]
    assert prompt["167"]["inputs"]["negative"] == ["156", 1]
    assert prompt["170"]["inputs"]["samples"] == ["167", 2]
    assert prompt["172"]["inputs"]["audio_latent"] == ["124", 0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o workflow API LTX-2.5 IA2V com IC-LoRA")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    template = build_template()
    validate_template(template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Workflow LTX-2.5 IA2V IC-LoRA criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
