#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import threading
import time

import gradio as gr
import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video, load_video


MODEL_REPOS = {
    "base": "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
    "distilled": "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
}

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "base").strip().lower()
GENERATION_LOCK = threading.Lock()


def selected_model_path() -> Path:
    if MODEL_VARIANT not in MODEL_REPOS:
        choices = ", ".join(MODEL_REPOS)
        raise RuntimeError(f"MODEL_VARIANT invalido: {MODEL_VARIANT!r}; use {choices}")
    return MODEL_ROOT / MODEL_VARIANT


def load_pipeline() -> ModularPipeline:
    if not torch.cuda.is_available():
        raise RuntimeError("Wan-Animate-2 requer uma GPU NVIDIA com CUDA")

    model_path = selected_model_path()
    if not (model_path / "modular_model_index.json").exists():
        raise RuntimeError(
            f"Modelo ausente em {model_path}. Execute download-models ou defina "
            "DOWNLOAD_MODELS_ON_START=1."
        )

    print(f"Carregando Wan-Animate-2 {MODEL_VARIANT} de {model_path}")
    pipe = ModularPipeline.from_pretrained(str(model_path), local_files_only=True)
    pipe.load_components(dtype=torch.bfloat16)

    # O transformer (~28 GB) e o KV cache de referencia (~21-35 GB) nao cabem
    # juntos com os demais componentes. Este e o arranjo recomendado pelo
    # Diffusers para uma unica GPU de 80 GB.
    pipe.transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        use_stream=True,
    )
    pipe.text_encoder.to("cuda")
    pipe.image_encoder.to("cuda")
    pipe.vae.to("cuda")
    pipe.transformer.compile_repeated_blocks(fullgraph=False)
    return pipe


PIPELINE = load_pipeline()


def generate(
    image,
    driving_video_path: str,
    prompt: str,
    width: int,
    height: int,
    fps: int,
    segment_frame_length: int,
    seed: int,
) -> tuple[str | None, str]:
    if image is None:
        return None, "Envie a imagem de referencia."
    if not driving_video_path:
        return None, "Envie o video-guia."
    if not prompt or not prompt.strip():
        return None, "Descreva objetivamente personagem, roupa e fundo no prompt."

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"wan-animate-2-{MODEL_VARIANT}-{timestamp}.mp4"
    driving_video, driving_fps = load_video(driving_video_path, return_fps=True)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    started_at = time.monotonic()
    try:
        with GENERATION_LOCK:
            videos = PIPELINE(
                image=image,
                driving_video=driving_video,
                driving_video_fps=driving_fps,
                prompt=prompt.strip(),
                width=int(width),
                height=int(height),
                fps=int(fps),
                segment_frame_length=int(segment_frame_length),
                generator=generator,
                output="videos",
            )
            export_to_video(videos[0], str(output_path), fps=int(fps))
        elapsed = time.monotonic() - started_at
        return str(output_path), f"Concluido em {elapsed / 60:.1f} min ({MODEL_VARIANT})."
    except Exception as exc:
        torch.cuda.empty_cache()
        return None, f"Falha na geracao: {exc}"


def create_ui() -> gr.Blocks:
    steps = 40 if MODEL_VARIANT == "base" else 10
    with gr.Blocks(title="Wan-Animate-2") as demo:
        gr.Markdown(
            "# Wan-Animate-2\n"
            f"Modelo **{MODEL_VARIANT}** ({steps} passos). A imagem define personagem, "
            "roupa e fundo; o video fornece movimento e expressoes."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Imagem de referencia final", type="pil")
                driving_video = gr.Video(label="Video-guia")
                prompt = gr.Textbox(
                    label="Descricao objetiva da imagem",
                    lines=5,
                    placeholder=(
                        "Descreva aparencia, roupa e fundo. Nao descreva a acao do "
                        "video-guia."
                    ),
                )
                with gr.Row():
                    width = gr.Number(label="Largura-alvo", value=640, precision=0)
                    height = gr.Number(label="Altura-alvo", value=800, precision=0)
                with gr.Row():
                    fps = gr.Number(label="FPS", value=24, precision=0)
                    segment = gr.Number(
                        label="Frames por segmento", value=81, precision=0
                    )
                    seed = gr.Number(label="Seed", value=123, precision=0)
                generate_button = gr.Button("Gerar video-base", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="Video-base gerado")
                status = gr.Textbox(label="Status", interactive=False)

        generate_button.click(
            fn=generate,
            inputs=[image, driving_video, prompt, width, height, fps, segment, seed],
            outputs=[output_video, status],
        )
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", "8188"))
    create_ui().queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
