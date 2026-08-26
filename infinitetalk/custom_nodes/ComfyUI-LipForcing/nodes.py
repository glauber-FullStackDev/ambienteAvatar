from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading

import folder_paths


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
RUNTIME_ROOT = Path(os.environ.get("LIPFORCING_ROOT", "/opt/LipForcing"))
RUNTIME_PYTHON = Path(
    os.environ.get("LIPFORCING_PYTHON", "/opt/lipforcing-venv/bin/python")
)
MODELS_ROOT = Path(
    os.environ.get("LIPFORCING_MODELS", "/opt/ComfyUI/models/lipforcing")
)
AUXILIARY_ROOT = Path(
    os.environ.get(
        "LIPFORCING_AUXILIARY",
        "/opt/ComfyUI/models/latentsync/auxiliary/models/buffalo_l",
    )
)
MODEL_PATHS = {
    "checkpoint": MODELS_ROOT / "lipforcing_14b.pth",
    "vae": MODELS_ROOT / "Wan2.1_VAE.pth",
    "wav2vec": MODELS_ROOT / "wav2vec2-base-960h",
    "mask": MODELS_ROOT / "mask.png",
    "taehv": MODELS_ROOT / "taew2_1.pth",
    "text_embeds": MODELS_ROOT / "text_emb_a_person_talking.pt",
}
# 48 GB datacenter/workstation cards can report about 44-48 GiB to CUDA.
# This threshold rejects 24/32 GB cards without falsely rejecting an A40.
MINIMUM_VRAM_BYTES = 44 * 1024**3
_RUN_LOCK = threading.Lock()

QUALITY_MODES = (
    "segmentwise_max_quality",
    "streaming_full_vae",
    "streaming_fast",
    "manual_decoder",
)
COMPOSITE_MODES = ("mouth_only", "full_face")


def _input_files(extensions: set[str]) -> list[str]:
    input_dir = Path(folder_paths.get_input_directory())
    return sorted(
        path.name
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def _resolved_input(filename: str, extensions: set[str]) -> Path:
    if not folder_paths.exists_annotated_filepath(filename):
        raise ValueError(f"Arquivo de entrada invalido: {filename}")
    path = Path(folder_paths.get_annotated_filepath(filename)).resolve()
    return _validate_resolved_input(path, extensions)


def _validate_resolved_input(path: Path, extensions: set[str]) -> Path:
    input_root = Path(folder_paths.get_input_directory()).resolve()
    if not path.is_file():
        raise ValueError(f"Arquivo de entrada inexistente: {path}")
    if not path.is_relative_to(input_root):
        raise ValueError("O arquivo precisa estar dentro da pasta input do ComfyUI")
    if path.suffix.lower() not in extensions:
        raise ValueError(f"Extensao nao suportada: {path.suffix}")
    return path


def _path_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_runtime() -> None:
    missing = []
    for path in (
        RUNTIME_PYTHON,
        RUNTIME_ROOT / "scripts/inference/inference_streaming.py",
        RUNTIME_ROOT / "scripts/inference/inference_segmentwise.py",
    ):
        if not path.exists():
            missing.append(path)
    for path in MODEL_PATHS.values():
        if not path.exists() or (path.is_file() and path.stat().st_size == 0):
            missing.append(path)
    for filename in (
        "det_10g.onnx",
        "2d106det.onnx",
        "1k3d68.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    ):
        path = AUXILIARY_ROOT / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(path)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            "Runtime ou modelos do Lip Forcing ausentes:\n"
            f"{formatted}\n"
            "Execute: /opt/infinitetalk-scripts/container-entrypoint.sh "
            "download-lipforcing-models"
        )


def _audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Duracao de audio invalida: {duration}")
    return duration


def _exact_latent_frames(duration: float, fps: float = 25.0, chunk_size: int = 3) -> int:
    # The Wan VAE produces 1 + (latent_frames - 1) * 4 video frames. Round
    # both conversions up so the generated stream always covers the complete
    # audio before the final frame-accurate trim.
    video_frames = max(1, math.ceil(duration * fps))
    latent_frames = 1 + math.ceil(max(0, video_frames - 1) / 4)
    return max(chunk_size, math.ceil(latent_frames / chunk_size) * chunk_size)


def _safe_prefix(value: str) -> str:
    value = value.strip().replace("\\", "/").split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "LipForcing14B_Final"


def _next_output(prefix: str) -> tuple[Path, str]:
    output_root = Path(folder_paths.get_output_directory())
    output_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{5}})\.mp4", re.IGNORECASE)
    counter = 0
    for path in output_root.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            counter = max(counter, int(match.group(1)))
    filename = f"{prefix}_{counter + 1:05d}.mp4"
    return output_root / filename, filename


def _mux_exact_audio(
    generated: Path,
    audio: Path,
    output: Path,
    duration: float,
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(generated),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-t",
            f"{duration:.6f}",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def _quality_pipeline(quality_mode: str, decoder: str) -> tuple[str, str | None]:
    if quality_mode == "segmentwise_max_quality":
        return "inference_segmentwise.py", None
    if quality_mode == "streaming_full_vae":
        return "inference_streaming.py", "wan_vae"
    if quality_mode == "streaming_fast":
        return "inference_streaming.py", "streaming_taehv"
    if quality_mode == "manual_decoder":
        return "inference_streaming.py", decoder
    raise ValueError(f"Modo de qualidade invalido: {quality_mode}")


def _check_vram() -> None:
    if os.environ.get("LIPFORCING_ALLOW_LOW_VRAM", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Lip Forcing 14B requer uma GPU NVIDIA CUDA")
    available = torch.cuda.get_device_properties(0).total_memory
    if available < MINIMUM_VRAM_BYTES:
        gib = available / 1024**3
        raise RuntimeError(
            f"Lip Forcing 14B requer GPU de 48 GB; GPU atual: {gib:.1f} GB. "
            "Use LIPFORCING_ALLOW_LOW_VRAM=1 somente para testes nao suportados."
        )


def _unload_comfy_models() -> None:
    import comfy.model_management as model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache()


def _run_streaming(command: list[str], cwd: Path) -> None:
    printable = " ".join(command[:2] + ["<modelos-e-entradas-omitidos>"])
    print(f"[LipForcing14B] Iniciando: {printable}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[LipForcing14B] {line}", end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Lip Forcing terminou com codigo {return_code}. "
            "Consulte /var/log/portal/comfyui.log para o erro completo."
        )


class LipForcing14B:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_input_files(VIDEO_EXTENSIONS),),
                "audio": (_input_files(AUDIO_EXTENSIONS),),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "decoder": (
                    ["streaming_taehv", "batch_taehv", "wan_vae"],
                    {"default": "streaming_taehv"},
                ),
                "exact_audio_duration": ("BOOLEAN", {"default": True}),
                "filename_prefix": (
                    "STRING",
                    {"default": "LipForcing14B_Final"},
                ),
            },
            "optional": {
                "quality_mode": (
                    list(QUALITY_MODES),
                    {"default": "segmentwise_max_quality"},
                ),
                "composite_mode": (
                    list(COMPOSITE_MODES),
                    {"default": "mouth_only"},
                ),
                "save_aligned_debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "VHS_FILENAMES")
    RETURN_NAMES = ("output_path", "filenames")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "InfiniteTalk/Lip Forcing"

    @classmethod
    def IS_CHANGED(cls, video: str, audio: str, **_kwargs):
        video_path = _resolved_input(video, VIDEO_EXTENSIONS)
        audio_path = _resolved_input(audio, AUDIO_EXTENSIONS)
        return f"{_path_digest(video_path)}:{_path_digest(audio_path)}"

    @classmethod
    def VALIDATE_INPUTS(cls, video: str, audio: str):
        try:
            _resolved_input(video, VIDEO_EXTENSIONS)
            _resolved_input(audio, AUDIO_EXTENSIONS)
        except (TypeError, ValueError, OSError) as error:
            return str(error)
        return True

    def generate(
        self,
        video: str,
        audio: str,
        seed: int,
        decoder: str,
        exact_audio_duration: bool,
        filename_prefix: str,
        quality_mode: str = "manual_decoder",
        composite_mode: str = "mouth_only",
        save_aligned_debug: bool = False,
    ):
        with _RUN_LOCK:
            _require_runtime()
            _check_vram()
            script_name, streaming_decoder = _quality_pipeline(
                quality_mode, decoder
            )
            if composite_mode not in COMPOSITE_MODES:
                raise ValueError(f"Modo de composicao invalido: {composite_mode}")
            if save_aligned_debug and script_name != "inference_segmentwise.py":
                raise ValueError(
                    "save_aligned_debug requer quality_mode="
                    "segmentwise_max_quality"
                )
            video_path = _resolved_input(video, VIDEO_EXTENSIONS)
            audio_path = _resolved_input(audio, AUDIO_EXTENSIONS)
            duration = _audio_duration(audio_path)
            latent_frames = _exact_latent_frames(duration)
            output_path, filename = _next_output(_safe_prefix(filename_prefix))
            temp_root = Path(folder_paths.get_temp_directory())
            temp_root.mkdir(parents=True, exist_ok=True)
            workdir = Path(tempfile.mkdtemp(prefix="lipforcing14b-", dir=temp_root))
            generated = workdir / "generated.mp4"
            generated_aligned = workdir / "generated_aligned.mp4"
            aligned_output_path = output_path.with_name(
                f"{output_path.stem}_aligned.mp4"
            )
            # The official cache filename is based only on the video stem.
            # Isolate it by content to prevent collisions between equal upload names.
            face_cache = MODELS_ROOT / "face_cache" / _path_digest(video_path)[:16]
            face_cache.mkdir(parents=True, exist_ok=True)
            try:
                _unload_comfy_models()
                command = [
                    str(RUNTIME_PYTHON),
                    str(RUNTIME_ROOT / "scripts/inference" / script_name),
                    "--ckpt_path",
                    str(MODEL_PATHS["checkpoint"]),
                    "--vae_path",
                    str(MODEL_PATHS["vae"]),
                    "--wav2vec_path",
                    str(MODEL_PATHS["wav2vec"]),
                    "--mask_path",
                    str(MODEL_PATHS["mask"]),
                    "--text_embeds_path",
                    str(MODEL_PATHS["text_embeds"]),
                    "--video_path",
                    str(video_path),
                    "--audio_path",
                    str(audio_path),
                    "--output_path",
                    str(generated),
                    "--seed",
                    str(seed),
                    "--fps",
                    "25",
                    "--dtype",
                    "bf16",
                    "--local_attn_size",
                    "7",
                    "--sink_size",
                    "1",
                    "--face_cache_dir",
                    str(face_cache),
                ]
                if streaming_decoder is not None:
                    command.extend(["--streaming_decoder", streaming_decoder])
                    if streaming_decoder in ("streaming_taehv", "batch_taehv"):
                        command.extend(["--taehv_ckpt", str(MODEL_PATHS["taehv"])])
                if composite_mode == "full_face":
                    command.append("--composite_full_face")
                if save_aligned_debug:
                    command.append("--save_aligned")
                if exact_audio_duration:
                    command.extend(["--num_latent_frames", str(latent_frames)])
                _run_streaming(command, RUNTIME_ROOT)
                if not generated.exists() or generated.stat().st_size == 0:
                    raise RuntimeError("Lip Forcing nao produziu o MP4 intermediario")

                if exact_audio_duration:
                    _mux_exact_audio(
                        generated, audio_path, output_path, duration
                    )
                else:
                    shutil.move(str(generated), str(output_path))
                if save_aligned_debug:
                    if not generated_aligned.exists() or generated_aligned.stat().st_size == 0:
                        raise RuntimeError(
                            "Lip Forcing nao produziu o video alinhado de diagnostico"
                        )
                    if exact_audio_duration:
                        _mux_exact_audio(
                            generated_aligned,
                            audio_path,
                            aligned_output_path,
                            duration,
                        )
                    else:
                        shutil.move(str(generated_aligned), str(aligned_output_path))
            finally:
                shutil.rmtree(workdir, ignore_errors=True)

            preview = {
                "filename": filename,
                "subfolder": "",
                "type": "output",
                "format": "video/h264-mp4",
                "frame_rate": 25,
                "fullpath": str(output_path),
            }
            previews = [preview]
            output_filenames = [str(output_path)]
            if save_aligned_debug:
                aligned_filename = aligned_output_path.name
                previews.append(
                    {
                        "filename": aligned_filename,
                        "subfolder": "",
                        "type": "output",
                        "format": "video/h264-mp4",
                        "frame_rate": 25,
                        "fullpath": str(aligned_output_path),
                    }
                )
                output_filenames.append(str(aligned_output_path))
                print(
                    f"[LipForcing14B] Diagnostico alinhado: {aligned_output_path}",
                    flush=True,
                )
            filenames = (True, output_filenames)
            print(f"[LipForcing14B] Saida final: {output_path}", flush=True)
            return {
                "ui": {"gifs": previews},
                "result": (str(output_path), filenames),
            }


NODE_CLASS_MAPPINGS = {
    "LipForcing14B": LipForcing14B,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LipForcing14B": "Lip Forcing 14B - Generate",
}
