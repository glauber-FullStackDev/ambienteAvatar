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


def _file_digest(filename: str) -> str:
    return _path_digest(Path(folder_paths.get_annotated_filepath(filename)))


class LipForcingLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_input_files(VIDEO_EXTENSIONS), {"video_upload": True})
            }
        }

    RETURN_TYPES = ("LIPFORCING_VIDEO_PATH",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "load"
    CATEGORY = "InfiniteTalk/Lip Forcing"

    def load(self, video: str):
        return (str(_resolved_input(video, VIDEO_EXTENSIONS)),)

    @classmethod
    def IS_CHANGED(cls, video: str):
        return _file_digest(video)

    @classmethod
    def VALIDATE_INPUTS(cls, video: str):
        try:
            _resolved_input(video, VIDEO_EXTENSIONS)
        except (ValueError, OSError) as error:
            return str(error)
        return True


class LipForcingLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (_input_files(AUDIO_EXTENSIONS), {"audio_upload": True})
            }
        }

    RETURN_TYPES = ("LIPFORCING_AUDIO_PATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "load"
    CATEGORY = "InfiniteTalk/Lip Forcing"

    def load(self, audio: str):
        return (str(_resolved_input(audio, AUDIO_EXTENSIONS)),)

    @classmethod
    def IS_CHANGED(cls, audio: str):
        return _file_digest(audio)

    @classmethod
    def VALIDATE_INPUTS(cls, audio: str):
        try:
            _resolved_input(audio, AUDIO_EXTENSIONS)
        except (ValueError, OSError) as error:
            return str(error)
        return True


def _require_runtime() -> None:
    missing = []
    for path in (RUNTIME_PYTHON, RUNTIME_ROOT / "scripts/inference/inference_streaming.py"):
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
                "video_path": ("LIPFORCING_VIDEO_PATH",),
                "audio_path": ("LIPFORCING_AUDIO_PATH",),
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
            }
        }

    RETURN_TYPES = ("STRING", "VHS_FILENAMES")
    RETURN_NAMES = ("output_path", "filenames")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "InfiniteTalk/Lip Forcing"

    def generate(
        self,
        video_path: str,
        audio_path: str,
        seed: int,
        decoder: str,
        exact_audio_duration: bool,
        filename_prefix: str,
    ):
        with _RUN_LOCK:
            _require_runtime()
            _check_vram()
            video = Path(video_path).resolve()
            audio = Path(audio_path).resolve()
            _validate_resolved_input(video, VIDEO_EXTENSIONS)
            _validate_resolved_input(audio, AUDIO_EXTENSIONS)
            duration = _audio_duration(audio)
            latent_frames = _exact_latent_frames(duration)
            output_path, filename = _next_output(_safe_prefix(filename_prefix))
            temp_root = Path(folder_paths.get_temp_directory())
            temp_root.mkdir(parents=True, exist_ok=True)
            workdir = Path(tempfile.mkdtemp(prefix="lipforcing14b-", dir=temp_root))
            generated = workdir / "generated.mp4"
            # The official cache filename is based only on the video stem.
            # Isolate it by content to prevent collisions between equal upload names.
            face_cache = MODELS_ROOT / "face_cache" / _path_digest(video)[:16]
            face_cache.mkdir(parents=True, exist_ok=True)
            try:
                _unload_comfy_models()
                command = [
                    str(RUNTIME_PYTHON),
                    str(RUNTIME_ROOT / "scripts/inference/inference_streaming.py"),
                    "--ckpt_path",
                    str(MODEL_PATHS["checkpoint"]),
                    "--vae_path",
                    str(MODEL_PATHS["vae"]),
                    "--wav2vec_path",
                    str(MODEL_PATHS["wav2vec"]),
                    "--mask_path",
                    str(MODEL_PATHS["mask"]),
                    "--taehv_ckpt",
                    str(MODEL_PATHS["taehv"]),
                    "--text_embeds_path",
                    str(MODEL_PATHS["text_embeds"]),
                    "--video_path",
                    str(video),
                    "--audio_path",
                    str(audio),
                    "--output_path",
                    str(generated),
                    "--seed",
                    str(seed),
                    "--fps",
                    "25",
                    "--dtype",
                    "bf16",
                    "--streaming_decoder",
                    decoder,
                    "--local_attn_size",
                    "7",
                    "--sink_size",
                    "1",
                    "--face_cache_dir",
                    str(face_cache),
                ]
                if exact_audio_duration:
                    command.extend(["--num_latent_frames", str(latent_frames)])
                _run_streaming(command, RUNTIME_ROOT)
                if not generated.exists() or generated.stat().st_size == 0:
                    raise RuntimeError("Lip Forcing nao produziu o MP4 intermediario")

                if exact_audio_duration:
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
                            str(output_path),
                        ],
                        check=True,
                    )
                else:
                    shutil.move(str(generated), str(output_path))
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
            filenames = (True, [str(output_path)])
            print(f"[LipForcing14B] Saida final: {output_path}", flush=True)
            return {
                "ui": {"gifs": [preview]},
                "result": (str(output_path), filenames),
            }


NODE_CLASS_MAPPINGS = {
    "LipForcingLoadVideo": LipForcingLoadVideo,
    "LipForcingLoadAudio": LipForcingLoadAudio,
    "LipForcing14B": LipForcing14B,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LipForcingLoadVideo": "Lip Forcing - Upload Video",
    "LipForcingLoadAudio": "Lip Forcing - Upload Audio",
    "LipForcing14B": "Lip Forcing 14B - Generate",
}
