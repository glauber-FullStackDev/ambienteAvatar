#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import boto3
from botocore.config import Config
import requests
import runpod

from workflow_api import build_job_workflow

LOG = logging.getLogger("ltx23-serverless")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
COMFYUI_URL = f"http://127.0.0.1:{os.environ.get('COMFYUI_PORT', '8188')}"
WORKFLOW_PATH = Path("/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_api.json")
PERSONAL_LORA_SOURCE = Path("/opt/ltx23-assets/glauberavatar.safetensors")
PERSONAL_LORA_TARGET = COMFYUI_HOME / "models/loras/glauberavatar.safetensors"
DEFAULTS = {
    "width": 720,
    "height": 1280,
    "duration_seconds": 18.0,
    "fps": 24,
    "audio_start_seconds": 0.0,
    "lora_strength": 1.0,
    "image_strength": 0.7,
}
MAX_INPUT_BYTES = int(os.environ.get("MAX_INPUT_BYTES", str(100 * 1024 * 1024)))
COMFY_TIMEOUT_SECONDS = int(os.environ.get("COMFY_TIMEOUT_SECONDS", "21600"))
POLL_SECONDS = float(os.environ.get("COMFY_POLL_SECONDS", "2"))


class InputError(ValueError):
    pass


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Variável obrigatória ausente: {name}")
    return value


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=_required_env("MINIO_ENDPOINT"),
        region_name=os.environ.get("MINIO_REGION", "us-east-1"),
        aws_access_key_id=_required_env("MINIO_ACCESS_KEY"),
        aws_secret_access_key=_required_env("MINIO_SECRET_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": os.environ.get("MINIO_ADDRESSING_STYLE", "path")}),
    )


def _allowed_input_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise InputError("URLs de entrada precisam ser HTTP(S) completas")
    configured = os.environ.get("MINIO_ALLOWED_HOST")
    expected = urlparse(_required_env("MINIO_ENDPOINT")).netloc
    allowed = {host.strip() for host in (configured or expected).split(",") if host.strip()}
    if parsed.netloc not in allowed:
        raise InputError("URL de entrada não pertence ao MinIO permitido")


def _file_extension(url: str, fallback: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix and len(suffix) <= 8 and suffix[1:].isalnum() else fallback


def _download(url: str, target: Path) -> None:
    _allowed_input_host(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with requests.get(url, stream=True, timeout=(15, 300)) as response:
        response.raise_for_status()
        with target.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_INPUT_BYTES:
                    raise InputError(f"arquivo excede MAX_INPUT_BYTES ({MAX_INPUT_BYTES} bytes)")
                file_handle.write(chunk)
    if total == 0:
        raise InputError("arquivo de entrada vazio")


def _number(value: Any, name: str, *, minimum: float, maximum: float, integer: bool = False) -> int | float:
    if isinstance(value, bool):
        raise InputError(f"{name} precisa ser numérico")
    try:
        converted = int(value) if integer else float(value)
    except (TypeError, ValueError) as error:
        raise InputError(f"{name} precisa ser numérico") from error
    if not minimum <= converted <= maximum:
        raise InputError(f"{name} precisa estar entre {minimum} e {maximum}")
    return converted


def validate_input(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InputError("input precisa ser um objeto JSON")
    image_url = raw.get("image_url")
    audio_url = raw.get("audio_url")
    prompt = raw.get("prompt")
    if not all(isinstance(value, str) and value.strip() for value in (image_url, audio_url, prompt)):
        raise InputError("image_url, audio_url e prompt são obrigatórios")
    if len(prompt) > 12_000:
        raise InputError("prompt excede 12000 caracteres")
    width = _number(raw.get("width", DEFAULTS["width"]), "width", minimum=256, maximum=1920, integer=True)
    height = _number(raw.get("height", DEFAULTS["height"]), "height", minimum=256, maximum=1920, integer=True)
    if width % 32 or height % 32:
        raise InputError("width e height precisam ser divisíveis por 32")
    seed = raw.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") % (2**63 - 1)
    else:
        seed = _number(seed, "seed", minimum=0, maximum=2**63 - 1, integer=True)
    return {
        "image_url": image_url,
        "audio_url": audio_url,
        "prompt": prompt.strip(),
        "width": width,
        "height": height,
        "duration_seconds": _number(raw.get("duration_seconds", DEFAULTS["duration_seconds"]), "duration_seconds", minimum=1, maximum=30),
        "fps": _number(raw.get("fps", DEFAULTS["fps"]), "fps", minimum=8, maximum=30, integer=True),
        "audio_start_seconds": _number(raw.get("audio_start_seconds", DEFAULTS["audio_start_seconds"]), "audio_start_seconds", minimum=0, maximum=3600),
        "seed": seed,
        "lora_strength": _number(raw.get("lora_strength", DEFAULTS["lora_strength"]), "lora_strength", minimum=0, maximum=2),
        "image_strength": _number(raw.get("image_strength", DEFAULTS["image_strength"]), "image_strength", minimum=0, maximum=1),
    }


def _ensure_personal_lora() -> None:
    if not PERSONAL_LORA_SOURCE.is_file():
        raise RuntimeError("LoRA pessoal não encontrado na imagem")
    PERSONAL_LORA_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not PERSONAL_LORA_TARGET.is_file():
        shutil.copyfile(PERSONAL_LORA_SOURCE, PERSONAL_LORA_TARGET)


def start_comfyui() -> None:
    _ensure_personal_lora()
    subprocess.run([sys.executable, "/opt/serverless/src/bootstrap_models.py"], check=True)
    log_path = Path("/var/log/portal/comfyui-serverless.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(COMFYUI_HOME / "main.py"), "--listen", "127.0.0.1", "--port", os.environ.get("COMFYUI_PORT", "8188"), "--preview-method", "none"]
    with log_path.open("ab") as log_file:
        subprocess.Popen(command, cwd=COMFYUI_HOME, stdout=log_file, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + int(os.environ.get("COMFY_STARTUP_TIMEOUT_SECONDS", "21600"))
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{COMFYUI_URL}/system_stats", timeout=5).ok:
                LOG.info("ComfyUI pronto")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"ComfyUI não ficou pronto; consulte {log_path}")


def _wait_for_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + COMFY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=20)
        response.raise_for_status()
        history = response.json().get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("completed"):
                return history
            messages = status.get("messages", [])
            if any(message[0] == "execution_error" for message in messages if message):
                raise RuntimeError(f"ComfyUI falhou: {messages}")
        time.sleep(POLL_SECONDS)
    raise TimeoutError("tempo máximo de execução ComfyUI excedido")


def _newest_file(folder: Path, suffix: str) -> Path:
    files = [path for path in folder.rglob(f"*{suffix}") if path.is_file()]
    if not files:
        raise RuntimeError(f"resultado {suffix} não encontrado em {folder}")
    return max(files, key=lambda path: path.stat().st_mtime)


def _upload_result(client, path: Path, key: str) -> str:
    bucket = _required_env("MINIO_BUCKET")
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "video/mp4" if path.suffix == ".mp4" else "image/png"})
    return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=int(os.environ.get("MINIO_PRESIGN_EXPIRES_SECONDS", "86400")))


def _cleanup(job_id: str) -> None:
    for path in (
        COMFYUI_HOME / "input/jobs" / job_id,
        COMFYUI_HOME / "output/video/jobs" / job_id,
        COMFYUI_HOME / "output/images/last_frame/jobs" / job_id,
    ):
        shutil.rmtree(path, ignore_errors=True)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_id = str(job.get("id") or uuid4())
    started_at = time.monotonic()
    values = validate_input(job.get("input", {}))
    image_name = f"jobs/{job_id}/input{_file_extension(values['image_url'], '.png')}"
    audio_name = f"jobs/{job_id}/audio{_file_extension(values['audio_url'], '.wav')}"
    values.update({"image_filename": image_name, "audio_filename": audio_name})
    input_root = COMFYUI_HOME / "input"
    try:
        _download(values["image_url"], input_root / image_name)
        _download(values["audio_url"], input_root / audio_name)
        template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        workflow = build_job_workflow(template, values, job_id)
        response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": job_id}, timeout=60)
        response.raise_for_status()
        prompt_id = response.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI recusou o workflow: {response.text}")
        _wait_for_history(prompt_id)
        video = _newest_file(COMFYUI_HOME / "output/video/jobs" / job_id, ".mp4")
        last_frame = _newest_file(COMFYUI_HOME / "output/images/last_frame/jobs" / job_id, ".png")
        prefix = os.environ.get("MINIO_OUTPUT_PREFIX", "ltx-ia2v/results").strip("/")
        client = _s3_client()
        video_url = _upload_result(client, video, f"{prefix}/{job_id}/video.mp4")
        last_frame_url = _upload_result(client, last_frame, f"{prefix}/{job_id}/last_frame.png")
        return {
            "job_id": job_id,
            "comfy_prompt_id": prompt_id,
            "video_url": video_url,
            "last_frame_url": last_frame_url,
            "parameters": {key: values[key] for key in DEFAULTS | {"width": None, "height": None, "seed": None}},
            "execution_seconds": round(time.monotonic() - started_at, 3),
        }
    finally:
        _cleanup(job_id)


if __name__ == "__main__":
    start_comfyui()
    runpod.serverless.start({"handler": handler})
