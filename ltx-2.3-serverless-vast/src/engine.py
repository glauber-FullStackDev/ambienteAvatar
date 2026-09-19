#!/usr/bin/env python3
"""Core engine: download inputs, run the ComfyUI workflow, upload results to MinIO.

Adapted from the Runpod serverless handler; replaces the Runpod queue lifecycle
with plain functions callable by the FastAPI model server, and adds MinIO key
namespacing by project so every job is traceable to its project.
"""
from __future__ import annotations

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

import boto3
from botocore.config import Config
import requests

from workflow_api import build_job_workflow

LOG = logging.getLogger("ltx23-vast.engine")

COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
COMFYUI_URL = f"http://127.0.0.1:{os.environ.get('COMFYUI_PORT', '8188')}"
WORKFLOW_PATH = Path("/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_api.json")
PERSONAL_LORA_SOURCE = Path("/opt/ltx23-assets/glauberavatar.safetensors")
PERSONAL_LORA_TARGET = COMFYUI_HOME / "models/loras/glauberavatar.safetensors"
MAX_INPUT_BYTES = int(os.environ.get("MAX_INPUT_BYTES", str(100 * 1024 * 1024)))
COMFY_TIMEOUT_SECONDS = int(os.environ.get("COMFY_TIMEOUT_SECONDS", "21600"))
POLL_SECONDS = float(os.environ.get("COMFY_POLL_SECONDS", "2"))
S3_CONNECT_TIMEOUT_SECONDS = int(os.environ.get("S3_CONNECT_TIMEOUT_SECONDS", "20"))
S3_READ_TIMEOUT_SECONDS = int(os.environ.get("S3_READ_TIMEOUT_SECONDS", "600"))
S3_UPLOAD_ATTEMPTS = int(os.environ.get("S3_UPLOAD_ATTEMPTS", "3"))
PERSIST_JOB_STATUS = os.environ.get("PERSIST_JOB_STATUS", "1") == "1"
MODEL_VOLUME_ROOT = Path(os.environ.get("MODEL_VOLUME_ROOT", "/vast-volume"))
ENV_ALIASES = {
    "MINIO_ENDPOINT": "S3_ENDPOINT",
    "MINIO_BUCKET": "S3_BUCKET",
    "MINIO_REGION": "S3_REGION",
    "MINIO_ACCESS_KEY": "S3_ACCESS_KEY_ID",
    "MINIO_SECRET_KEY": "S3_SECRET_ACCESS_KEY",
}


def _required_env(name: str) -> str:
    value = os.environ.get(name) or os.environ.get(ENV_ALIASES.get(name, ""))
    if not value:
        raise RuntimeError(f"Variável obrigatória ausente: {name}")
    return value


def _s3_client():
    client = boto3.client(
        "s3",
        endpoint_url=_required_env("MINIO_ENDPOINT"),
        region_name=os.environ.get("MINIO_REGION", "us-east-1"),
        aws_access_key_id=_required_env("MINIO_ACCESS_KEY"),
        aws_secret_access_key=_required_env("MINIO_SECRET_KEY"),
        config=Config(
            signature_version="s3v4",
            connect_timeout=S3_CONNECT_TIMEOUT_SECONDS,
            read_timeout=S3_READ_TIMEOUT_SECONDS,
            retries={"max_attempts": 5, "mode": "standard"},
            s3={"addressing_style": os.environ.get("MINIO_ADDRESSING_STYLE", "path")},
        ),
    )

    def remove_expect_header(request, **_):
        request.headers.pop("Expect", None)
        return None

    client.meta.events.register("before-send.s3.PutObject", remove_expect_header)
    return client


def _allowed_input_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("URLs de entrada precisam ser HTTP(S) completas")
    configured = os.environ.get("MINIO_ALLOWED_HOST")
    expected = urlparse(_required_env("MINIO_ENDPOINT")).netloc
    allowed = {host.strip() for host in (configured or expected).split(",") if host.strip()}
    if parsed.netloc not in allowed:
        raise ValueError("URL de entrada não pertence ao MinIO permitido")


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
                    raise ValueError(f"arquivo excede MAX_INPUT_BYTES ({MAX_INPUT_BYTES} bytes)")
                file_handle.write(chunk)
    if total == 0:
        raise ValueError("arquivo de entrada vazio")


def _ensure_personal_lora() -> None:
    if not PERSONAL_LORA_SOURCE.is_file():
        raise RuntimeError("LoRA pessoal não encontrado na imagem")
    PERSONAL_LORA_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not PERSONAL_LORA_TARGET.is_file():
        shutil.copyfile(PERSONAL_LORA_SOURCE, PERSONAL_LORA_TARGET)


def _configure_model_storage() -> None:
    """Symlink ComfyUI models to the attached persistent Vast volume when present."""
    if not MODEL_VOLUME_ROOT.is_dir():
        LOG.warning("Volume persistente ausente em %s; modelos serão efêmeros", MODEL_VOLUME_ROOT)
        return
    persistent_models = MODEL_VOLUME_ROOT / "models"
    persistent_models.mkdir(parents=True, exist_ok=True)
    models_path = COMFYUI_HOME / "models"
    if models_path.is_symlink():
        if models_path.resolve() == persistent_models.resolve():
            return
        models_path.unlink()
    elif models_path.exists():
        shutil.rmtree(models_path)
    models_path.symlink_to(persistent_models, target_is_directory=True)
    os.environ["COMFYUI_MODELS"] = str(models_path)
    os.environ["HF_HOME"] = str(MODEL_VOLUME_ROOT / "huggingface")
    LOG.info("Modelos persistentes configurados em %s", persistent_models)


def _assert_workflow_nodes_available() -> None:
    template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    required = {node["class_type"] for node in template.values()}
    response = requests.get(f"{COMFYUI_URL}/object_info", timeout=30)
    response.raise_for_status()
    available = set(response.json())
    missing = sorted(required - available)
    if missing:
        raise RuntimeError("ComfyUI iniciou, mas faltam nós do workflow: " + ", ".join(missing))


def start_comfyui() -> None:
    _configure_model_storage()
    _ensure_personal_lora()
    subprocess.run([sys.executable, "/opt/serverless-vast/src/bootstrap_models.py"], check=True)
    log_path = Path("/var/log/portal/comfyui-serverless.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(COMFYUI_HOME / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        os.environ.get("COMFYUI_PORT", "8188"),
        "--preview-method",
        "none",
    ]
    with log_path.open("ab") as log_file:
        subprocess.Popen(command, cwd=COMFYUI_HOME, stdout=log_file, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + int(os.environ.get("COMFY_STARTUP_TIMEOUT_SECONDS", "21600"))
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{COMFYUI_URL}/system_stats", timeout=5).ok:
                _assert_workflow_nodes_available()
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
    content_type = "video/mp4" if path.suffix == ".mp4" else "image/png"
    for attempt in range(1, S3_UPLOAD_ATTEMPTS + 1):
        try:
            client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
            break
        except Exception:
            if attempt == S3_UPLOAD_ATTEMPTS:
                raise
            LOG.warning("Falha ao enviar %s (tentativa %s/%s)", key, attempt, S3_UPLOAD_ATTEMPTS)
            time.sleep(min(attempt * 3, 10))
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(os.environ.get("MINIO_PRESIGN_EXPIRES_SECONDS", "86400")),
    )


def _cleanup(job_id: str) -> None:
    for path in (
        COMFYUI_HOME / "input/jobs" / job_id,
        COMFYUI_HOME / "output/video/jobs" / job_id,
        COMFYUI_HOME / "output/images/last_frame/jobs" / job_id,
    ):
        shutil.rmtree(path, ignore_errors=True)


def output_prefix() -> str:
    return os.environ.get("MINIO_OUTPUT_PREFIX", "ltx-ia2v/results").strip("/")


def result_key(values: dict[str, Any], filename: str) -> str:
    return f"{output_prefix()}/{values['project_id']}/{values['job_id']}/{filename}"


def run_job(values: dict[str, Any], job_id: str) -> dict[str, Any]:
    started_at = time.monotonic()
    image_name = f"jobs/{job_id}/input{_file_extension(values['image_url'], '.png')}"
    audio_name = f"jobs/{job_id}/audio{_file_extension(values['audio_url'], '.wav')}"
    values = {**values, "image_filename": image_name, "audio_filename": audio_name}
    input_root = COMFYUI_HOME / "input"
    try:
        _download(values["image_url"], input_root / image_name)
        _download(values["audio_url"], input_root / audio_name)
        template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        workflow = build_job_workflow(template, values, job_id)
        response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": job_id}, timeout=60)
        if not response.ok:
            raise RuntimeError(f"ComfyUI recusou o workflow (HTTP {response.status_code}): {response.text[:10_000]}")
        prompt_id = response.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI recusou o workflow: {response.text}")
        _wait_for_history(prompt_id)
        video = _newest_file(COMFYUI_HOME / "output/video/jobs" / job_id, ".mp4")
        last_frame = _newest_file(COMFYUI_HOME / "output/images/last_frame/jobs" / job_id, ".png")
        client = _s3_client()
        video_url = _upload_result(client, video, result_key(values, "video.mp4"))
        last_frame_url = _upload_result(client, last_frame, result_key(values, "last_frame.png"))
        return {
            "job_id": job_id,
            "project_id": values["project_id"],
            "comfy_prompt_id": prompt_id,
            "video_url": video_url,
            "last_frame_url": last_frame_url,
            "parameters": {key: values[key] for key in ("width", "height", "duration_seconds", "fps", "audio_start_seconds", "seed", "lora_strength", "image_strength")},
            "execution_seconds": round(time.monotonic() - started_at, 3),
        }
    finally:
        _cleanup(job_id)


def persist_job_status(record: Any) -> str | None:
    """Mirror the job state into MinIO so the project backend can sync even if the webhook fails."""
    if not PERSIST_JOB_STATUS:
        return None
    key = result_key(record.params, "status.json")
    body = json.dumps(record.to_dict(), ensure_ascii=False).encode("utf-8")
    client = _s3_client()
    client.put_object(
        Bucket=_required_env("MINIO_BUCKET"),
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": _required_env("MINIO_BUCKET"), "Key": key},
        ExpiresIn=int(os.environ.get("MINIO_PRESIGN_EXPIRES_SECONDS", "86400")),
    )