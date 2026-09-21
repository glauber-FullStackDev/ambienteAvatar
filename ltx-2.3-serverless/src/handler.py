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
WORKFLOW_PATH = Path(
    os.environ.get(
        "WORKFLOW_PATH",
        "/opt/defaults/workflows/video_ltx2_5_ia2v_iclora_api.json",
    )
)
LEGACY_WORKFLOW_PATH = Path(
    os.environ.get(
        "LEGACY_WORKFLOW_PATH",
        "/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_api.json",
    )
)
PERSONAL_LORA_SOURCE = Path("/opt/ltx23-assets/glauberavatar.safetensors")
PERSONAL_LORA_TARGET = COMFYUI_HOME / "models/loras/glauberavatar.safetensors"
DEFAULTS = {
    "width": 704,
    "height": 1280,
    "duration_seconds": 18.0,
    "fps": 24,
    "audio_start_seconds": 0.0,
    "lora_strength": 0.7,
    "first_frame_strength": 1.0,
    "guiding_strength": 0.8,
    "iclora_strength": 0.9,
    "cfg": 1.0,
    "image_strength": 1.0,
    "enable_prompt_enhance": None,
}
DEFAULT_PROMPT = (
    "The person remains in the exact composition of the initial frame. "
    "He speaks following the supplied audio with precise lip sync. "
    "He looks directly at the camera. Only subtle natural facial movement, "
    "blinking, breathing and minimal head movement. "
    "The camera remains completely stationary."
)
MAX_INPUT_BYTES = int(os.environ.get("MAX_INPUT_BYTES", str(100 * 1024 * 1024)))
COMFY_TIMEOUT_SECONDS = int(os.environ.get("COMFY_TIMEOUT_SECONDS", "21600"))
POLL_SECONDS = float(os.environ.get("COMFY_POLL_SECONDS", "2"))
S3_CONNECT_TIMEOUT_SECONDS = int(os.environ.get("S3_CONNECT_TIMEOUT_SECONDS", "20"))
S3_READ_TIMEOUT_SECONDS = int(os.environ.get("S3_READ_TIMEOUT_SECONDS", "600"))
S3_UPLOAD_ATTEMPTS = int(os.environ.get("S3_UPLOAD_ATTEMPTS", "3"))
RUNPOD_VOLUME_ROOT = Path(os.environ.get("RUNPOD_VOLUME_ROOT", "/runpod-volume"))
ENV_ALIASES = {
    "MINIO_ENDPOINT": "S3_ENDPOINT",
    "MINIO_BUCKET": "S3_BUCKET",
    "MINIO_REGION": "S3_REGION",
    "MINIO_ACCESS_KEY": "S3_ACCESS_KEY_ID",
    "MINIO_SECRET_KEY": "S3_SECRET_ACCESS_KEY",
}


class InputError(ValueError):
    pass


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
    # Some S3-compatible gateways close a request while boto3 waits for the
    # optional 100-continue response. Sending the body directly is compatible
    # with S3 and avoids that proxy-specific failure mode.
    def remove_expect_header(request, **_):
        request.headers.pop("Expect", None)
        # Event handlers may return an HTTP response to short-circuit botocore;
        # this mutation hook must explicitly return None instead.
        return None

    client.meta.events.register("before-send.s3.PutObject", remove_expect_header)
    return client


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


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise InputError(f"{name} precisa ser booleano")


def _sigmas(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} precisa ser uma string de sigmas não vazia")
    value = value.strip()
    if len(value) > 500:
        raise InputError(f"{name} excede 500 caracteres")
    return value


def validate_input(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InputError("input precisa ser um objeto JSON")
    image_url = raw.get("image_url")
    audio_url = raw.get("audio_url")
    prompt = raw.get("prompt")
    if not all(isinstance(value, str) and value.strip() for value in (image_url, audio_url)):
        raise InputError("image_url e audio_url são obrigatórios")
    if prompt is not None and not isinstance(prompt, str):
        raise InputError("prompt precisa ser texto")
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    if len(prompt) > 12_000:
        raise InputError("prompt excede 12000 caracteres")
    if not prompt:
        prompt = DEFAULT_PROMPT
    negative_prompt = raw.get("negative_prompt")
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise InputError("negative_prompt precisa ser texto")
    negative_prompt = (negative_prompt or "").strip()
    if len(negative_prompt) > 12_000:
        raise InputError("negative_prompt excede 12000 caracteres")
    reference_image_url = raw.get("reference_image_url")
    if reference_image_url is not None:
        if not isinstance(reference_image_url, str) or not reference_image_url.strip():
            raise InputError("reference_image_url precisa ser uma URL válida")
        reference_image_url = reference_image_url.strip()
    reference_frame_idx = raw.get("reference_frame_idx")
    if reference_frame_idx is None:
        reference_frame_idx = -1
    else:
        reference_frame_idx = _number(reference_frame_idx, "reference_frame_idx", minimum=-4096, maximum=4096, integer=True)
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
        "reference_image_url": reference_image_url,
        "reference_frame_idx": reference_frame_idx,
        "reference_guiding_strength": _number(raw.get("reference_guiding_strength", DEFAULTS["guiding_strength"]), "reference_guiding_strength", minimum=0, maximum=1),
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "duration_seconds": _number(raw.get("duration_seconds", DEFAULTS["duration_seconds"]), "duration_seconds", minimum=1, maximum=30),
        "fps": _number(raw.get("fps", DEFAULTS["fps"]), "fps", minimum=8, maximum=30, integer=True),
        "audio_start_seconds": _number(raw.get("audio_start_seconds", DEFAULTS["audio_start_seconds"]), "audio_start_seconds", minimum=0, maximum=3600),
        "seed": seed,
        "lora_strength": _number(raw.get("lora_strength", DEFAULTS["lora_strength"]), "lora_strength", minimum=0, maximum=1),
        "first_frame_strength": _number(raw.get("first_frame_strength", DEFAULTS["first_frame_strength"]), "first_frame_strength", minimum=0, maximum=1),
        "guiding_strength": _number(raw.get("guiding_strength", DEFAULTS["guiding_strength"]), "guiding_strength", minimum=0, maximum=1),
        "iclora_strength": _number(raw.get("iclora_strength", DEFAULTS["iclora_strength"]), "iclora_strength", minimum=0, maximum=1.5),
        "cfg": _number(raw.get("cfg", DEFAULTS["cfg"]), "cfg", minimum=0, maximum=8),
        "image_strength": _number(raw.get("image_strength", DEFAULTS["image_strength"]), "image_strength", minimum=0, maximum=1),
        "enable_prompt_enhance": None if raw.get("enable_prompt_enhance") is None else _boolean(raw.get("enable_prompt_enhance"), "enable_prompt_enhance"),
        "base_sigmas": _sigmas(raw.get("base_sigmas"), "base_sigmas"),
        "refine_sigmas": _sigmas(raw.get("refine_sigmas"), "refine_sigmas"),
    }


def _ensure_personal_lora() -> None:
    if not PERSONAL_LORA_SOURCE.is_file():
        raise RuntimeError("LoRA pessoal não encontrado na imagem")
    PERSONAL_LORA_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not PERSONAL_LORA_TARGET.is_file():
        shutil.copyfile(PERSONAL_LORA_SOURCE, PERSONAL_LORA_TARGET)


def _configure_model_storage() -> None:
    """Use the attached Runpod Network Volume without requiring a custom mount.

    Serverless mounts a Network Volume at /runpod-volume.  The model downloader
    and ComfyUI both use /opt/ComfyUI/models, so make that directory a symlink
    when a volume is available.  With no attached volume, local container disk
    remains a supported development fallback.
    """
    if not RUNPOD_VOLUME_ROOT.is_dir():
        LOG.warning("Network Volume ausente; modelos serão efêmeros neste worker")
        return

    persistent_models = RUNPOD_VOLUME_ROOT / "models"
    persistent_models.mkdir(parents=True, exist_ok=True)
    models_path = COMFYUI_HOME / "models"
    if models_path.is_symlink():
        if models_path.resolve() == persistent_models.resolve():
            return
        models_path.unlink()
    elif models_path.exists():
        # This directory is created empty by the Docker image.  Model files are
        # downloaded only after this point, so replacing it cannot lose output.
        shutil.rmtree(models_path)
    models_path.symlink_to(persistent_models, target_is_directory=True)
    os.environ["COMFYUI_MODELS"] = str(models_path)
    os.environ["HF_HOME"] = str(RUNPOD_VOLUME_ROOT / "huggingface")
    LOG.info("Modelos persistentes configurados em %s", persistent_models)


def _assert_workflow_nodes_available() -> None:
    """Fail the worker at boot with a useful message if a custom node is absent."""
    template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    nodes = template["prompt"].values() if "prompt" in template else template.values()
    required = {
        node["class_type"]
        for node in nodes
        if isinstance(node, dict) and "class_type" in node
    }
    response = requests.get(f"{COMFYUI_URL}/object_info", timeout=30)
    response.raise_for_status()
    available = set(response.json())
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            "ComfyUI iniciou, mas faltam nós do workflow: " + ", ".join(missing)
        )


_COMFYUI_PROC: subprocess.Popen | None = None


def _comfyui_alive() -> bool:
    try:
        return requests.get(f"{COMFYUI_URL}/system_stats", timeout=5).ok
    except requests.RequestException:
        return False


def _launch_comfyui() -> None:
    global _COMFYUI_PROC
    log_path = Path("/var/log/portal/comfyui-serverless.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(COMFYUI_HOME / "main.py"), "--listen", "127.0.0.1", "--port", os.environ.get("COMFYUI_PORT", "8188"), "--preview-method", "none"]
    with log_path.open("ab") as log_file:
        _COMFYUI_PROC = subprocess.Popen(command, cwd=COMFYUI_HOME, stdout=log_file, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + int(os.environ.get("COMFY_STARTUP_TIMEOUT_SECONDS", "21600"))
    while time.monotonic() < deadline:
        if _comfyui_alive():
            _assert_workflow_nodes_available()
            LOG.info("ComfyUI pronto")
            return
        if _COMFYUI_PROC.poll() is not None:
            break
        time.sleep(2)
    raise RuntimeError(f"ComfyUI não ficou pronto; consulte {log_path}")


def _stop_comfyui() -> None:
    global _COMFYUI_PROC
    process = _COMFYUI_PROC
    _COMFYUI_PROC = None
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _restart_comfyui(reason: str) -> None:
    LOG.warning("Reiniciando ComfyUI: %s", reason)
    _stop_comfyui()
    shutil.rmtree(COMFYUI_HOME / "temp", ignore_errors=True)
    _launch_comfyui()


def _ensure_comfyui() -> None:
    if _comfyui_alive():
        return
    _restart_comfyui("ComfyUI não respondeu ao health check")


def start_comfyui() -> None:
    _configure_model_storage()
    _ensure_personal_lora()
    subprocess.run([sys.executable, "/opt/serverless/src/bootstrap_models.py"], check=True)
    _launch_comfyui()


def _wait_for_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + COMFY_TIMEOUT_SECONDS
    consecutive_failures = 0
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=20)
            response.raise_for_status()
        except requests.RequestException as error:
            consecutive_failures += 1
            if consecutive_failures >= 3 or not _comfyui_alive():
                _restart_comfyui(f"ComfyUI indisponível durante o job (prompt {prompt_id})")
                raise RuntimeError(
                    "ComfyUI caiu durante a execução do job; prompt perdido"
                ) from error
            LOG.warning(
                "Falha transitória ao consultar /history (%s/3): %s",
                consecutive_failures,
                error,
            )
            time.sleep(min(POLL_SECONDS * consecutive_failures, 10))
            continue
        consecutive_failures = 0
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
    reference_name = (
        f"jobs/{job_id}/reference{_file_extension(values['reference_image_url'], '.png')}"
        if values.get("reference_image_url")
        else None
    )
    values.update(
        {
            "image_filename": image_name,
            "audio_filename": audio_name,
            "reference_image_filename": reference_name,
        }
    )
    input_root = COMFYUI_HOME / "input"
    try:
        _ensure_comfyui()
        _download(values["image_url"], input_root / image_name)
        _download(values["audio_url"], input_root / audio_name)
        if reference_name:
            _download(values["reference_image_url"], input_root / reference_name)
        template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        workflow = build_job_workflow(template, values, job_id)
        response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": job_id}, timeout=60)
        if not response.ok:
            raise RuntimeError(
                f"ComfyUI recusou o workflow (HTTP {response.status_code}): "
                f"{response.text[:10_000]}"
            )
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
            "parameters": {
                key: values[key]
                for key in (
                    "prompt",
                    "negative_prompt",
                    "duration_seconds",
                    "fps",
                    "audio_start_seconds",
                    "lora_strength",
                    "first_frame_strength",
                    "guiding_strength",
                    "reference_frame_idx",
                    "reference_guiding_strength",
                    "iclora_strength",
                    "cfg",
                    "enable_prompt_enhance",
                    "base_sigmas",
                    "refine_sigmas",
                )
            },
            "execution_seconds": round(time.monotonic() - started_at, 3),
        }
    finally:
        _cleanup(job_id)


if __name__ == "__main__":
    start_comfyui()
    runpod.serverless.start({"handler": handler})
