#!/usr/bin/env python3
"""Boot the Vast serverless model server.

Starts ComfyUI, waits for it, brings up the FastAPI model server, writes the
readiness marker the PyWorker tails, then launches worker.py from PYWORKER_REPO
(option A) — or the baked-in mirror — without duplicating a worker already
managed by the Vast provisioning layer.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import requests

from engine import start_comfyui

LOG = logging.getLogger("ltx23-vast.boot")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "18080"))
WORKER_PORT = int(os.environ.setdefault("WORKER_PORT", "3000"))
MODEL_LOG_FILE = Path(os.environ.get("MODEL_LOG_FILE", "/var/log/model/model.log"))
PYWORKER_REPO = os.environ.get("PYWORKER_REPO", "").strip()
PYWORKER_REF = os.environ.get("PYWORKER_REF", "").strip()
PYWORKER_MIRROR = Path(os.environ.get("PYWORKER_SRC", "/opt/serverless-vast/pyworker"))
PYWORKER_CLONE = Path("/opt/pyworker/repo")
BOOT_LOG_FILE = Path("/var/log/portal/boot.log")
COMFYUI_LOG_FILE = Path("/var/log/portal/comfyui-serverless.log")


def _write_ready_marker() -> None:
    MODEL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write("Model server ready\n")


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _wait_model_server() -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{MODEL_SERVER_PORT}/health", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("model server não ficou pronto a tempo")


def _resolve_worker() -> Path:
    if PYWORKER_REPO:
        try:
            shutil.rmtree(PYWORKER_CLONE, ignore_errors=True)
            command = ["git", "clone", "--depth", "1"]
            if PYWORKER_REF:
                command += ["--branch", PYWORKER_REF, "--single-branch"]
            command += [PYWORKER_REPO, str(PYWORKER_CLONE)]
            subprocess.run(command, check=True, timeout=300)
            requirements = PYWORKER_CLONE / "requirements.txt"
            if requirements.is_file():
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=False)
            worker = PYWORKER_CLONE / "worker.py"
            if not worker.is_file():
                raise RuntimeError("PYWORKER_REPO não contém worker.py na raiz")
            LOG.info("PyWorker clonado de %s", PYWORKER_REPO)
            return worker
        except Exception as error:  # noqa: BLE001 - fall back to mirror
            LOG.warning("Falha ao preparar PYWORKER_REPO (%s); usando mirror da imagem", error)

    worker = PYWORKER_MIRROR / "worker.py"
    if not worker.is_file():
        raise RuntimeError("nenhum worker.py disponível: defina PYWORKER_REPO ou use a imagem com mirror")
    LOG.info("PyWorker usando mirror embutido: %s", worker)
    return worker


def _uvicorn_log_config() -> dict:
    """Uvicorn logging with a file sink so ASGI errors land in boot.log."""
    from copy import deepcopy

    from uvicorn.config import LOGGING_CONFIG

    config = deepcopy(LOGGING_CONFIG)
    config["handlers"]["file"] = {
        "class": "logging.FileHandler",
        "filename": str(BOOT_LOG_FILE),
        "encoding": "utf-8",
    }
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = config.setdefault("loggers", {}).setdefault(name, {})
        logger.setdefault("handlers", []).append("file")
    config.setdefault("root", {"handlers": ["file"], "level": "INFO"})
    return config


def _install_crash_log_handlers() -> None:
    """Persist worker logs to MinIO on every container stop or crash.

    The engine destroys failed workers without keeping docker logs, so the
    boot/model/comfyui logs are uploaded to MinIO while shutting down.
    """

    def _handle_terminate(signum, _frame) -> None:
        LOG.error("recebido signal %s; enviando logs antes de encerrar", signum)
        try:
            _upload_crash_logs(f"signal-{signum}")
        finally:
            os._exit(128 + int(signum))

    signal.signal(signal.SIGTERM, _handle_terminate)
    signal.signal(signal.SIGINT, _handle_terminate)


def _upload_crash_logs(reason: str) -> None:
    try:
        from engine import _required_env, _s3_client

        client = _s3_client()
        bucket = _required_env("MINIO_BUCKET")
        prefix = os.environ.get("MINIO_OUTPUT_PREFIX", "ltx-ia2v/results").strip("/")
        stamp = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime())
        tag = os.environ.get("CONTAINER_ID") or socket.gethostname()
        for name, path in (
            ("boot.log", BOOT_LOG_FILE),
            ("model.log", MODEL_LOG_FILE),
            ("comfyui.log", COMFYUI_LOG_FILE),
        ):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            key = f"{prefix}/worker-logs/{stamp}-{tag}-{name}"
            try:
                client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "text/plain"})
                LOG.error("log de crash enviado (%s): %s", reason, key)
            except Exception as exc:  # noqa: BLE001 - best effort
                LOG.error("falha ao enviar %s: %s", key, exc)
    except Exception as exc:  # noqa: BLE001 - best effort
        LOG.error("não foi possível enviar logs de crash (%s): %s", reason, exc)


def _keep_alive() -> None:
    while True:
        time.sleep(3600)


def main() -> None:
    _install_crash_log_handlers()
    BOOT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    boot_file_handler = logging.FileHandler(BOOT_LOG_FILE, encoding="utf-8")
    boot_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(boot_file_handler)
    MODEL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_LOG_FILE.write_text("", encoding="utf-8")

    start_comfyui()

    import uvicorn

    import api

    config = uvicorn.Config(
        api.app,
        host="0.0.0.0",
        port=MODEL_SERVER_PORT,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
        log_config=_uvicorn_log_config(),
    )
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

    _wait_model_server()
    _write_ready_marker()
    LOG.info("Model server pronto em http://127.0.0.1:%s", MODEL_SERVER_PORT)

    if _port_open(WORKER_PORT):
        LOG.warning("PyWorker já ativo na porta %s; não vou duplicar", WORKER_PORT)
        _keep_alive()
        return

    worker = _resolve_worker()
    LOG.info("Iniciando PyWorker %s", worker)
    result = subprocess.run([sys.executable, "-u", str(worker)], check=False)
    if result.returncode != 0:
        _upload_crash_logs(f"worker-exit-{result.returncode}")
        sys.exit(result.returncode)
    _keep_alive()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("boot falhou")
        _upload_crash_logs("boot-exception")
        raise