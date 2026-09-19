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
WORKER_PORT = int(os.environ.get("WORKER_PORT", "3000"))
MODEL_LOG_FILE = Path(os.environ.get("MODEL_LOG_FILE", "/var/log/model/model.log"))
PYWORKER_REPO = os.environ.get("PYWORKER_REPO", "").strip()
PYWORKER_REF = os.environ.get("PYWORKER_REF", "").strip()
PYWORKER_MIRROR = Path(os.environ.get("PYWORKER_SRC", "/opt/serverless-vast/pyworker"))
PYWORKER_CLONE = Path("/opt/pyworker/repo")


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


def _keep_alive() -> None:
    while True:
        time.sleep(3600)


def main() -> None:
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
    subprocess.run([sys.executable, "-u", str(worker)], check=False)
    _keep_alive()


if __name__ == "__main__":
    main()