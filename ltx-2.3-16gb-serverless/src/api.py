#!/usr/bin/env python3
"""FastAPI server for the LTX 2.3 GGUF 16 GB worker (execucao local via Docker).

Routes (contrato herdado do worker Vast, ../ltx-2.3-serverless-vast/src/api.py):
  POST /submit      - aceita o job e responde 202 com job_id (assincrono)
  POST /status      - estado do job ({job_id})
  GET  /health      - readiness da API e do ComfyUI

Camada de compatibilidade com o protocolo do proxy RunPod (backend
runpod-render.service): POST /{endpointId}/run, POST /{endpointId}/runsync,
GET /{endpointId}/status/{jobId}, GET /{endpointId}/health — ver
runpod_compat.py.

Diferencas vs worker Vast:
- auth opcional por Bearer token (env API_TOKEN); sem token, API aberta na
  rede local;
- /submit responde 503 enquanto o bootstrap dos modelos/ComfyUI nao termina.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from handler import (
    InputError,
    comfyui_ready,
    persist_job_status,
    run_job,
    startup_error,
    start_comfyui_async,
    validate_input,
    STARTUP_DONE,
)
from jobs import JobBusyError, JobManager, JobNotFoundError
from runpod_compat import (
    RUNSYNC_WAIT_SECONDS,
    deliver_runpod,
    runpod_output,
    runpod_request_to_vast,
    to_runpod_status,
)
from webhook import deliver

LOG = logging.getLogger("ltx23-16gb.api")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

manager = JobManager()

# project_id default para jobs do backend Runpod (que não envia project_id).
RUNPOD_DEFAULT_PROJECT_ID = (os.environ.get("RUNPOD_DEFAULT_PROJECT_ID") or "default").strip()


def _gate() -> None:
    if not STARTUP_DONE.is_set():
        raise HTTPException(status_code=503, detail="worker em inicialização (bootstrap dos modelos)")
    if startup_error():
        raise HTTPException(status_code=503, detail=f"worker indisponível: {startup_error()}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Bootstrap dos modelos + ComfyUI roda em paralelo; /health reflete o estado."""
    start_comfyui_async()
    yield


app = FastAPI(title="LTX 2.3 IA2V Personal LoRA — GGUF 16 GB worker", version="1.0", lifespan=lifespan)


def _require_auth(request: Request) -> None:
    token = (os.environ.get("API_TOKEN") or "").strip()
    if not token:
        return
    header = (request.headers.get("authorization") or "").strip()
    if header != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="token inválido")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "comfyui": comfyui_ready(),
        "startup_done": STARTUP_DONE.is_set(),
        "startup_error": startup_error(),
    }


def _execute(job_id: str) -> None:
    error: str | None = None
    record = None
    try:
        record = manager.mark_running(job_id)
        record.status_url = persist_job_status(record)
        result = run_job(record.params, job_id)
        record = manager.complete(job_id, result)
        LOG.info("Job %s concluído em %ss", job_id, result.get("execution_seconds"))
    except Exception as exc:  # noqa: BLE001 - webhook must still fire on failure
        error = str(exc)
        LOG.exception("Job %s falhou", job_id)
        try:
            record = manager.fail(job_id, error)
        except JobNotFoundError:
            return
    status_url = persist_job_status(record)
    record.status_url = record.status_url or status_url
    if record.params.get("_protocol") == "runpod":
        # Jobs submetidos via rotas compatíveis com o proxy RunPod recebem o
        # webhook no formato {"id", "status", "output", "error"}.
        deliver_runpod(record, error=error)
    else:
        deliver(
            record,
            "job.completed" if record.status == "completed" else "job.failed",
            error=error,
            status_url=record.status_url,
        )


@app.post("/submit")
def submit(payload: dict[str, Any], request: Request) -> JSONResponse:
    _require_auth(request)
    _gate()
    try:
        values = validate_input(payload)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if manager.busy():
        raise HTTPException(status_code=409, detail="worker ocupado com outro job")

    try:
        record = manager.submit(values)
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    record.status_url = persist_job_status(record)
    response_status = record.status
    threading.Thread(target=_execute, args=(record.job_id,), daemon=True).start()
    if os.environ.get("WEBHOOK_STARTED", "0") == "1":
        deliver(record, "job.started", status_url=record.status_url)
    LOG.info("Job %s submetido (projeto %s)", record.job_id, record.project_id)
    return JSONResponse(
        status_code=202,
        content={"job_id": record.job_id, "project_id": record.project_id, "status": response_status},
    )


@app.post("/status")
def status(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    _require_auth(request)
    job_id = (payload or {}).get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise HTTPException(status_code=422, detail="job_id é obrigatório")
    try:
        record = manager.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="job desconhecido") from exc
    return record.to_dict()


# ---------------------------------------------------------------------------
# Camada de compatibilidade com o protocolo do proxy RunPod serverless.
#
# O backend de produção (runpod-render.service) foi construído contra o worker
# Runpod (`ltx-2.3-serverless`) e fala este protocolo; as rotas abaixo traduzem
# para o contrato interno sem tocar nas rotas vast acima.
#
#   POST /{endpointId}/run           -> {"id", "status": "IN_QUEUE"}
#   POST /{endpointId}/runsync       -> espera até RUNSYNC_WAIT_SECONDS
#   GET  /{endpointId}/status/{id}   -> {"id", "status", "output", "error"}
#   GET  /{endpointId}/health        -> {"jobs": {...}}
# ---------------------------------------------------------------------------


def _runpod_status_payload(record) -> dict[str, Any]:
    content: dict[str, Any] = {"id": record.job_id, "status": to_runpod_status(record.status)}
    if record.status == "completed":
        content["output"] = runpod_output(record)
    elif record.status == "failed":
        content["output"] = None
        content["error"] = record.error
    return content


def _runpod_submit(payload: dict[str, Any], request: Request, endpoint_id: str):
    """Fluxo comum de submissão das rotas /run e /runsync; retorna o record."""
    _require_auth(request)
    _gate()
    try:
        body = runpod_request_to_vast(payload, default_project_id=RUNPOD_DEFAULT_PROJECT_ID)
        values = validate_input(body)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if manager.busy():
        raise HTTPException(status_code=409, detail="worker ocupado com outro job")

    values["_protocol"] = "runpod"
    try:
        record = manager.submit(values)
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    record.status_url = persist_job_status(record)
    threading.Thread(target=_execute, args=(record.job_id,), daemon=True).start()
    LOG.info("Job %s submetido via protocolo RunPod (endpoint %s)", record.job_id, endpoint_id)
    return record


@app.post("/{endpoint_id}/run")
def runpod_run(endpoint_id: str, payload: dict[str, Any], request: Request) -> JSONResponse:
    record = _runpod_submit(payload, request, endpoint_id)
    return JSONResponse(status_code=200, content={"id": record.job_id, "status": "IN_QUEUE"})


@app.post("/{endpoint_id}/runsync")
def runpod_runsync(endpoint_id: str, payload: dict[str, Any], request: Request) -> dict[str, Any]:
    record = _runpod_submit(payload, request, endpoint_id)
    deadline = time.monotonic() + RUNSYNC_WAIT_SECONDS
    while time.monotonic() < deadline:
        record = manager.get(record.job_id)
        if record.status in ("completed", "failed"):
            break
        time.sleep(0.25)
    return _runpod_status_payload(record)


@app.get("/{endpoint_id}/status/{job_id}")
def runpod_status(endpoint_id: str, job_id: str, request: Request) -> dict[str, Any]:
    _require_auth(request)
    try:
        record = manager.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return _runpod_status_payload(record)


@app.get("/{endpoint_id}/health")
def runpod_health(endpoint_id: str, request: Request) -> dict[str, Any]:
    _require_auth(request)
    with manager._lock:  # noqa: SLF001 - contagem apenas para o health do RunPod
        statuses = [record.status for record in manager._jobs.values()]
    return {
        "jobs": {
            "completed": statuses.count("completed"),
            "failed": statuses.count("failed"),
            "in_progress": statuses.count("running"),
            "queued": statuses.count("queued"),
        },
        "workers": {"ready": STARTUP_DONE.is_set() and comfyui_ready()},
    }
