#!/usr/bin/env python3
"""FastAPI server for the LTX 2.3 GGUF 16 GB worker (execucao local via Docker).

Routes (contrato herdado do worker Vast, ../ltx-2.3-serverless-vast/src/api.py):
  POST /submit      - aceita o job e responde 202 com job_id (assincrono)
  POST /status      - estado do job ({job_id})
  GET  /health      - readiness da API e do ComfyUI

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
from webhook import deliver

LOG = logging.getLogger("ltx23-16gb.api")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

manager = JobManager()


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
    deliver(
        record,
        "job.completed" if record.status == "completed" else "job.failed",
        error=error,
        status_url=record.status_url,
    )


@app.post("/submit")
def submit(payload: dict[str, Any], request: Request) -> JSONResponse:
    _require_auth(request)
    if not STARTUP_DONE.is_set():
        raise HTTPException(status_code=503, detail="worker em inicialização (bootstrap dos modelos)")
    if startup_error():
        raise HTTPException(status_code=503, detail=f"worker indisponível: {startup_error()}")
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
