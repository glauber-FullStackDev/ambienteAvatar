#!/usr/bin/env python3
"""FastAPI model server exposed through the Vast PyWorker.

Routes:
  POST /submit      - accept a job, return 202 with job_id (async)
  POST /status      - job state for session-based polling ({job_id})
  GET  /health      - PyWorker readiness check
  GET  /benchmark   - cheap route used by PyWorker benchmarking
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from engine import persist_job_status, run_job
from jobs import JobBusyError, JobManager, JobNotFoundError
from validate import InputError, validate_input
from webhook import deliver

LOG = logging.getLogger("ltx23-vast.api")

manager = JobManager()


app = FastAPI(title="LTX 2.3 IA2V Personal LoRA — Vast model server", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/benchmark")
@app.post("/benchmark")
def benchmark() -> dict[str, str]:
    return {"status": "ok", "benchmark": True}


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
def submit(payload: dict[str, Any]) -> JSONResponse:
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
def status(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = (payload or {}).get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise HTTPException(status_code=422, detail="job_id é obrigatório")
    try:
        record = manager.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="job desconhecido") from exc
    return record.to_dict()