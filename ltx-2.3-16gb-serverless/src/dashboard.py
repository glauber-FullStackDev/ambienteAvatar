#!/usr/bin/env python3
"""Painel de acompanhamento servido pelo próprio worker.

  GET /dashboard             - página HTML (estática, sem dados)
  GET /dashboard/api/summary - JSON com job atual (progresso) e histórico

O resumo mostra o job em execução (fase + % derivado do WebSocket do ComfyUI,
ver job_progress.py) e o histórico dos gerados com tempo de geração.  As rotas
de dados são protegidas por API_TOKEN (Bearer ou ?token=); a página em si é
aberta porque não expõe nada.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from job_progress import tracker

router = APIRouter()
HTML_PATH = Path(__file__).with_name("dashboard.html")


@router.get("/dashboard", include_in_schema=False)
def dashboard_page() -> HTMLResponse:
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


def _entry(record: Any, progress: dict[str, Any] | None = None, queue_position: int | None = None) -> dict[str, Any]:
    result = record.result or {}
    # Cancelamento (job_queue.cancel): status "completed" com output nulo.
    cancelled = record.status == "completed" and record.result is None and record.error is None
    entry: dict[str, Any] = {
        "job_id": record.job_id,
        "project_id": record.project_id,
        "status": record.status,
        "cancelled": cancelled,
        "prompt": (record.params or {}).get("prompt"),
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "duration_seconds": (
            round(record.finished_at - record.started_at, 1)
            if record.started_at and record.finished_at
            else None
        ),
        "video_url": result.get("video_url"),
        "last_frame_url": result.get("last_frame_url"),
        "error": record.error,
    }
    if progress is not None:
        entry["progress"] = progress
    if queue_position is not None:
        entry["queue_position"] = queue_position
    return entry


@router.get("/dashboard/api/summary", include_in_schema=False)
def dashboard_summary(request: Request) -> JSONResponse:
    from api import _require_auth, manager  # lazy: evita import circular no boot
    from handler import STARTUP_DONE, comfyui_ready, startup_error

    _require_auth(request, allow_query=True)

    with manager._lock:  # noqa: SLF001 - leitura apenas para o painel
        records = list(manager._jobs.values())
        queue_order = list(getattr(manager, "_queue", []))  # ordem FIFO real

    positions = {job_id: index + 1 for index, job_id in enumerate(queue_order)}

    ordered = sorted(records, key=lambda r: r.created_at, reverse=True)  # último primeiro
    current_records = [r for r in ordered if r.status == "running"]
    current_record = current_records[0] if current_records else None
    queued_records = sorted(
        (r for r in ordered if r.status == "queued" and r.job_id in positions),
        key=lambda r: positions[r.job_id],
    )
    finished_records = [r for r in ordered if r.status in ("completed", "failed")]

    current: dict[str, Any] | None = None
    if current_record is not None:
        progress = tracker.snapshot(current_record.job_id) or {
            "phase": "queued",
            "percent": 0.0,
            "step": None,
            "total_steps": None,
        }
        current = _entry(current_record, progress)

    cancelled_count = sum(
        1 for r in finished_records
        if r.status == "completed" and r.result is None and r.error is None
    )
    counts = {
        "queued": len(queued_records),
        "running": len(current_records),
        "completed": sum(1 for r in finished_records if r.status == "completed") - cancelled_count,
        "cancelled": cancelled_count,
        "failed": sum(1 for r in finished_records if r.status == "failed"),
    }

    return JSONResponse(
        {
            "worker": {
                "startup_done": STARTUP_DONE.is_set(),
                "comfyui": comfyui_ready(),
                "startup_error": startup_error(),
                "busy": manager.busy(),
                "queue_size": len(queued_records),
            },
            "current": current,
            "queue": [
                _entry(record, queue_position=positions[record.job_id])
                for record in queued_records
            ],
            "history": [_entry(record) for record in finished_records][:20],
            "counts": counts,
        }
    )
