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


def _entry(record: Any, progress: dict[str, Any] | None = None) -> dict[str, Any]:
    result = record.result or {}
    entry: dict[str, Any] = {
        "job_id": record.job_id,
        "project_id": record.project_id,
        "status": record.status,
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
    return entry


@router.get("/dashboard/api/summary", include_in_schema=False)
def dashboard_summary(request: Request) -> JSONResponse:
    from api import _require_auth, manager  # lazy: evita import circular no boot
    from handler import STARTUP_DONE, comfyui_ready, startup_error

    _require_auth(request, allow_query=True)

    with manager._lock:  # noqa: SLF001 - leitura apenas para o painel
        records = list(manager._jobs.values())

    ordered = sorted(records, key=lambda r: r.created_at)  # ordem da fila FIFO
    running = [r for r in ordered if r.status == "running"]
    queued = [r for r in ordered if r.status == "queued"]
    finished = [r for r in ordered if r.status in ("completed", "failed")]

    # Job atual: o em execução; sem nenhum, o próximo da fila (mostra o
    # progresso "Na fila" enquanto aguarda).  Os demais vão para `queue`.
    current_record = running[0] if running else (queued[0] if queued else None)
    pending = queued if running else queued[1:]

    current: dict[str, Any] | None = None
    if current_record is not None:
        progress = tracker.snapshot(current_record.job_id) or {
            "phase": "queued",
            "percent": 0.0,
            "step": None,
            "total_steps": None,
        }
        current = _entry(current_record, progress)

    return JSONResponse(
        {
            "worker": {
                "startup_done": STARTUP_DONE.is_set(),
                "comfyui": comfyui_ready(),
                "startup_error": startup_error(),
                "busy": manager.busy(),
                "queue_size": len(pending),
            },
            "current": current,
            "queue": [_entry(record) for record in pending],
            "history": [_entry(record) for record in reversed(finished)][:20],
        }
    )
