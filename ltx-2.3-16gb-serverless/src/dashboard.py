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
        records = sorted(manager._jobs.values(), key=lambda r: r.created_at, reverse=True)

    current: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for record in records:
        if record.status in ("queued", "running") and current is None:
            progress = tracker.snapshot(record.job_id) or {
                "phase": "queued",
                "percent": 0.0,
                "step": None,
                "total_steps": None,
            }
            current = _entry(record, progress)
        elif record.status in ("completed", "failed"):
            history.append(_entry(record))

    return JSONResponse(
        {
            "worker": {
                "startup_done": STARTUP_DONE.is_set(),
                "comfyui": comfyui_ready(),
                "startup_error": startup_error(),
                "busy": manager.busy(),
            },
            "current": current,
            "history": history[:20],
        }
    )
