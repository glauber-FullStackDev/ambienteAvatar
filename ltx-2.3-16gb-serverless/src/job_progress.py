#!/usr/bin/env python3
"""Progresso por job: fases + percentual derivado do WebSocket do ComfyUI.

O ComfyUI emite eventos `progress` (value/max por node) e `executing` via
`/ws?clientId=<id>`.  O listener conecta usando o mesmo client_id usado na
submissão do workflow (o job_id) e traduz os eventos num percentual com pesos
por fase:

  download do input    0-3    (checkpoint no run_job)
  fila do ComfyUI      3-5    (checkpoint)
  sampling             5-80   (SamplerCustomAdvanced, base + refine)
  VAEDecodeTiled       80-97  (node 316)
  upload MinIO         97-99  (checkpoint)
  concluído            100

Estado fica em memória, thread-safe, por job_id — sem tocar em jobs.py
(paridade byte-a-byte com o worker Vast).  Se o WebSocket falhar, o job
continua normalmente; o painel mostra apenas a fase por checkpoints.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

LOG = logging.getLogger("ltx23-16gb.progress")

# node id do VAEDecodeTiled no workflow GGUF 16 GB (ver workflow_api.py)
DECODE_NODE_ID = "316"

PHASE_RANGES: dict[str, tuple[float, float]] = {
    "downloading": (0.0, 3.0),
    "queueing": (3.0, 5.0),
    "sampling": (5.0, 80.0),
    "decoding": (80.0, 97.0),
    "uploading": (97.0, 99.0),
    "done": (100.0, 100.0),
}


def percent_for_phase(phase: str, ratio: float | None = None) -> float:
    """Mapeia (fase, razão do passo) para um percentual global do job."""
    lo, hi = PHASE_RANGES.get(phase, (0.0, 0.0))
    if ratio is None:
        return lo
    clamped = max(0.0, min(1.0, ratio))
    return round(lo + (hi - lo) * clamped, 1)


def _phase_for_node(node: str) -> str:
    return "decoding" if node == DECODE_NODE_ID else "sampling"


class ProgressTracker:
    """Estado de progresso em memória, thread-safe, por job_id."""

    def __init__(self, max_entries: int = 50) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._stops: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._max_entries = max_entries

    # -- ciclo de vida ------------------------------------------------------

    def start(self, job_id: str, project_id: str = "", prompt: str = "") -> None:
        with self._lock:
            stale = self._stops.pop(job_id, None)
            self._jobs[job_id] = {
                "phase": "downloading",
                "percent": 0.0,
                "step": None,
                "total_steps": None,
                "project_id": project_id,
                "prompt": prompt,
                "started_at": time.time(),
                "updated_at": time.time(),
                "finished_at": None,
            }
            self._prune_locked()
        if stale:
            stale.set()

    def attach_comfy(self, job_id: str) -> None:
        """Inicia o listener do WebSocket do ComfyUI para o job (best effort)."""
        stop = threading.Event()
        thread = threading.Thread(target=self._listen, args=(job_id, stop), daemon=True)
        with self._lock:
            self._stops[job_id] = stop
            self._threads[job_id] = thread
        thread.start()

    def finish(self, job_id: str, ok: bool = True) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            stop = self._stops.pop(job_id, None)
            thread = self._threads.pop(job_id, None)
        if stop:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)
        with self._lock:
            if not state:
                return
            state["phase"] = "done" if ok else "failed"
            if ok:
                state["percent"] = 100.0
            state["step"] = None
            state["total_steps"] = None
            state["finished_at"] = time.time()
            state["updated_at"] = time.time()

    # -- checkpoints das fases (sem eventos WS) ------------------------------

    def set_phase(self, job_id: str, phase: str) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return
            state["phase"] = phase
            state["percent"] = max(state["percent"], percent_for_phase(phase))
            state["step"] = None
            state["total_steps"] = None
            state["updated_at"] = time.time()

    # -- eventos do ComfyUI --------------------------------------------------

    def handle_event(self, job_id: str, event: Any) -> None:
        if not isinstance(event, dict):
            return
        etype = event.get("type")
        data = event.get("data") or {}
        if etype == "progress":
            maximum = data.get("max")
            if not isinstance(maximum, (int, float)) or maximum <= 0:
                return
            value = data.get("value") or 0
            ratio = value / maximum
            node = str(data.get("node") or "")
            self._record(
                job_id,
                _phase_for_node(node),
                ratio,
                step=value,
                total=maximum,
            )

    def _record(
        self, job_id: str, phase: str, ratio: float, step: Any = None, total: Any = None
    ) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return
            state["phase"] = phase
            percent = percent_for_phase(phase, ratio)
            # monotônico: o sampler refine recomeça do passo 1 e não deve
            # derrubar a barra; dentro da mesma fase vale o máximo visto.
            if percent >= state["percent"]:
                state["percent"] = percent
            state["step"] = step
            state["total_steps"] = total
            state["updated_at"] = time.time()

    # -- consulta -------------------------------------------------------------

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._jobs.get(job_id)
            return dict(state) if state else None

    def drop(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            self._stops.pop(job_id, None)
            self._threads.pop(job_id, None)

    # -- internals ------------------------------------------------------------

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self._max_entries:
            return
        oldest = sorted(self._jobs.items(), key=lambda kv: kv[1].get("started_at") or 0.0)
        for job_id, _ in oldest[: len(self._jobs) - self._max_entries]:
            self._jobs.pop(job_id, None)

    def _listen(self, job_id: str, stop: threading.Event) -> None:
        try:
            import websocket
        except ImportError:  # pragma: no cover - dependência vem do requirements.txt
            LOG.warning("websocket-client não instalado; progresso fino desativado")
            return
        port = os.environ.get("COMFYUI_PORT", "8188")
        url = f"ws://127.0.0.1:{port}/ws?clientId={job_id}"
        try:
            ws = websocket.create_connection(url, timeout=1.0)
        except Exception as exc:  # noqa: BLE001 - progresso é best-effort
            LOG.warning("WebSocket do ComfyUI indisponível para %s: %s", job_id, exc)
            return
        LOG.info("Listener de progresso conectado para %s", job_id)
        try:
            while not stop.is_set():
                try:
                    message = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception:  # noqa: BLE001 - conexão caiu; best-effort
                    break
                if isinstance(message, bytes):
                    continue  # frames binários de preview
                try:
                    event = json.loads(message)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    self.handle_event(job_id, event)
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass


tracker = ProgressTracker()
