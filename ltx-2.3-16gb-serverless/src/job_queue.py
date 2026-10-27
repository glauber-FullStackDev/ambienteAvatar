#!/usr/bin/env python3
"""Fila FIFO sobre o JobManager para as rotas compatíveis com o RunPod.

`jobs.py` permanece byte-idêntico ao do worker Vast (garantia em
tests/test_serverless_contract.py::test_jobs_module_matches_vast); a fila vive
aqui, como extensão, sem tocar no contrato vast.

Comportamento igual ao proxy RunPod serverless:
- enqueue: com a GPU ocupada o job entra na fila (antes: 409) e responde
  IN_QUEUE; o backend pode mandar vários vídeos seguidos.
- next_queued: o dispatcher da API bloqueia aqui e recebe o próximo job em
  ordem de chegada quando o worker libera (um job por vez).
- cancel: remove da fila um job ainda não iniciado (jobs em execução não
  são canceláveis, como no RunPod).
"""
from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any

from jobs import JobManager, JobRecord


class JobQueueManager(JobManager):
    """JobManager + fila FIFO (um job por vez, ordem de chegada)."""

    def __init__(self) -> None:
        super().__init__()
        # A Condition compartilha o MESMO lock do JobManager: os métodos do
        # pai (submit/mark_running/complete/fail) continuam atômicos em
        # relação à fila.
        self._cond = threading.Condition(self._lock)
        self._queue: deque[str] = deque()

    # -- fila (rotas compatíveis com o RunPod) -----------------------------

    def enqueue(self, values: dict[str, Any]) -> JobRecord:
        """Aceita o job como `queued` mesmo com a GPU ocupada (fila RunPod)."""
        with self._cond:
            record = JobRecord(
                job_id=values["job_id"],
                project_id=values["project_id"],
                status="queued",
                params=values,
                webhook_url=values.get("webhook_url", ""),
                webhook_extra_params=values.get("webhook_extra_params", {}),
            )
            self._jobs[record.job_id] = record
            self._queue.append(record.job_id)
            self._cond.notify_all()
        return record

    def queue_size(self) -> int:
        """Jobs aguardando na fila (não conta o job em execução)."""
        with self._cond:
            return len(self._queue)

    def next_queued(self, timeout: float | None = None) -> str | None:
        """Dispatcher: espera o worker liberar e devolve o próximo job (FIFO).

        Bloqueia até existir job na fila com a GPU livre; já transiciona o
        record para `running` (atomicamente, sob lock).  Com `timeout`,
        devolve None se nada aparecer — o dispatcher reavalia o manager
        global (facilita os testes, que trocam o manager).
        """
        with self._cond:
            if timeout is None:
                while not self._queue or self._running:
                    self._cond.wait()
            else:
                deadline = time.monotonic() + timeout
                while not self._queue or self._running:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
            job_id = self._queue.popleft()
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = time.time()
            self._running = True
            return job_id

    def cancel(self, job_id: str) -> tuple[JobRecord, bool]:
        """Cancela um job ainda na fila.

        Igual ao RunPod: o job cancelado fica `completed` com output nulo
        (nunca mais aparece como IN_QUEUE).  Jobs em execução ou já
        finalizados não mudam.  Retorna (record, cancelado).
        """
        with self._cond:
            record = self._get_locked(job_id)  # JobNotFoundError se desconhecido
            if record.status != "queued" or job_id not in self._queue:
                return record, False
            self._queue.remove(job_id)
            record.status = "completed"
            record.result = None
            record.error = None
            record.finished_at = time.time()
            return record, True

    # -- hooks que acordam o dispatcher ------------------------------------

    def mark_running(self, job_id: str) -> JobRecord:
        # Igual ao pai, mas idempotente: o dispatcher já marcou started_at
        # ao tirar o job da fila; não resetar o relógio.
        with self._cond:
            record = self._get_locked(job_id)
            record.status = "running"
            if record.started_at is None:
                record.started_at = time.time()
            self._running = True
        return record

    def complete(self, job_id: str, result: dict[str, Any]) -> JobRecord:
        record = super().complete(job_id, result)
        with self._cond:
            self._cond.notify_all()
        return record

    def fail(self, job_id: str, error: str) -> JobRecord:
        record = super().fail(job_id, error)
        with self._cond:
            self._cond.notify_all()
        return record
