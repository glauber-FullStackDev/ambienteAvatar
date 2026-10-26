#!/usr/bin/env python3
"""Job state manager kept in memory on the model server (single GPU worker)."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any


class JobBusyError(RuntimeError):
    pass


class JobNotFoundError(KeyError):
    pass


@dataclass
class JobRecord:
    job_id: str
    project_id: str
    status: str
    params: dict[str, Any]
    webhook_url: str = ""
    webhook_extra_params: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    status_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "status_url": self.status_url,
        }


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._running = False

    def submit(self, values: dict[str, Any]) -> JobRecord:
        with self._lock:
            if self._running:
                raise JobBusyError("worker ocupado com outro job")
            record = JobRecord(
                job_id=values["job_id"],
                project_id=values["project_id"],
                status="queued",
                params=values,
                webhook_url=values.get("webhook_url", ""),
                webhook_extra_params=values.get("webhook_extra_params", {}),
            )
            self._jobs[record.job_id] = record
            self._running = True
        return record

    def mark_running(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._get_locked(job_id)
            record.status = "running"
            record.started_at = time.time()
            self._running = True
        return record

    def complete(self, job_id: str, result: dict[str, Any]) -> JobRecord:
        with self._lock:
            record = self._get_locked(job_id)
            record.status = "completed"
            record.result = result
            record.finished_at = time.time()
            self._running = False
        return record

    def fail(self, job_id: str, error: str) -> JobRecord:
        with self._lock:
            record = self._get_locked(job_id)
            record.status = "failed"
            record.error = error
            record.finished_at = time.time()
            self._running = False
        return record

    def set_status_url(self, job_id: str, status_url: str | None) -> None:
        with self._lock:
            record = self._get_locked(job_id)
            record.status_url = status_url

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            return self._jobs[job_id]

    def busy(self) -> bool:
        with self._lock:
            return self._running

    def _get_locked(self, job_id: str) -> JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobNotFoundError(job_id) from None