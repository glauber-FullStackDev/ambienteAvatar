#!/usr/bin/env python3
"""Webhook delivery for finished jobs: retries, HMAC signature and allow-list."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

LOG = logging.getLogger("ltx23-vast.webhook")

RESERVED_KEYS = {
    "event",
    "job_id",
    "project_id",
    "status",
    "timestamp",
    "status_url",
    "parameters",
    "result",
    "error",
    "execution_seconds",
}

WEBHOOK_RETRIES = int(os.environ.get("WEBHOOK_RETRIES", "5"))
WEBHOOK_TIMEOUT = float(os.environ.get("WEBHOOK_TIMEOUT", "10"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_webhook_url() -> str:
    return (os.environ.get("WEBHOOK_URL") or "").strip()


def _allowed_hosts() -> set[str]:
    configured = (os.environ.get("WEBHOOK_ALLOWED_HOSTS") or "").strip()
    if configured:
        return {host.strip() for host in configured.split(",") if host.strip()}
    url = default_webhook_url()
    if url:
        parsed = urlparse(url)
        if parsed.netloc:
            return {parsed.netloc}
    return set()


def url_allowed(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"https", "http"} and bool(parsed.netloc) and parsed.netloc in _allowed_hosts()


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def build_payload(event: str, record: Any, *, error: str | None = None, status_url: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in (record.webhook_extra_params or {}).items():
        if key not in RESERVED_KEYS and isinstance(value, (str, int, float, bool, list, dict)):
            payload[key] = value
    payload.update(
        {
            "event": event,
            "job_id": record.job_id,
            "project_id": record.project_id,
            "status": record.status,
            "status_url": status_url,
            "timestamp": _now_iso(),
            "result": record.result,
            "error": error,
            "execution_seconds": round(record.finished_at - record.started_at, 3) if record.started_at and record.finished_at else None,
            "parameters": _public_parameters(record.params),
        }
    )
    return payload


def _public_parameters(params: dict[str, Any]) -> dict[str, Any]:
    keys = {"width", "height", "duration_seconds", "fps", "audio_start_seconds", "seed", "lora_strength", "image_strength"}
    return {key: params[key] for key in keys if key in params}


def deliver(
    record: Any,
    event: str,
    *,
    error: str | None = None,
    status_url: str | None = None,
) -> None:
    import requests  # lazy import keeps non-network modules importable without 3rd-party deps

    url = record.webhook_url or default_webhook_url()
    if not url:
        LOG.info("Nenhum WEBHOOK_URL configurado; notificação de %s omitida para %s", event, record.job_id)
        return
    if not url_allowed(url):
        LOG.error("Host do webhook %s bloqueado por WEBHOOK_ALLOWED_HOSTS", urlparse(url).netloc)
        return

    payload = build_payload(event, record, error=error, status_url=status_url)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    secret = (os.environ.get("WEBHOOK_SECRET") or "").strip()
    headers = {"Content-Type": "application/json", "User-Agent": "ltx23-vast-worker"}
    if secret:
        headers["X-Webhook-Signature"] = _sign(secret, body)

    delay = 1.0
    for attempt in range(1, WEBHOOK_RETRIES + 1):
        try:
            response = requests.post(url, data=body, headers=headers, timeout=WEBHOOK_TIMEOUT)
            if response.ok:
                LOG.info("Webhook %s enviado para %s (%s): %s", event, url, response.status_code, record.job_id)
                return
            LOG.warning("Webhook %s: HTTP %s na tentativa %s", event, response.status_code, attempt)
        except Exception as exc:  # noqa: BLE001 - retries below
            LOG.warning("Webhook %s: erro na tentativa %s: %s", event, attempt, exc)
        if attempt < WEBHOOK_RETRIES:
            time.sleep(delay)
            delay = min(delay * 2, 16.0)
    LOG.error("Webhook %s falhou após %s tentativas para %s", event, WEBHOOK_RETRIES, record.job_id)