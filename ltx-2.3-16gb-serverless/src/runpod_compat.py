#!/usr/bin/env python3
"""Helpers de compatibilidade com o protocolo do proxy RunPod serverless.

O backend de produção (runpod-render.service) foi construído contra o worker
Runpod (`ltx-2.3-serverless`) e fala este protocolo:

  Submeter:  POST /{endpointId}/run   body: {"input": {...}, "webhook": "https://..."}
  Resposta:  {"id": "<jobId>", "status": "IN_QUEUE"}                  (HTTP 200)
  Status:    GET  /{endpointId}/status/{jobId}
             -> {"id", "status": IN_QUEUE|IN_PROGRESS|COMPLETED|FAILED,
                 "output": <retorno do handler>, "error": ...}

O webhook delivery reproduz o formato do proxy RunPod
({"id", "status", "output", "error"}) e o campo `output` replica o dict que o
worker `ltx-2.3-serverless` devolvia — zero mudança no backend.

As rotas do contrato vast (/submit, /status) continuam disponíveis em
paralelo; este módulo é apenas um adaptador de transporte. As rotas HTTP
ficam em api.py; aqui ficam só as funções puras de tradução e a entrega.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

from handler import InputError
from webhook import WEBHOOK_RETRIES, WEBHOOK_TIMEOUT, default_webhook_url, url_allowed

LOG = logging.getLogger("ltx23-16gb.runpod")

# Status usados pelo proxy RunPod; o worker interno usa queued/running/...
RUNPOD_STATUS = {
    "queued": "IN_QUEUE",
    "running": "IN_PROGRESS",
    "completed": "COMPLETED",
    "failed": "FAILED",
}

# /runsync responde bloqueando até este limite (abaixo do timeout proxied do
# Cloudflare, ~100s); depois disso responde IN_PROGRESS como o RunPod.
RUNSYNC_WAIT_SECONDS = float(os.environ.get("RUNSYNC_WAIT_SECONDS", "85"))

# Chaves exatas do dict `parameters` retornado pelo worker `ltx-2.3-serverless`.
RUNPOD_OUTPUT_PARAMETER_KEYS = (
    "prompt",
    "negative_prompt",
    "duration_seconds",
    "fps",
    "audio_start_seconds",
    "lora_strength",
    "first_frame_strength",
    "cfg",
    "decode_tile_size",
    "enable_prompt_enhance",
    "base_sigmas",
    "refine_sigmas",
)


def to_runpod_status(internal: str) -> str:
    return RUNPOD_STATUS.get(internal, "IN_QUEUE")


def runpod_request_to_vast(payload: Any, *, default_project_id: str = "default") -> dict[str, Any]:
    """Traduz o envelope RunPod {"input": {...}, "webhook": "url"} para o validador.

    O backend Runpod envia `webhook` como string no topo do corpo e não envia
    `project_id`; aqui o webhook vira {"url": ...} e o project_id recebe o
    default configurável (env RUNPOD_DEFAULT_PROJECT_ID).
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("input"), dict):
        raise InputError('corpo precisa ser {"input": {...}}')
    input_obj = dict(payload["input"])
    if not input_obj.get("project_id"):
        input_obj["project_id"] = default_project_id
    webhook = payload.get("webhook")
    if isinstance(webhook, str) and webhook.strip() and "webhook" not in input_obj:
        # validate_input lê o webhook de DENTRO de input (o envelope RunPod o
        # traz no topo do corpo); aqui ele é movido para o lugar certo.
        input_obj["webhook"] = {"url": webhook.strip()}
    elif isinstance(webhook, dict) and "webhook" not in input_obj:
        input_obj["webhook"] = webhook
    return {"input": input_obj}


def runpod_output(record: Any) -> dict[str, Any] | None:
    """Replica o dict de saída do worker `ltx-2.3-serverless` (contrato Runpod)."""
    if record.result is None:
        return None
    params = record.params
    return {
        "job_id": record.job_id,
        "comfy_prompt_id": record.result.get("comfy_prompt_id"),
        "video_url": record.result.get("video_url"),
        "last_frame_url": record.result.get("last_frame_url"),
        "image_sha": record.result.get("image_sha"),
        "parameters": {
            key: params[key] for key in RUNPOD_OUTPUT_PARAMETER_KEYS if key in params
        },
        "execution_seconds": record.result.get("execution_seconds"),
    }


def build_runpod_webhook(record: Any, *, error: str | None = None) -> dict[str, Any]:
    """Corpo no formato que o proxy RunPod entrega ao webhook do backend."""
    return {
        "id": record.job_id,
        "status": "COMPLETED" if record.status == "completed" else "FAILED",
        "output": record.result if record.status == "completed" else None,
        "error": error,
    }


def deliver_runpod(record: Any, *, error: str | None = None) -> None:
    """Entrega ao webhook do backend no formato do proxy RunPod.

    Mesma mecânica de webhook.deliver (retries/backoff/allowlist/HMAC), porém
    com o corpo {"id", "status", "output", "error"} que o backend espera.
    """
    import requests  # lazy import keeps non-network modules importable without 3rd-party deps

    url = record.webhook_url or default_webhook_url()
    if not url:
        LOG.info("Nenhum webhook configurado; entrega RunPod omitida para %s", record.job_id)
        return
    if not url_allowed(url):
        LOG.error("Host do webhook %s bloqueado por WEBHOOK_ALLOWED_HOSTS", url)
        return

    payload = build_runpod_webhook(record, error=error)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    secret = (os.environ.get("WEBHOOK_SECRET") or "").strip()
    headers = {"Content-Type": "application/json", "User-Agent": "ltx23-16gb-worker"}
    if secret:
        headers["X-Webhook-Signature"] = "sha256=" + hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()

    delay = 1.0
    for attempt in range(1, WEBHOOK_RETRIES + 1):
        try:
            response = requests.post(url, data=body, headers=headers, timeout=WEBHOOK_TIMEOUT)
            if response.ok:
                LOG.info(
                    "Webhook RunPod %s enviado para %s (%s)",
                    payload["status"], url, record.job_id,
                )
                return
            LOG.warning(
                "Webhook RunPod %s: HTTP %s na tentativa %s",
                payload["status"], response.status_code, attempt,
            )
        except Exception as exc:  # noqa: BLE001 - retries abaixo
            LOG.warning("Webhook RunPod: erro na tentativa %s: %s", attempt, exc)
        if attempt < WEBHOOK_RETRIES:
            time.sleep(delay)
            delay = min(delay * 2, 16.0)
    LOG.error(
        "Webhook RunPod falhou após %s tentativas para %s (fallback: GET /status/%s)",
        WEBHOOK_RETRIES, record.job_id, record.job_id,
    )
