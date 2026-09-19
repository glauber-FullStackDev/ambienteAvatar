#!/usr/bin/env python3
"""Validate and normalize a job request for the LTX 2.3 IA2V Vast worker."""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

DEFAULTS = {
    "width": 704,
    "height": 1280,
    "duration_seconds": 18.0,
    "fps": 24,
    "audio_start_seconds": 0.0,
    "lora_strength": 1.0,
    "image_strength": 0.7,
}


class InputError(ValueError):
    pass


def _number(value: Any, name: str, *, minimum: float, maximum: float, integer: bool = False) -> int | float:
    if isinstance(value, bool):
        raise InputError(f"{name} precisa ser numérico")
    try:
        converted = int(value) if integer else float(value)
    except (TypeError, ValueError) as error:
        raise InputError(f"{name} precisa ser numérico") from error
    if not minimum <= converted <= maximum:
        raise InputError(f"{name} precisa estar entre {minimum} e {maximum}")
    return converted


def _identifier(value: Any, name: str, *, required: bool) -> str:
    if value is None:
        if required:
            raise InputError(f"{name} é obrigatório")
        return ""
    if not isinstance(value, str):
        raise InputError(f"{name} precisa ser texto")
    value = value.strip()
    if not value:
        if required:
            raise InputError(f"{name} é obrigatório")
        return ""
    if len(value) > 128:
        raise InputError(f"{name} excede 128 caracteres")
    if not all(char.isalnum() or char in "-._" for char in value):
        raise InputError(f"{name} só aceita letras, números, '-', '_' e '.'")
    return value


def _optional_text(value: Any, name: str, *, maximum: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InputError(f"{name} precisa ser texto")
    value = value.strip()
    if len(value) > maximum:
        raise InputError(f"{name} excede {maximum} caracteres")
    return value


def validate_input(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InputError("input precisa ser um objeto JSON")
    if "input" in raw and isinstance(raw["input"], dict):
        raw = raw["input"]

    project_id = _identifier(raw.get("project_id"), "project_id", required=True)
    job_id = _identifier(raw.get("job_id") or str(uuid4().hex), "job_id", required=True)

    image_url = raw.get("image_url")
    audio_url = raw.get("audio_url")
    prompt = raw.get("prompt")
    if not all(isinstance(value, str) and value.strip() for value in (image_url, audio_url, prompt)):
        raise InputError("image_url, audio_url e prompt são obrigatórios")
    if len(prompt) > 12_000:
        raise InputError("prompt excede 12000 caracteres")

    webhook_raw = raw.get("webhook") or {}
    if not isinstance(webhook_raw, dict):
        raise InputError("webhook precisa ser um objeto")
    webhook_url = _optional_text(webhook_raw.get("url"), "webhook.url", maximum=2048)
    if webhook_url:
        parsed = urlparse(webhook_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise InputError("webhook.url precisa ser um endereço HTTP(S) completo")
    extra_params = webhook_raw.get("extra_params") or {}
    if not isinstance(extra_params, dict):
        raise InputError("webhook.extra_params precisa ser um objeto")

    width = _number(raw.get("width", DEFAULTS["width"]), "width", minimum=256, maximum=1920, integer=True)
    height = _number(raw.get("height", DEFAULTS["height"]), "height", minimum=256, maximum=1920, integer=True)
    if width % 32 or height % 32:
        raise InputError("width e height precisam ser divisíveis por 32")
    seed = raw.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") % (2**63 - 1)
    else:
        seed = _number(seed, "seed", minimum=0, maximum=2**63 - 1, integer=True)

    return {
        "project_id": project_id,
        "job_id": job_id,
        "image_url": image_url,
        "audio_url": audio_url,
        "prompt": prompt.strip(),
        "width": width,
        "height": height,
        "duration_seconds": _number(raw.get("duration_seconds", DEFAULTS["duration_seconds"]), "duration_seconds", minimum=1, maximum=30),
        "fps": _number(raw.get("fps", DEFAULTS["fps"]), "fps", minimum=8, maximum=30, integer=True),
        "audio_start_seconds": _number(raw.get("audio_start_seconds", DEFAULTS["audio_start_seconds"]), "audio_start_seconds", minimum=0, maximum=3600),
        "seed": seed,
        "lora_strength": _number(raw.get("lora_strength", DEFAULTS["lora_strength"]), "lora_strength", minimum=0, maximum=2),
        "image_strength": _number(raw.get("image_strength", DEFAULTS["image_strength"]), "image_strength", minimum=0, maximum=1),
        "webhook_url": webhook_url,
        "webhook_extra_params": dict(extra_params),
    }