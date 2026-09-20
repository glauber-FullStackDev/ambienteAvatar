#!/usr/bin/env python3
"""Vast.ai PyWorker exposing the LTX 2.3 IA2V model server.

This is the canonical worker.py for the dedicated PYWORKER_REPO repository
(`ltx23-vast-pyworker`). It wires the FastAPI model server into the Vast
serverless engine: readiness via model log, benchmarking, and cost-per-request
workload accounting. The same file is mirrored in the image for local tests.
"""
from __future__ import annotations

import os

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1")
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "18080"))
MODEL_LOG_FILE = os.environ.get("MODEL_LOG_FILE", "/var/log/model/model.log")
BENCHMARK_RUNS = int(os.environ.get("BENCHMARK_RUNS", "1"))
BENCHMARK_CONCURRENCY = int(os.environ.get("BENCHMARK_CONCURRENCY", "1"))


def _number(payload: dict, name: str, default: float) -> float:
    value = payload.get(name)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def workload(payload: dict) -> float:
    """Cost proxy: pixel-frames generated. 1 Megapixel-frame ~= 1.0 workload."""
    width = _number(payload, "width", 704)
    height = _number(payload, "height", 1280)
    fps = _number(payload, "fps", 24)
    duration = _number(payload, "duration_seconds", 18)
    return max(1.0, duration * fps * width * height / 1_000_000.0)


def benchmark_payload() -> dict:
    return {
        "project_id": "benchmark",
        "image_url": "",
        "audio_url": "",
        "prompt": "benchmark",
        "width": 256,
        "height": 256,
        "fps": 8,
        "duration_seconds": 1,
    }


def benchmark_workload(payload: dict) -> float:
    return 1.0


config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url="/health",
    # Client dispatches create short-lived sessions; with max_workers=1 the
    # default cap of 10 exhausts after a few renders, so lift the limit here
    # (0 = unlimited per vastai SDK) and let idle workers cool down instead.
    max_sessions=0,
    handlers=[
        HandlerConfig(
            route="/submit",
            workload_calculator=workload,
        ),
        HandlerConfig(route="/status"),
        HandlerConfig(route="/health"),
        HandlerConfig(
            route="/benchmark",
            workload_calculator=benchmark_workload,
            benchmark_config=BenchmarkConfig(
                generator=benchmark_payload,
                runs=BENCHMARK_RUNS,
                concurrency=BENCHMARK_CONCURRENCY,
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["Model server ready"],
        on_error=[
            "Traceback (most recent call last):",
            "Model server failed",
        ],
    ),
)

if __name__ == "__main__":
    Worker(config).run()