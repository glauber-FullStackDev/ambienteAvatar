#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


SCRIPTS_HOME = Path(__file__).resolve().parent


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_downloader(verify_only: bool = False) -> int:
    command = [sys.executable, str(SCRIPTS_HOME / "download_models.py")]
    if verify_only:
        command.append("--verify-only")
    return subprocess.call(command)


def serve(extra_args: list[str]) -> None:
    Path(os.environ.get("MODEL_ROOT", "/models")).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get("OUTPUT_DIR", "/outputs")).mkdir(parents=True, exist_ok=True)
    if env_flag("DOWNLOAD_MODELS_ON_START"):
        result = run_downloader()
        if result != 0:
            raise SystemExit(result)
    app = Path("/opt/wan-animate-2/app.py")
    os.execvp(sys.executable, [sys.executable, str(app), *extra_args])


def main() -> None:
    action, *extra_args = sys.argv[1:] or ["serve"]
    if action == "serve":
        serve(extra_args)
    if action == "download-models":
        raise SystemExit(run_downloader())
    if action == "verify":
        raise SystemExit(run_downloader(verify_only=True))
    os.execvp(action, [action, *extra_args])


if __name__ == "__main__":
    main()
