#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


# PyPI only publishes Decord 0.6.0 with old wheel metadata. It imports and runs
# on the linux/amd64 CUDA image, but `pip check` reports the metadata warning as
# a failure. The custom-node smoke test imports Decord separately.
IGNORED_MESSAGES = {
    "decord 0.6.0 is not supported on this platform",
}


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip() or "Dependências Python verificadas.")
        return 0

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    unexpected = [line for line in lines if line not in IGNORED_MESSAGES]
    ignored = [line for line in lines if line in IGNORED_MESSAGES]

    for line in ignored:
        print(f"IGNORADO (metadado Decord conhecido): {line}")
    if result.stderr.strip():
        unexpected.extend(
            line.strip() for line in result.stderr.splitlines() if line.strip()
        )
    if unexpected:
        print("Falha na verificação de dependências:", file=sys.stderr)
        for line in unexpected:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("Dependências Python verificadas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
