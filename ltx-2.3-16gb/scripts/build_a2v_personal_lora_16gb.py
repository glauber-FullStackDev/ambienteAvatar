#!/usr/bin/env python3
"""Cria o workflow A2V + LoRA pessoal GGUF 16 GB a partir do IA2V pessoal.

Ordem importa: a remocao do branch I2V acontece primeiro (nodes 294/296/297/
325/334, incluindo um consumer do VAE do checkpoint) e a conversao GGUF roda
depois, herdando apenas os consumers restantes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from gguf_convert import convert_to_gguf, validate_gguf
from personal_workflows import build_a2v, validate_a2v
from workflow_utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cria o workflow A2V pessoal LTX 2.3 GGUF 16 GB"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workflow = build_a2v(load_json(args.source))
    validate_a2v(workflow)
    workflow = convert_to_gguf(workflow)
    validate_gguf(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Workflow A2V GGUF 16 GB criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
