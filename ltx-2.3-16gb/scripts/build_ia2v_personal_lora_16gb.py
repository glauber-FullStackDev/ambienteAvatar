#!/usr/bin/env python3
"""Cria o workflow IA2V + LoRA pessoal GGUF 16 GB a partir do workflow pessoal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from gguf_convert import convert_to_gguf, validate_gguf
from personal_workflows import add_last_frame_export, validate_personal_lora
from workflow_utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cria o workflow IA2V pessoal LTX 2.3 GGUF 16 GB"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workflow = add_last_frame_export(load_json(args.source))
    validate_personal_lora(workflow)
    workflow = convert_to_gguf(workflow)
    validate_gguf(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Workflow IA2V GGUF 16 GB criado: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise SystemExit(1) from error
