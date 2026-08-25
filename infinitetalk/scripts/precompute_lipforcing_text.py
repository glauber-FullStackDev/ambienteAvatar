#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


EXPECTED_SHAPE = (1, 512, 4096)


def verify(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        print(f"Embedding Lip Forcing ausente: {path}")
        return False
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        for key in ("context", "text_emb", "prompt_emb"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != EXPECTED_SHAPE:
        print(f"Embedding Lip Forcing invalido: {path}")
        return False
    if value.dtype != torch.bfloat16:
        print(f"Embedding Lip Forcing deve ser bf16; encontrado {value.dtype}")
        return False
    print(f"Embedding Lip Forcing OK: {path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=Path("/opt/LipForcing"))
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="a person talking")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        return 0 if verify(args.output) else 1
    if args.text_encoder is None or not args.text_encoder.exists():
        raise SystemExit("Informe --text-encoder existente para gerar o embedding")

    inference_dir = args.runtime_root / "scripts/inference"
    sys.path.insert(0, str(inference_dir))
    from _common import load_or_encode_text

    options = SimpleNamespace(
        text_embeds_path=None,
        text_encoder_path=str(args.text_encoder),
        prompt=args.prompt,
    )
    embedding = load_or_encode_text(options, "cpu", torch.bfloat16).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding, args.output)
    return 0 if verify(args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
