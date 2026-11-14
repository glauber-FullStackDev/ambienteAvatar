#!/usr/bin/env python3
"""Diagnóstico somente de leitura do host e de uma instância ComfyUI."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def command(args):
    if not shutil.which(args[0]):
        return {'disponivel': False}
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=20, check=False)
        return {'codigo': p.returncode, 'saida': p.stdout.strip(), 'erro': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'erro': str(exc)}


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.load(response)
    except (OSError, ValueError) as exc:
        return {'erro': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-url', default='http://127.0.0.1:8188')
    parser.add_argument('--models', type=Path, help='Diretório de modelos ComfyUI, se conhecido')
    args = parser.parse_args()
    usage = shutil.disk_usage(args.models if args.models and args.models.exists() else '.')
    result = {
        'sistema': platform.platform(), 'arquitetura': platform.machine(),
        'memoria': command(['free', '-h']),
        'disco_gib': {'total': round(usage.total / 2**30, 2), 'livre': round(usage.free / 2**30, 2)},
        'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv']),
        'cuda': command(['nvcc', '--version']),
        'docker': command(['docker', 'ps', '-a', '--format', '{{.Names}}\t{{.Image}}\t{{.Status}}']),
        'volumes_docker': command(['docker', 'volume', 'ls', '--format', '{{.Name}}']),
        'comfy_system_stats': get_json(args.comfy_url.rstrip('/') + '/system_stats'),
    }
    names = ['MiniMaxH3ImageToVideo', 'MiniMaxH3AddGuide', 'MiniMaxH3Sampler', 'LoadAudio', 'VAELoader', 'UNETLoader', 'CLIPLoader']
    result['nos'] = {name: get_json(args.comfy_url.rstrip('/') + '/object_info/' + name) for name in names}
    if args.models:
        result['modelos'] = [str(p.relative_to(args.models)) for p in args.models.rglob('*') if p.is_file() and p.suffix.lower() in {'.safetensors', '.gguf', '.pt', '.pth', '.ckpt'}] if args.models.is_dir() else {'erro': 'Diretório não encontrado'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
