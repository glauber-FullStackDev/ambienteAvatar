#!/usr/bin/env python3
"""Baixa apenas os arquivos escolhidos; o cache HF permite retomada."""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--sem-turbo', action='store_true', help='Não baixar a LoRA de 4 passos')
p.add_argument('--perfil', choices=['fl2va', 'fasth3'], default='fl2va', help='Modelo para o workflow escolhido')
a = p.parse_args()
root = Path('/opt/ComfyUI/models')
files = [('Comfy-Org/MiniMax-H3','vae/minimax_h3_video_vae_int8_convrot.safetensors'),('Comfy-Org/MiniMax-H3','vae/minimax_h3_audio_vae_fp32.safetensors')]
if a.perfil == 'fl2va':
    files += [('unsloth/MiniMax-H3-GGUF','minimax_h3_fl2va_pruned-Q4_K.gguf'),('unsloth/MiniMax-H3-GGUF','qwen3vl_32b_minimax_h3-Q2_K_M.gguf')]
elif a.perfil == 'fasth3':
    files += [('FastVideo/FastH3-Omni-NVFP4-QAD-Comfy', 'fasth3_omniref_qad_nvfp4_comfy.safetensors')]
if a.perfil != 'fl2va':
    files += [('Comfy-Org/MiniMax-H3', 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')]
if a.perfil == 'fl2va' and not a.sem_turbo:
    files.append(('lightx2v/Minimax-h3-Turbo','minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors'))
for repo, filename in files:
    folder = filename.split('/')[0] if '/' in filename else ('text_encoders' if filename.startswith('qwen') else 'loras' if 'turbo' in filename else 'diffusion_models')
    print('Baixando', repo, filename, flush=True)
    if '/' in filename:
        hf_hub_download(repo_id=repo, filename=filename, local_dir=root)
    else:
        hf_hub_download(repo_id=repo, filename=filename, local_dir=root / folder)
