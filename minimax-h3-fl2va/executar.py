#!/usr/bin/env python3
"""Executa o grafo dentro do container, usando o diretório input do ComfyUI."""
import argparse
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE = 'http://127.0.0.1:8188'
INPUT = Path('/opt/ComfyUI/input')


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'ComfyUI HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first-frame', type=Path, required=True)
    p.add_argument('--last-frame', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--width', type=int, default=576)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--steps', type=int, help='Padrão: 4 em Turbo, 20 em qualidade')
    p.add_argument('--quality', action='store_true', help='Usa o grafo base de 20 passos sem LoRA')
    p.add_argument('--output-path', type=Path, required=True, help='Caminho dentro do container')
    p.add_argument('--timeout', type=int, default=14400)
    a = p.parse_args()
    if a.steps is None: a.steps = 20 if a.quality else 4
    if not a.quality and a.steps != 4: p.error('Turbo 4 requer quatro passos; use --quality para outro número')
    if a.width < 32 or a.height < 32 or a.width % 32 or a.height % 32 or a.steps < 1 or a.timeout < 1:
        p.error('dimensões devem ser múltiplos positivos de 32; steps e timeout devem ser positivos')
    for path in (a.first_frame, a.last_frame, a.audio):
        if not path.is_file(): p.error(f'arquivo ausente: {path}')
    duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(a.audio)], text=True))
    if duration > 124 / 24:
        p.error('áudio excede 124/24 segundos; não será cortado ou acelerado')
    task = uuid.uuid4().hex
    INPUT.mkdir(parents=True, exist_ok=True)
    first, last, audio = (f'h3_{task}_first{a.first_frame.suffix.lower()}', f'h3_{task}_last{a.last_frame.suffix.lower()}', f'h3_{task}.wav')
    shutil.copyfile(a.first_frame, INPUT / first)
    shutil.copyfile(a.last_frame, INPUT / last)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(a.audio),'-vn','-af',f'apad=whole_dur={124/24:.9f}','-ar','32000','-ac','2',str(INPUT/audio)], check=True)
    graph_file = 'h3_first_last_audio_api.json' if a.quality else 'h3_first_last_audio_turbo4_api.json'
    graph = json.loads((Path('/opt/pipeline') / graph_file).read_text())
    graph['5']['inputs']['image'], graph['6']['inputs']['image'], graph['7']['inputs']['audio'] = first, last, audio
    graph['8']['inputs'].update(prompt=a.prompt, width=a.width, height=a.height)
    graph['11']['inputs']['noise_seed'] = a.seed
    graph['13']['inputs']['steps'] = a.steps
    graph['18']['inputs']['filename_prefix'] = 'h3/' + task
    info = request('/object_info')
    missing = sorted({n['class_type'] for n in graph.values()} - set(info))
    if missing: raise RuntimeError('Nós ausentes: ' + ', '.join(missing))
    submitted = request('/prompt', {'prompt':graph, 'client_id':task})
    prompt_id = submitted['prompt_id']
    print('Solicitação:', prompt_id, flush=True)
    end = time.monotonic() + a.timeout
    while time.monotonic() < end:
        history = request('/history/' + urllib.parse.quote(prompt_id))
        if prompt_id in history:
            result = history[prompt_id]
            status = result.get('status', {})
            if status.get('status_str') == 'error':
                raise RuntimeError('Falha de execução: ' + json.dumps(status.get('messages'), ensure_ascii=False))
            if status.get('completed'):
                outputs = result.get('outputs', {}).get('18', {})
                files = outputs.get('videos') or outputs.get('gifs') or []
                if not files: raise RuntimeError('ComfyUI não retornou vídeo: ' + json.dumps(outputs))
                file = files[0]
                query = urllib.parse.urlencode({k:file.get(k,'') for k in ('filename','subfolder','type')})
                a.output_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostic = a.output_path.with_name(a.output_path.stem + '_gerado.mp4')
                with urllib.request.urlopen(BASE + '/view?' + query, timeout=60) as response, diagnostic.open('wb') as out:
                    shutil.copyfileobj(response, out)
                generated_audio = a.output_path.with_name(a.output_path.stem + '_audio_gerado.wav')
                subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(diagnostic),'-vn','-c:a','pcm_s16le',str(generated_audio)],check=True)
                subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(diagnostic),'-i',str(a.audio),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-af',f'apad=whole_dur={124/24:.9f}',str(a.output_path)],check=True)
                print('Final:', a.output_path, '\nDiagnóstico:', diagnostic, '\nÁudio gerado:', generated_audio)
                return
        time.sleep(5)
    raise TimeoutError(f'Tempo esgotado; tarefa {prompt_id} pode continuar no ComfyUI')


if __name__ == '__main__':
    main()
