"""Qwen prompt enhancer via a short-lived, GPU-offloaded llama.cpp server."""
import base64
import io
import json
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

import urllib3

LOG = logging.getLogger('qwen.enhancer')
GGUF_REPO = 'mozophe/Qwen-Image-2.1-PE-MTP-GGUF'
GGUF_REVISION = 'b56146a6dd48f7a7eb33a0a85ad3f55376abe449'
T2I_MODEL = 'Qwen/Qwen-Image-2.1-PE-T2I'
EDIT_MODEL = 'Qwen/Qwen-Image-2.1-PE-I2I'
SOURCE_REVISIONS = {'t2i': 'f3ed7985c788ad75b3ab7223e0c4c51e2a43545b', 'i2i': '72927bc08afc99b7888ceb7d7d51a12db3700bbd'}


def enabled():
    value = os.getenv('PROMPT_ENHANCER_ENABLED', 'true').strip().lower()
    if value not in {'true', 'false', '1', '0'}:
        raise ValueError('PROMPT_ENHANCER_ENABLED deve ser true ou false.')
    return value in {'true', '1'}


def parse_rewrite(text):
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    elif '<think>' in text:
        raise ValueError('Prompt enhancer não terminou de gerar a resposta.')
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        prompt = value.get('rewritten_prompt') if isinstance(value, dict) else None
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    raise ValueError('Prompt enhancer não retornou um prompt válido.')


def _port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _image_data_url(image):
    rgb = image.convert('RGB')
    buffer = io.BytesIO()
    rgb.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def _payload(prompt, images, system_prompt):
    content = [{'type': 'image_url', 'image_url': {'url': _image_data_url(image)}} for image in images]
    content.append({'type': 'text', 'text': prompt})
    return {'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': content}],
            'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0,
            'presence_penalty': 0.0 if images else 1.5,
            'max_tokens': int(os.getenv('PROMPT_ENHANCER_MAX_TOKENS', '8192')),
            'chat_template_kwargs': {'enable_thinking': True}, 'stream': False}


def _wait_ready(http, base, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'llama-server encerrou durante inicialização (código {process.returncode}); consulte os logs.')
        try:
            response = http.request('GET', base + '/health', timeout=urllib3.Timeout(connect=1, read=2), retries=False)
            if response.status == 200:
                return
        except urllib3.exceptions.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError('Tempo limite ao iniciar o llama-server do prompt enhancer.')


def _stop(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def enhance(prompt, images):
    from huggingface_hub import hf_hub_download

    started = time.monotonic()
    kind = 'i2i' if images else 't2i'
    repo = os.getenv('PROMPT_ENHANCER_GGUF_REPO', GGUF_REPO)
    revision = os.getenv('PROMPT_ENHANCER_GGUF_REVISION', GGUF_REVISION)
    filename = f'qwen3.5_9b_qwen_image_2.1_pe_{kind}.mtp.Q4_K_M.gguf'
    LOG.info('Preparando modelo GGUF %s para %s', filename, kind)
    model = hf_hub_download(repo, filename, revision=revision)
    mmproj = hf_hub_download(repo, 'qwen3.5_9b_qwen_image_2.1_pe_i2i.mmproj.bf16.gguf', revision=revision) if images else None
    source_repo = EDIT_MODEL if images else T2I_MODEL
    system_prompt = Path(hf_hub_download(source_repo, 'system_prompt.txt', revision=SOURCE_REVISIONS[kind])).read_text(encoding='utf-8').strip()
    port = _port()
    base = f'http://127.0.0.1:{port}'
    command = [os.getenv('LLAMA_SERVER_PATH', '/opt/llama/bin/llama-server'), '-m', model,
               '--host', '127.0.0.1', '--port', str(port), '-ngl', '99', '-fa', 'on',
               '--fit', 'off', '-c', os.getenv('PROMPT_ENHANCER_CONTEXT_SIZE', '24576'),
               '--parallel', '1', '--jinja', '--reasoning-format', 'none', '--spec-type', 'draft-mtp']
    if mmproj:
        command += ['--mmproj', mmproj]
    process = None
    http = urllib3.PoolManager(retries=False)
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        _wait_ready(http, base, process, float(os.getenv('PROMPT_ENHANCER_START_TIMEOUT', '300')))
        inference_started = time.monotonic()
        LOG.info('Prompt enhancer pronto em %.1fs; iniciando reescrita', inference_started - started)
        payload = _payload(prompt, images, system_prompt)
        response = http.request('POST', base + '/v1/chat/completions', body=json.dumps(payload).encode('utf-8'),
                                headers={'Content-Type': 'application/json'},
                                timeout=urllib3.Timeout(connect=5, read=float(os.getenv('PROMPT_ENHANCER_TIMEOUT', '1800'))), retries=False)
        if response.status != 200:
            raise RuntimeError(f'llama-server retornou HTTP {response.status}; consulte os logs.')
        data = json.loads(response.data)
        choice = data['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise RuntimeError('Prompt enhancer atingiu o limite de tokens antes de concluir.')
        result = parse_rewrite(choice['message']['content'] or '')
        LOG.info('Prompt reescrito em %.1fs (total %.1fs; %s tokens)', time.monotonic() - inference_started, time.monotonic() - started, (data.get('usage') or {}).get('completion_tokens', '?'))
        return result
    finally:
        http.clear()
        _stop(process)
        LOG.info('llama-server encerrado; memória da GPU liberada para geração')
