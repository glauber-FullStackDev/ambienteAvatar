import base64
import binascii
import io
import ipaddress
import logging
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit

import urllib3
from PIL import Image, UnidentifiedImageError
import webhook
from reuse_encoder import reuse_cached_encoder

MODEL = 'black-forest-labs/flux-3-image-klein'
MODEL_ALIAS = 'qwen/qwen-image-2.1'
MODELS = {MODEL, MODEL_ALIAS}
SIZES_2K = {'1:1': (2048, 2048), '4:3': (2400, 1792), '3:4': (1792, 2400), '3:2': (2528, 1696), '2:3': (1696, 2528), '16:9': (2752, 1536), '9:16': (1536, 2752)}
SIZES = {'2K': SIZES_2K, '1K': {k: tuple(round(v / 64) * 32 for v in size) for k, size in SIZES_2K.items()}}
MAX_REFERENCE_BYTES = int(os.getenv('MAX_REFERENCE_BYTES', str(10 * 1024 * 1024)))
MAX_PIXELS = int(os.getenv('MAX_REFERENCE_PIXELS', '40000000'))
LOG = logging.getLogger('flux-3-image-klein')

class InputError(ValueError):
    pass


def validate(payload):
    if not isinstance(payload, dict):
        raise InputError('O corpo deve ser um objeto JSON.')
    allowed = {'model', 'prompt', 'seed', 'resolution', 'aspect_ratio', 'size', 'n', 'output_format', 'input_references', 'stream'}
    unknown = set(payload) - allowed
    if unknown:
        raise InputError('Parâmetros não suportados: ' + ', '.join(sorted(unknown)))
    if not isinstance(payload.get('model'), str) or payload['model'] not in MODELS:
        raise InputError(f'model deve ser {MODEL} ou {MODEL_ALIAS}.')
    prompt = payload.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10000:
        raise InputError('prompt deve conter entre 1 e 10000 caracteres.')
    if payload.get('stream', False) is not False:
        raise InputError('stream não é suportado.')
    if type(payload.get('n', 1)) is not int or payload.get('n', 1) != 1:
        raise InputError('Somente n=1 é suportado.')
    if payload.get('output_format', 'png') != 'png':
        raise InputError('Somente output_format=png é suportado.')
    seed = payload.get('seed')
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**63 - 1):
        raise InputError('seed deve ser inteiro entre 0 e 2^63-1.')
    resolution, ratio, size = payload.get('resolution', '1K'), payload.get('aspect_ratio', '1:1'), payload.get('size')
    if size is not None:
        if not isinstance(size, str):
            raise InputError('size inválido.')
        if size in SIZES:
            if 'resolution' in payload and resolution != size:
                raise InputError('size e resolution são conflitantes.')
            resolution = size
        else:
            matches = [(tier, ar) for tier, ratios in SIZES.items() for ar, dims in ratios.items() if size == f'{dims[0]}x{dims[1]}']
            if not matches:
                raise InputError('size não é suportado.')
            tier, ar = matches[0]
            if ('resolution' in payload and resolution != tier) or ('aspect_ratio' in payload and ratio != ar):
                raise InputError('size, resolution e aspect_ratio são conflitantes.')
            resolution, ratio = tier, ar
    if not isinstance(resolution, str) or resolution not in SIZES or not isinstance(ratio, str) or ratio not in SIZES[resolution]:
        raise InputError('resolution ou aspect_ratio não suportado.')
    refs = payload.get('input_references', [])
    if not isinstance(refs, list) or len(refs) > 10:
        raise InputError('input_references deve ser uma lista com até 10 imagens.')
    urls = []
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {'type', 'image_url'} or ref['type'] != 'image_url' or not isinstance(ref['image_url'], dict) or set(ref['image_url']) != {'url'} or not isinstance(ref['image_url']['url'], str):
            raise InputError('Referência deve ter type=image_url e image_url.url.')
        urls.append(ref['image_url']['url'])
    return {'prompt': prompt, 'seed': seed, 'width': SIZES[resolution][ratio][0], 'height': SIZES[resolution][ratio][1], 'references': urls}


def fetch_url(url):
    # Connect to a validated IP directly, preserving Host and TLS SNI: avoids DNS rebinding.
    for _ in range(4):
        try:
            parts = urlsplit(url)
        except ValueError as exc:
            raise InputError('URL de referência inválida.') from exc
        if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
            raise InputError('URL de referência inválida.')
        try:
            port = parts.port or (443 if parts.scheme == 'https' else 80)
            addresses = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
            ips = {item[4][0] for item in addresses}
            if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
                raise InputError('URL de referência aponta para endereço não público.')
            ip = sorted(ips)[0]
            host = parts.hostname if parts.port is None else f'{parts.hostname}:{port}'
            path = parts.path or '/'
            if parts.query:
                path += '?' + parts.query
            if parts.scheme == 'https':
                pool = urllib3.HTTPSConnectionPool(ip, port, assert_hostname=parts.hostname, server_hostname=parts.hostname, timeout=urllib3.Timeout(connect=5, read=15), retries=False)
            else:
                pool = urllib3.HTTPConnectionPool(ip, port, timeout=urllib3.Timeout(connect=5, read=15), retries=False)
            response = pool.request('GET', path, headers={'Host': host}, preload_content=False, redirect=False)
            try:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location')
                    if not location:
                        raise InputError('Redirecionamento sem destino.')
                    url = urljoin(url, location)
                    continue
                if response.status != 200:
                    raise InputError('Não foi possível baixar a imagem de referência.')
                data = bytearray()
                deadline = time.monotonic() + 30
                while True:
                    if time.monotonic() > deadline:
                        raise InputError('Download da referência excedeu o tempo limite.')
                    chunk = response.read1(min(65536, MAX_REFERENCE_BYTES + 1 - len(data)))
                    if not chunk:
                        return bytes(data)
                    data.extend(chunk)
                    if len(data) > MAX_REFERENCE_BYTES:
                        raise InputError('Imagem de referência excede o limite de bytes.')
            finally:
                response.close()
                pool.close()
        except InputError:
            raise
        except (ValueError, OSError, urllib3.exceptions.HTTPError) as exc:
            raise InputError('Não foi possível baixar a imagem de referência.') from exc
    raise InputError('Muitos redirecionamentos na referência.')


def load_reference(url):
    if url.startswith('data:'):
        try:
            header, encoded = url.split(',', 1)
            if not header.startswith('data:image/') or not header.endswith(';base64') or len(encoded) > ((MAX_REFERENCE_BYTES + 2) // 3) * 4:
                raise ValueError()
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise InputError('Data URL de imagem inválida ou muito grande.') from exc
    else:
        data = fetch_url(url)
    if len(data) > MAX_REFERENCE_BYTES:
        raise InputError('Imagem de referência excede o limite de bytes.')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise InputError('Imagem de referência excede o limite de pixels.')
            image.load()
            return image.convert('RGBA' if 'A' in image.getbands() else 'RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InputError('Imagem de referência inválida.') from exc


COMFY_REPO = 'black-forest-labs/flux-3-image-klein-comfy'
COMFY_FILES = (
    'diffusion_models/flux3-image-klein-bf16.safetensors',
    'text_encoders/qwen3vl_4b_bf16.safetensors',
    'vae/flux2-vae.safetensors',
)


def build_workflow(values, image_names):
    def node(name, inputs): return {'class_type': name, 'inputs': inputs}
    nodes = {
        '1': node('UNETLoader', {'unet_name': COMFY_FILES[0].split('/')[-1], 'weight_dtype': 'default'}),
        '2': node('CLIPLoader', {'clip_name': COMFY_FILES[1].split('/')[-1], 'type': 'flux3', 'device': 'default'}),
        '3': node('VAELoader', {'vae_name': COMFY_FILES[2].split('/')[-1]}),
        '6': node('CLIPTextEncode', {'clip': ['2', 0], 'text': values['prompt']}),
        '7': node('ConditioningZeroOut', {'conditioning': ['6', 0]}),
        '8': node('EmptyFlux2LatentImage', {'width': values['width'], 'height': values['height'], 'batch_size': 1}),
        '9': node('KSampler', {'model': ['1', 0], 'positive': ['6', 0], 'negative': ['7', 0],
                    'latent_image': ['8', 0], 'seed': values['seed'], 'steps': 16, 'cfg': 1.0,
                    'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}),
        '10': node('VAEDecode', {'samples': ['9', 0], 'vae': ['3', 0]}),
        '11': node('PreviewImage', {'images': ['10', 0]}),
    }
    if image_names:
        inputs = {'positive': ['6', 0], 'negative': ['7', 0], 'vae': ['3', 0],
                  'resolution': 'resize', 'resolution.megapixels': 1.0}
        for i, name in enumerate(image_names):
            key = str(100 + i)
            nodes[key] = node('LoadImage', {'image': name})
            inputs[f'images.image{i}'] = [key, 0]
        nodes['12'] = node('ReferenceMultiLatent', inputs)
        nodes['9']['inputs'].update({'positive': ['12', 0], 'negative': ['12', 1]})
    return nodes


class Pipeline:
    def __init__(self):
        import atexit
        from huggingface_hub import hf_hub_download
        self.base = Path(os.getenv('COMFY_DATA_DIR', '/data/comfy'))
        self.input_dir = self.base / 'input'
        self.input_dir.mkdir(parents=True, exist_ok=True)
        models = self.base / 'models'
        for filename in COMFY_FILES:
            if filename == COMFY_FILES[1] and reuse_cached_encoder(os.getenv('HF_HOME', '/data/huggingface'), models / filename):
                continue
            hf_hub_download(COMFY_REPO, filename, local_dir=str(models),
                            revision=os.getenv('COMFY_MODEL_REVISION', 'main'))
        self.http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5, read=30), retries=False)
        self.url = 'http://127.0.0.1:8188'
        self.process = subprocess.Popen(['python', '/opt/ComfyUI/main.py', '--listen', '127.0.0.1',
                                         '--port', '8188', '--base-directory', str(self.base),
                                         '--disable-auto-launch', '--disable-all-custom-nodes',
                                         '--lowvram', '--disable-smart-memory'])
        atexit.register(self.close)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('ComfyUI encerrou durante a inicialização; consulte os logs.')
                try:
                    response = self.http.request('GET', self.url + '/object_info')
                    if response.status == 200:
                        available = json.loads(response.data)
                        required = {'ReferenceMultiLatent', 'EmptyFlux2LatentImage'}
                        if not required.issubset(available):
                            raise RuntimeError('A versão do ComfyUI não contém os nós necessários para FLUX 3.')
                        return
                except urllib3.exceptions.HTTPError:
                    pass
                time.sleep(1)
            raise RuntimeError('ComfyUI não iniciou dentro do tempo limite.')
        except Exception:
            self.close()
            raise

    def close(self):
        process = getattr(self, 'process', None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def generate(self, values, progress):
        names = []
        progress('loading_references', None, None)
        try:
            for url in values['references']:
                name = f'flux-reference-{uuid.uuid4().hex}.png'
                names.append(name)
                load_reference(url).convert('RGB').save(self.input_dir / name, format='PNG')
            data = dict(values)
            data['seed'] = values['seed'] if values['seed'] is not None else int.from_bytes(os.urandom(8), 'big') % 2**63
            workflow = build_workflow(data, names)
            workflow['9']['inputs']['steps'] = int(os.getenv('INFERENCE_STEPS', '16'))
            progress('preparing', None, None)
            response = self.http.request('POST', self.url + '/prompt',
                                         body=json.dumps({'prompt': workflow}).encode(),
                                         headers={'Content-Type': 'application/json'})
            if response.status != 200:
                LOG.error('ComfyUI recusou workflow (%s): %s', response.status, response.data[:4000].decode(errors='replace'))
                raise RuntimeError('ComfyUI recusou o workflow; consulte os logs.')
            prompt_id = json.loads(response.data)['prompt_id']
            progress('sampling', None, None)
            deadline = time.monotonic() + int(os.getenv('GENERATION_TIMEOUT', '3600'))
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('ComfyUI encerrou durante a geração.')
                response = self.http.request('GET', self.url + '/history/' + prompt_id)
                if response.status != 200:
                    raise RuntimeError('Falha ao consultar o resultado do ComfyUI.')
                history = json.loads(response.data).get(prompt_id)
                if history:
                    if history.get('status', {}).get('status_str') == 'error':
                        LOG.error('Erro do ComfyUI: %s', history.get('status'))
                        raise RuntimeError('ComfyUI falhou na geração; consulte os logs.')
                    images = history.get('outputs', {}).get('11', {}).get('images', [])
                    if not images:
                        raise RuntimeError('ComfyUI não retornou imagem.')
                    image = images[0]
                    query = urlencode({'filename': image['filename'], 'subfolder': image.get('subfolder', ''), 'type': image.get('type', 'temp')})
                    result = self.http.request('GET', self.url + '/view?' + query)
                    if image.get('type', 'temp') == 'temp':
                        temp_dir = (self.base / 'temp').resolve()
                        temp_file = (temp_dir / image.get('subfolder', '') / image['filename']).resolve()
                        if temp_file.is_relative_to(temp_dir) and temp_file != temp_dir:
                            temp_file.unlink(missing_ok=True)
                    if result.status != 200:
                        raise RuntimeError('Não foi possível obter imagem do ComfyUI.')
                    progress('encoding', None, None)
                    with Image.open(io.BytesIO(result.data)) as output_image:
                        output = io.BytesIO()
                        output_image.save(output, format='PNG')
                    return base64.b64encode(output.getvalue()).decode('ascii')
                time.sleep(1)
            raise RuntimeError('Geração excedeu o tempo limite.')
        finally:
            for name in names:
                (self.input_dir / name).unlink(missing_ok=True)


@dataclass
class Job:
    values: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    status: str = 'queued'
    phase: str = 'queued'
    step: int | None = None
    total_steps: int | None = None
    result: str | None = None
    error: str | None = None
    error_code: int = 500
    disconnected: bool = False
    webhook_url: str | None = None
    webhook_status: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


class QueueManager:
    def __init__(self, pipeline_factory=Pipeline, max_queue=None, history_size=None):
        self.factory = pipeline_factory
        self.max_queue = int(os.getenv('MAX_QUEUE_SIZE', '10')) if max_queue is None else max_queue
        self.history = deque(maxlen=int(os.getenv('HISTORY_SIZE', '100')) if history_size is None else history_size)
        self.queue = deque()
        self.current = None
        self.counts = Counter()
        self.lock = threading.Condition()
        self.ready = False
        self.startup_error = None
        self.stopping = False
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self.stopping = True
            while self.queue:
                job = self.queue.popleft()
                self._finish(job, 'failed', 'Serviço encerrado.', 503)
            self.lock.notify_all()

    def submit(self, values, webhook_url=None):
        with self.lock:
            if not self.ready or self.stopping:
                raise RuntimeError('Modelo ainda não está pronto.')
            if len(self.queue) >= self.max_queue and (self.current is not None or self.queue):
                raise OverflowError('Fila cheia.')
            job = Job(values, webhook_url=webhook_url, webhook_status='pending' if webhook_url else None)
            self.queue.append(job)
            self.lock.notify()
            return job

    def disconnect(self, job):
        with self.lock:
            job.disconnected = True
            job.result = None
            if job.status == 'queued':
                try:
                    self.queue.remove(job)
                except ValueError:
                    return
                self._finish(job, 'cancelled', 'Cliente desconectado.', 499)

    def _finish(self, job, status, error=None, code=500):
        job.status, job.phase, job.error, job.error_code, job.finished_at = status, status, error, code, time.time()
        job.values['references'] = []
        self.counts[status] += 1
        self.history.appendleft(job)
        job.done.set()

    def _run(self):
        try:
            pipeline = self.factory()
            with self.lock:
                self.ready = True
        except Exception as exc:
            LOG.exception('Falha ao carregar modelo')
            with self.lock:
                self.startup_error = str(exc)
            return
        while True:
            with self.lock:
                while not self.queue and not self.stopping:
                    self.lock.wait()
                if self.stopping:
                    if hasattr(pipeline, 'close'):
                        pipeline.close()
                    return
                job = self.queue.popleft()
                self.current = job
                job.status, job.started_at = 'running', time.time()
            def progress(phase, step, total):
                with self.lock:
                    job.phase, job.step, job.total_steps = phase, step, total
            try:
                result = pipeline.generate(job.values, progress)
                with self.lock:
                    job.result = None if job.disconnected else result
                    self._finish(job, 'completed')
                    self.current = None
                if job.webhook_url:
                    self._deliver_async(job)
            except Exception as exc:
                LOG.exception('Falha no job %s', job.id)
                with self.lock:
                    self._finish(job, 'failed', str(exc) if isinstance(exc, InputError) else 'Falha na geração da imagem.', 400 if isinstance(exc, InputError) else 500)
                    self.current = None
                if job.webhook_url:
                    self._deliver_async(job)

    def _deliver_async(self, job):
        payload = {'event': 'job.completed' if job.status == 'completed' else 'job.failed',
                   'job_id': job.id, 'status': job.status,
                   'result': {'created': int(job.finished_at), 'data': [{'b64_json': job.result, 'media_type': 'image/png'}]} if job.status == 'completed' else None,
                   'error': job.error}
        def run():
            try:
                sent = webhook.deliver(job.webhook_url, payload)
            except Exception:
                LOG.exception('Falha inesperada no webhook do job %s', job.id)
                sent = False
            with self.lock:
                job.webhook_status = 'delivered' if sent else 'failed'
        threading.Thread(target=run, daemon=True).start()

    def job_status(self, job_id):
        with self.lock:
            jobs = ([self.current] if self.current else []) + list(self.queue) + list(self.history)
            job = next ( (j for j in jobs if j.id == job_id and j.webhook_url), None)
            if not job:
                return None
            position = next((i + 1 for i, j in enumerate(self.queue) if j.id == job_id), None)
            return {'job_id': job.id, 'status': job.status, 'queue_position': position,
                    'progress': {'phase': job.phase, 'step': job.step, 'total_steps': job.total_steps},
                    'webhook_status': job.webhook_status, 'error': job.error,
                    'result': {'created': int(job.finished_at), 'data': [{'b64_json': job.result, 'media_type': 'image/png'}]} if job.status == 'completed' else None}

    def summary(self):
        with self.lock:
            def entry(job):
                return {'job_id': job.id, 'prompt': job.values['prompt'], 'status': job.status, 'created_at': job.created_at, 'started_at': job.started_at, 'finished_at': job.finished_at, 'error': job.error, 'progress': {'phase': job.phase, 'step': job.step, 'total_steps': job.total_steps, 'percent': round(job.step * 100 / job.total_steps, 1) if job.phase == 'sampling' and job.total_steps else None}}
            queue = [dict(entry(job), queue_position=i + 1) for i, job in enumerate(self.queue)]
            return {'worker': {'ready': self.ready, 'startup_error': self.startup_error, 'busy': self.current is not None}, 'current': entry(self.current) if self.current else None, 'queue': queue, 'history': [entry(j) for j in self.history], 'counts': {'queued': len(queue), 'running': int(self.current is not None), **{s: self.counts[s] for s in ('completed', 'failed', 'cancelled')}}}
