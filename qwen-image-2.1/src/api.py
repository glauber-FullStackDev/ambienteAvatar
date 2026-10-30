import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from service import InputError, MODEL, QueueManager, SIZES, validate

manager = QueueManager()

@asynccontextmanager
async def lifespan(app):
    manager.start()
    yield
    manager.stop()

app = FastAPI(lifespan=lifespan)


def error(message, code):
    return JSONResponse(status_code=code, content={'error': {'message': message, 'code': code}})


def authorized(request):
    token = os.getenv('API_TOKEN', '')
    if not token:
        return True
    header = request.headers.get('authorization', '')
    return hmac.compare_digest(header, 'Bearer ' + token)


@app.get('/health')
def health():
    state = manager.summary()['worker']
    return JSONResponse(status_code=200 if state['ready'] else 503, content=state)


@app.get('/dashboard', response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8'))


@app.get('/dashboard/api/summary')
def summary(request: Request):
    if not authorized(request):
        return error('Não autorizado.', 401)
    return manager.summary()


def model_info():
    return {'id': MODEL, 'name': 'Qwen Image 2.1', 'description': 'Geração e edição local de imagens.', 'architecture': {'input_modalities': ['text', 'image'], 'output_modalities': ['image']}, 'supported_parameters': {'resolution': {'type': 'enum', 'values': list(SIZES)}, 'aspect_ratio': {'type': 'enum', 'values': list(SIZES['1K'])}, 'n': {'type': 'range', 'min': 1, 'max': 1}, 'input_references': {'type': 'range', 'min': 0, 'max': 10}, 'output_format': {'type': 'enum', 'values': ['png']}, 'seed': {'type': 'boolean'}}, 'supports_streaming': False, 'endpoints': f'/api/v1/images/models/{MODEL}/endpoints'}


@app.get('/api/v1/images/models')
def models(request: Request):
    if not authorized(request):
        return error('Não autorizado.', 401)
    return {'data': [model_info()]}


@app.get('/api/v1/images/models/{model_id:path}/endpoints')
def endpoints(model_id: str, request: Request):
    if not authorized(request):
        return error('Não autorizado.', 401)
    if model_id != MODEL:
        return error('Modelo desconhecido.', 404)
    return {'id': MODEL, 'endpoints': [{'provider_name': 'Local', 'provider_slug': 'local', 'provider_tag': 'local', 'supported_parameters': model_info()['supported_parameters'], 'allowed_passthrough_parameters': [], 'supports_streaming': False, 'pricing': []}]}


@app.post('/api/v1/images')
async def images(request: Request):
    if not authorized(request):
        return error('Não autorizado.', 401)
    try:
        content_length = int(request.headers.get('content-length', '0'))
        if content_length > int(os.getenv('MAX_REQUEST_BYTES', str(150 * 1024 * 1024))):
            return error('Corpo da solicitação excede o limite.', 413)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > int(os.getenv('MAX_REQUEST_BYTES', str(150 * 1024 * 1024))):
                return error('Corpo da solicitação excede o limite.', 413)
        import json
        values = validate(json.loads(body))
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        return error(str(exc) if isinstance(exc, InputError) else 'JSON inválido.', 400)
    try:
        job = manager.submit(values)
    except OverflowError as exc:
        return error(str(exc), 429)
    except RuntimeError as exc:
        return error(str(exc), 503)
    try:
        while not job.done.is_set():
            if await request.is_disconnected():
                manager.disconnect(job)
                return error('Cliente desconectado.', 499)
            await asyncio.sleep(0.2)
        if job.status != 'completed':
            return error(job.error or 'Falha na geração.', job.error_code)
        return {'created': int(time.time()), 'data': [{'b64_json': job.result, 'media_type': 'image/png'}]}
    except asyncio.CancelledError:
        manager.disconnect(job)
        raise
    finally:
        with manager.lock:
            if job.done.is_set():
                job.result = None
