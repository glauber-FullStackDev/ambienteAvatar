import base64
import io
import os
import sys
import threading
import time
import tempfile
import unittest
import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import api
import webhook
from service import InputError, MODEL, MODEL_ALIAS, Pipeline, QueueManager, build_workflow, fetch_url, load_reference, validate

PAYLOAD = {'model': MODEL, 'prompt': 'uma paisagem'}

class FakePipeline:
    def generate(self, values, progress):
        progress('sampling', 1, 2)
        data = io.BytesIO()
        Image.new('RGB', (4, 4), 'red').save(data, format='PNG')
        return base64.b64encode(data.getvalue()).decode()

class ValidationTests(unittest.TestCase):
    def test_dimensions_and_conflicts(self):
        self.assertEqual((validate(PAYLOAD)['width'], validate(PAYLOAD)['height']), (1024, 1024))
        self.assertEqual(validate({**PAYLOAD, 'size': '2752x1536'})['height'], 1536)
        for extra in ({'stream': True}, {'n': 2}, {'seed': True}, {'size': '2048x2048', 'aspect_ratio': '16:9'}, {'quality': 'high'}):
            with self.subTest(extra=extra), self.assertRaises(InputError):
                validate({**PAYLOAD, **extra})

    def test_qwen_alias_and_ten_references(self):
        refs = [{'type': 'image_url', 'image_url': {'url': 'https://example.com/a.png'}}] * 10
        expected = validate({**PAYLOAD, 'input_references': refs})
        self.assertEqual(validate({**PAYLOAD, 'model': MODEL_ALIAS, 'input_references': refs}), expected)
        with self.assertRaises(InputError):
            validate({**PAYLOAD, 'input_references': refs + refs[:1]})

    def test_reference_validation(self):
        image = io.BytesIO()
        Image.new('RGB', (2, 2)).save(image, 'PNG')
        url = 'data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()
        self.assertEqual(load_reference(url).size, (2, 2))
        for url in ('data:image/png;base64,@@@', 'http://127.0.0.1/test', 'http://localhost/test'):
            with self.subTest(url=url), self.assertRaises(InputError):
                load_reference(url)

    def test_public_redirect_to_private_address_is_rejected(self):
        class Response:
            status = 302
            headers = {'Location': 'http://127.0.0.1/admin'}
            def close(self): pass
        class Pool:
            def __init__(self, *args, **kwargs): pass
            def request(self, *args, **kwargs): return Response()
            def close(self): pass
        def resolve(host, port, **kwargs):
            ip = '93.184.216.34' if host == 'example.com' else '127.0.0.1'
            return [(None, None, None, None, (ip, port))]
        with patch('service.socket.getaddrinfo', side_effect=resolve), patch('service.urllib3.HTTPSConnectionPool', Pool):
            with self.assertRaises(InputError):
                fetch_url('https://example.com/image')

class QueueTests(unittest.TestCase):
    def test_workflow_passes_all_references_and_preserves_dimensions(self):
        values = validate({**PAYLOAD, 'seed': 42, 'resolution': '2K', 'aspect_ratio': '16:9'})
        names = [f'reference-{i}.png' for i in range(10)]
        graph = build_workflow(values, names)
        self.assertEqual(graph['6']['inputs']['text'], PAYLOAD['prompt'])
        self.assertNotIn('TextGenerate', [node['class_type'] for node in graph.values()])
        self.assertEqual(graph['12']['inputs']['images.image9'], ['109', 0])
        self.assertEqual(graph['9']['inputs']['positive'], ['12', 0])
        self.assertEqual(graph['8']['inputs']['width'], 2752)
        self.assertEqual(graph['8']['inputs']['height'], 1536)
        self.assertEqual(graph['9']['inputs']['seed'], 42)
        self.assertEqual(graph['1']['inputs']['unet_name'], 'flux3-image-klein-bf16.safetensors')
        self.assertEqual(graph['2']['inputs']['clip_name'], 'qwen3vl_4b_bf16.safetensors')
        plain = build_workflow(values, [])
        self.assertEqual(plain['6']['inputs']['text'], PAYLOAD['prompt'])
        self.assertNotIn('12', plain)
        self.assertEqual(plain['9']['inputs']['positive'], ['6', 0])

    def test_generation_submits_workflow_and_returns_png(self):
        calls = []
        image = io.BytesIO()
        Image.new('RGB', (2, 2)).save(image, 'PNG')
        class Http:
            def request(self, method, url, **kwargs):
                calls.append((method, url, kwargs))
                if url.endswith('/prompt'):
                    return SimpleNamespace(status=200, data=b'{"prompt_id":"abc"}')
                if '/history/' in url:
                    return SimpleNamespace(status=200, data=b'{"abc":{"outputs":{"11":{"images":[{"filename":"x.png","type":"temp"}]}}}}')
                return SimpleNamespace(status=200, data=image.getvalue())
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.url = 'http://127.0.0.1:8188'
        pipeline.http = Http()
        pipeline.process = SimpleNamespace(poll=lambda: None)
        with tempfile.TemporaryDirectory() as folder:
            pipeline.input_dir = Path(folder)
            pipeline.base = Path(folder)
            result = pipeline.generate(validate({**PAYLOAD, 'seed': 42}), lambda *p: None)
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(result))).format, 'PNG')
        sent = json.loads(calls[0][2]['body'])['prompt']
        self.assertEqual(sent['9']['inputs']['seed'], 42)
        self.assertEqual(sent['6']['inputs']['text'], PAYLOAD['prompt'])

    def test_fifo_limit_progress_and_disconnect(self):
        started, release = threading.Event(), threading.Event()
        order = []
        class Blocking:
            def generate(self, values, progress):
                order.append(values['prompt'])
                progress('sampling', 1, 4)
                started.set()
                release.wait(5)
                return 'image'
        q = QueueManager(Blocking, max_queue=2)
        q.start()
        for _ in range(100):
            if q.ready: break
            time.sleep(.01)
        first = q.submit(validate({**PAYLOAD, 'prompt': 'first'}))
        self.assertTrue(started.wait(2))
        second = q.submit(validate({**PAYLOAD, 'prompt': 'second'}))
        third = q.submit(validate({**PAYLOAD, 'prompt': 'third'}))
        with self.assertRaises(OverflowError): q.submit(validate(PAYLOAD))
        summary = q.summary()
        self.assertEqual(summary['current']['progress']['percent'], 25)
        self.assertEqual([x['queue_position'] for x in summary['queue']], [1, 2])
        q.disconnect(second)
        self.assertEqual(second.status, 'cancelled')
        release.set()
        self.assertTrue(third.done.wait(2))
        self.assertEqual(order, ['first', 'third'])
        self.assertEqual(q.summary()['counts']['completed'], 2)
        q.stop()

    def test_running_disconnected_discards_result(self):
        started, release = threading.Event(), threading.Event()
        class Blocking:
            def generate(self, values, progress):
                started.set(); release.wait(2); return 'image'
        q = QueueManager(Blocking)
        q.start()
        for _ in range(100):
            if q.ready: break
            time.sleep(.01)
        job = q.submit(validate(PAYLOAD))
        self.assertTrue(started.wait(2))
        q.disconnect(job)
        release.set()
        self.assertTrue(job.done.wait(2))
        self.assertIsNone(job.result)
        q.stop()

class ApiTests(unittest.TestCase):
    def test_async_job_and_webhook(self):
        received = []
        delivered = threading.Event()
        def fake_deliver(url, payload):
            received.append((url, payload))
            delivered.set()
            return True
        q = QueueManager(FakePipeline)
        env = {'API_TOKEN': 'secret', 'WEBHOOK_ALLOWED_HOSTS': 'backend.example.com'}
        with patch.object(api, 'manager', q), patch('service.webhook.deliver', side_effect=fake_deliver), patch.dict(os.environ, env):
            with TestClient(api.app) as client:
                for _ in range(100):
                    if q.ready: break
                    time.sleep(.01)
                headers = {'Authorization': 'Bearer secret'}
                invalid = {**PAYLOAD, 'webhook': {'url': 'https://other.example.com/callback'}}
                self.assertEqual(client.post('/api/v1/images/async', headers=headers, json=invalid).status_code, 400)
                response = client.post('/api/v1/images/async', headers=headers, json={**PAYLOAD, 'webhook': {'url': 'https://backend.example.com/callback'}})
                self.assertEqual(response.status_code, 202, response.text)
                job_id = response.json()['job_id']
                self.assertTrue(delivered.wait(2))
                self.assertEqual(received[0][1]['event'], 'job.completed')
                self.assertEqual(received[0][1]['job_id'], job_id)
                self.assertEqual(received[0][1]['result']['data'][0]['media_type'], 'image/png')
                state = client.get(f'/api/v1/images/jobs/{job_id}', headers=headers).json()
                self.assertEqual(state['status'], 'completed')
                self.assertEqual(state['result'], received[0][1]['result'])

    def test_async_failure_sends_error(self):
        received, delivered = [], threading.Event()
        class Broken:
            def generate(self, values, progress): raise RuntimeError('internal error')
        def fake_deliver(url, payload):
            received.append(payload); delivered.set(); return True
        q = QueueManager(Broken)
        with patch.object(api, 'manager', q), patch('service.webhook.deliver', side_effect=fake_deliver), patch.dict(os.environ, {'API_TOKEN': '', 'WEBHOOK_ALLOWED_HOSTS': 'backend.example.com'}):
            with TestClient(api.app) as client:
                for _ in range(100):
                    if q.ready: break
                    time.sleep(.01)
                response = client.post('/api/v1/images/async', json={**PAYLOAD, 'webhook': {'url': 'https://backend.example.com/hook'}})
                self.assertEqual(response.status_code, 202)
                self.assertTrue(delivered.wait(2))
                self.assertEqual(received[0]['event'], 'job.failed')
                self.assertIsNone(received[0]['result'])
                self.assertNotIn('internal error', received[0]['error'])

    def test_concurrent_requests_wait_for_fifo(self):
        started, release = threading.Event(), threading.Event()
        order, responses = [], {}
        class Blocking:
            def generate(self, values, progress):
                order.append(values['prompt'])
                started.set()
                release.wait(5)
                return 'encoded-' + values['prompt']
        q = QueueManager(Blocking, max_queue=1)
        with patch.object(api, 'manager', q), patch.dict(os.environ, {'API_TOKEN': ''}):
            with TestClient(api.app) as client:
                for _ in range(100):
                    if q.ready: break
                    time.sleep(.01)
                def post(name):
                    responses[name] = client.post('/api/v1/images', json={**PAYLOAD, 'prompt': name})
                first = threading.Thread(target=post, args=('first',))
                first.start()
                self.assertTrue(started.wait(2))
                second = threading.Thread(target=post, args=('second',))
                second.start()
                for _ in range(100):
                    if q.summary()['counts']['queued'] == 1: break
                    time.sleep(.01)
                self.assertEqual(q.summary()['counts']['queued'], 1)
                self.assertEqual(client.post('/api/v1/images', json=PAYLOAD).status_code, 429)
                release.set()
                first.join(3); second.join(3)
                self.assertEqual(order, ['first', 'second'])
                self.assertEqual(responses['first'].json()['data'][0]['b64_json'], 'encoded-first')
                self.assertEqual(responses['second'].json()['data'][0]['b64_json'], 'encoded-second')

    def test_http_contract_auth_dashboard(self):
        q = QueueManager(FakePipeline)
        with patch.object(api, 'manager', q), patch.dict(os.environ, {'API_TOKEN': 'secret'}):
            with TestClient(api.app) as client:
                for _ in range(100):
                    if q.ready: break
                    time.sleep(.01)
                self.assertEqual(client.post('/api/v1/images', json=PAYLOAD).status_code, 401)
                headers = {'Authorization': 'Bearer secret'}
                self.assertEqual(client.get('/dashboard/api/summary').status_code, 401)
                self.assertEqual(client.get('/dashboard').status_code, 200)
                self.assertEqual(client.get('/health').status_code, 200)
                self.assertEqual(client.get('/api/v1/images/models', headers=headers).json()['data'][0]['id'], MODEL)
                self.assertEqual(client.get(f'/api/v1/images/models/{MODEL}/endpoints', headers=headers).status_code, 200)
                self.assertEqual(client.post('/api/v1/images', json={**PAYLOAD, 'stream': True}, headers=headers).status_code, 400)
                response = client.post('/api/v1/images', json=PAYLOAD, headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                item = response.json()['data'][0]
                self.assertEqual(item['media_type'], 'image/png')
                self.assertEqual(Image.open(io.BytesIO(base64.b64decode(item['b64_json']))).size, (4, 4))
                self.assertEqual(client.get('/dashboard/api/summary', headers=headers).json()['counts']['completed'], 1)
                alias_endpoint = client.get(f'/api/v1/images/models/{MODEL_ALIAS}/endpoints', headers=headers)
                self.assertEqual(alias_endpoint.json()['id'], MODEL_ALIAS)
                alias_response = client.post('/api/v1/images', json={**PAYLOAD, 'model': MODEL_ALIAS}, headers=headers)
                self.assertEqual(alias_response.status_code, 200, alias_response.text)
                self.assertEqual(alias_response.json()['data'][0]['media_type'], 'image/png')

class WebhookTests(unittest.TestCase):
    def test_signature_retry_and_redirect_rejection(self):
        calls = []
        class Response:
            status = 302
            def close(self): pass
        class Pool:
            def __init__(self, **kwargs): pass
            def request(self, method, url, body, headers, **kwargs):
                calls.append((body, headers, kwargs)); return Response()
            def clear(self): pass
        with patch('webhook.urllib3.PoolManager', Pool), patch('webhook.time.sleep'), patch.dict(os.environ, {'WEBHOOK_SECRET':'abc', 'WEBHOOK_RETRIES':'2'}):
            self.assertFalse(webhook.deliver('https://example.com/hook', {'job_id':'123'}))
        self.assertEqual(len(calls), 2)
        body, headers, options = calls[0]
        self.assertEqual(headers['X-Webhook-Signature'], 'sha256=' + hmac.new(b'abc', body, hashlib.sha256).hexdigest())
        self.assertFalse(options['redirect'])

if __name__ == '__main__': unittest.main()
