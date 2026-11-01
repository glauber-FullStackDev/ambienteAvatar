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
import prompt_enhancer
from service import InputError, MODEL, Pipeline, QueueManager, fetch_url, load_reference, validate

PAYLOAD = {'model': MODEL, 'prompt': 'uma paisagem'}

class FakePipeline:
    def generate(self, values, progress):
        progress('sampling', 1, 2)
        data = io.BytesIO()
        Image.new('RGB', (4, 4), 'red').save(data, format='PNG')
        return base64.b64encode(data.getvalue()).decode()

class ValidationTests(unittest.TestCase):
    def test_enhancer_parses_rewritten_prompt_and_rejects_incomplete_output(self):
        self.assertEqual(prompt_enhancer.parse_rewrite('reasoning</think>\n{"rewritten_prompt":"Detailed scene", "wh_ratio":"16:9"}'), 'Detailed scene')
        self.assertEqual(prompt_enhancer.parse_rewrite('```json\n{"rewritten_prompt":"  A cat  "}\n```'), 'A cat')
        for text in ('<think>unfinished', '{"wh_ratio":"1:1"}', 'invalid'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                prompt_enhancer.parse_rewrite(text)

    def test_enhancer_enabled_by_default_and_configurable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(prompt_enhancer.enabled())
        with patch.dict(os.environ, {'PROMPT_ENHANCER_ENABLED': 'false'}):
            self.assertFalse(prompt_enhancer.enabled())
        with patch.dict(os.environ, {'PROMPT_ENHANCER_ENABLED': 'invalid'}), self.assertRaises(ValueError):
            prompt_enhancer.enabled()

    def test_dimensions_and_conflicts(self):
        self.assertEqual((validate(PAYLOAD)['width'], validate(PAYLOAD)['height']), (1024, 1024))
        self.assertEqual(validate({**PAYLOAD, 'size': '2752x1536'})['height'], 1536)
        for extra in ({'stream': True}, {'n': 2}, {'seed': True}, {'size': '2048x2048', 'aspect_ratio': '16:9'}, {'quality': 'high'}):
            with self.subTest(extra=extra), self.assertRaises(InputError):
                validate({**PAYLOAD, **extra})

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
    def test_pipeline_passes_enhanced_prompt_to_diffusion(self):
        calls, phases = [], []
        class Generator:
            def __init__(self, device): pass
            def manual_seed(self, seed): return self
        class Diffusion:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(images=[Image.new('RGB', (2, 2))])
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.torch = SimpleNamespace(Generator=Generator, cuda=SimpleNamespace(empty_cache=lambda: None))
        pipeline.pipe = Diffusion()
        pipeline.steps = 40
        pipeline.enhancer_enabled = True
        with patch('service.prompt_enhancer.enhance', return_value='Prompt melhorado') as enhancer:
            pipeline.generate(validate(PAYLOAD), lambda *p: phases.append(p))
        enhancer.assert_called_once_with(PAYLOAD['prompt'], [])
        self.assertEqual(calls[0]['prompt'], 'Prompt melhorado')
        self.assertIn(('enhancing_prompt', None, None), phases)
        pipeline.enhancer_enabled = False
        with patch('service.prompt_enhancer.enhance') as enhancer:
            pipeline.generate(validate(PAYLOAD), lambda *p: None)
        enhancer.assert_not_called()
        self.assertEqual(calls[-1]['prompt'], PAYLOAD['prompt'])
        pipeline.enhancer_enabled = True
        reference = Image.new('RGB', (3, 3))
        values = validate({**PAYLOAD, 'input_references': [{'type': 'image_url', 'image_url': {'url': 'https://example.com/ref.png'}}]})
        with patch('service.load_reference', return_value=reference), patch('service.prompt_enhancer.enhance', return_value='Edição aprimorada') as enhancer:
            pipeline.generate(values, lambda *p: None)
        enhancer.assert_called_once_with(PAYLOAD['prompt'], [reference])
        self.assertEqual(calls[-1]['prompt'], 'Edição aprimorada')
        self.assertIs(calls[-1]['image'], reference)

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

class LlamaEnhancerTests(unittest.TestCase):
    def test_process_is_stopped_after_generation_failure(self):
        class Process:
            returncode = None
            def __init__(self, *args, **kwargs): self.stopped = False
            def poll(self): return None if not self.stopped else 0
            def terminate(self): self.stopped = True
            def wait(self, timeout): return 0
        class Pool:
            def __init__(self, **kwargs): pass
            def clear(self): pass
            def request(self, method, url, **kwargs):
                if method == 'GET': return SimpleNamespace(status=200)
                return SimpleNamespace(status=500, data=b'error')
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'file'
            file.write_text('system', encoding='utf-8')
            fake_hub = SimpleNamespace(hf_hub_download=lambda *args, **kwargs: str(file))
            process = Process()
            with patch.dict(sys.modules, {'huggingface_hub': fake_hub}), patch('prompt_enhancer.subprocess.Popen', return_value=process), patch('prompt_enhancer.urllib3.PoolManager', Pool), patch('prompt_enhancer._port', return_value=12345):
                with self.assertRaisesRegex(RuntimeError, 'HTTP 500'):
                    prompt_enhancer.enhance('x', [])
            self.assertTrue(process.stopped)

    def test_gguf_modes_cleanup_and_multimodal_payload(self):
        downloads, commands, requests, processes = [], [], [], []
        class Process:
            returncode = None
            def __init__(self, command, **kwargs):
                commands.append(command); processes.append(self); self.stopped = False
            def poll(self): return None if not self.stopped else 0
            def terminate(self): self.stopped = True
            def wait(self, timeout): return 0
        class Pool:
            def __init__(self, **kwargs): pass
            def clear(self): pass
            def request(self, method, url, **kwargs):
                if method == 'GET': return SimpleNamespace(status=200)
                requests.append(json.loads(kwargs['body']))
                return SimpleNamespace(status=200, data=json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': '<think>x</think>{"rewritten_prompt":"melhorado"}'}}]}).encode())
        with tempfile.TemporaryDirectory() as directory:
            prompt_file = Path(directory) / 'system_prompt.txt'
            prompt_file.write_text('instrução', encoding='utf-8')
            def download(repo, filename, revision):
                downloads.append((repo, filename, revision))
                return str(prompt_file if filename == 'system_prompt.txt' else Path(directory) / filename)
            fake_hub = SimpleNamespace(hf_hub_download=download)
            with patch.dict(sys.modules, {'huggingface_hub': fake_hub}), patch('prompt_enhancer.subprocess.Popen', Process), patch('prompt_enhancer.urllib3.PoolManager', Pool), patch('prompt_enhancer._port', return_value=12345):
                self.assertEqual(prompt_enhancer.enhance('um gato', []), 'melhorado')
                self.assertEqual(prompt_enhancer.enhance('mude o fundo', [Image.new('RGB', (2, 2))]), 'melhorado')
        self.assertTrue(all(process.stopped for process in processes))
        self.assertIn('pe_t2i.mtp.Q4_K_M.gguf', downloads[0][1])
        self.assertTrue(any('pe_i2i.mtp.Q4_K_M.gguf' in item[1] for item in downloads))
        self.assertIn('--mmproj', commands[1])
        self.assertIn('--spec-type', commands[0])
        self.assertNotIn('--mmproj', commands[0])
        content = requests[1]['messages'][1]['content']
        self.assertTrue(content[0]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(content[1]['text'], 'mude o fundo')

if __name__ == '__main__': unittest.main()
