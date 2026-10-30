import base64
import io
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import api
from service import InputError, MODEL, QueueManager, fetch_url, load_reference, validate

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

if __name__ == '__main__': unittest.main()
