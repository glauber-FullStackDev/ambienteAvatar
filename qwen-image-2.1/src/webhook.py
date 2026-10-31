"""Webhook delivery for asynchronous image jobs."""
import hashlib
import hmac
import json
import logging
import os
import time
from urllib.parse import urlsplit

import urllib3

LOG = logging.getLogger('qwen.webhook')


def validate_url(url):
    if not isinstance(url, str) or len(url) > 2048:
        return False
    try:
        parsed = urlsplit(url)
        allowed = {host.strip().lower() for host in os.getenv('WEBHOOK_ALLOWED_HOSTS', '').split(',') if host.strip()}
        return (parsed.scheme == 'https' and bool(parsed.hostname) and not parsed.username and not parsed.password
                and not parsed.fragment and parsed.netloc.lower() in allowed and parsed.port != 0)
    except ValueError:
        return False


def deliver(url, payload):
    body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'User-Agent': 'qwen-image-webhook/1.0'}
    secret = os.getenv('WEBHOOK_SECRET', '')
    if secret:
        headers['X-Webhook-Signature'] = 'sha256=' + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    retries = max(1, int(os.getenv('WEBHOOK_RETRIES', '5')))
    timeout = float(os.getenv('WEBHOOK_TIMEOUT', '10'))
    http = urllib3.PoolManager(timeout=urllib3.Timeout(total=timeout), retries=False)
    try:
        for attempt in range(retries):
            try:
                response = http.request('POST', url, body=body, headers=headers, redirect=False, preload_content=False)
                status = response.status
                response.close()
                if 200 <= status < 300:
                    return True
                LOG.warning('Webhook job %s: HTTP %s', payload['job_id'], status)
            except Exception:
                LOG.warning('Webhook job %s: falha na entrega', payload['job_id'], exc_info=True)
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 16))
        return False
    finally:
        http.clear()
