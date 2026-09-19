# LTX 2.3 IA2V Personal LoRA — Vast.ai Serverless (PyWorker)

Gera uma imagem independente para o workflow **LTX 2.3 IA2V + LoRA pessoal**
rodando no **serverless da Vast.ai**, como evolução do endpoint Queue do
Runpod (`../ltx-2.3-serverless/`).

O worker recebe URLs pré-assinadas de imagem e áudio no MinIO, executa o
workflow e devolve URLs pré-assinadas para o MP4 e o último frame PNG. Cada
job é **correlacionado a um projeto** (`project_id`) e, ao terminar, dispara um
**webhook** para o backend do projeto.

## Arquitetura

```
Cliente (backend/FlowScript)
  └─ SDK vastai: endpoint.request("/submit", payload, cost=…)
       └─ PyWorker (worker.py, porta 3000)  → roteado pelo engine Vast
            └─ Model server FastAPI (porta MODEL_SERVER_PORT, padrão 18080)
                 ├─ JobManager (fila/estados em memória, GPU única)
                 ├─ ComfyUI 127.0.0.1:8188 (inputs → workflow → output)
                 ├─ MinIO: {MINIO_OUTPUT_PREFIX}/{project_id}/{job_id}/…
                 └─ Webhook ao concluir/falhar (HMAC + retries)
```

O PyWorker provém do repositório dedicado (opção A):

| Componente | Onde vive |
| --- | --- |
| ComfyUI + nós + modelos LTX + FastAPI/engine | imagem `ambienteavatar-ltx23-serverless-vast` |
| `worker.py` + `requirements.txt` (interface com Vast) | repo `glauber-FullStackDev/ltx23-vast-pyworker`, espelhado em `pyworker/` para testes |

## Pré-requisitos

- Conta Vast.ai com créditos e **API key** (SDK `pip install vastai`).
- GPU de **48 GB** (A6000, A40, L40, RTX 6000 Ada) filtrada no workergroup.
- Acesso a `ghcr.io/glauber-fullstackdev` para publicar a imagem.
- MinIO acessível publicamente pelos workers Vast via HTTPS.
- Modelos LTX não vêm na imagem; o primeiro boot baixa só o que o IA2V Personal
  LoRA usa e persiste em um **volume Vast** montado em `/vast-volume`
  (`MODEL_VOLUME_ROOT`), evitando novo download após scale-to-zero.

## Build e publicação

```bash
docker buildx build --platform linux/amd64 \
  -f ltx-2.3-serverless-vast/Dockerfile \
  -t ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23-serverless-vast:v1 \
  --push .
```

A imagem não usa `runpod`; o CMD sobe ComfyUI + o model server FastAPI. O
`worker.py` do PyWorker vem do `PYWORKER_REPO` (repositório dedicado), com
fallback para o mirror embutido em `/opt/serverless-vast/pyworker` (útil em
testes locais e caso o clone falhe).

## Repositório do PyWorker (opção A)

Crie um repositório público mínimo **`glauber-FullStackDev/ltx23-vast-pyworker`**
com a raiz contendo:

```
worker.py           # espelho de ltx-2.3-serverless-vast/pyworker/worker.py
requirements.txt    # vastai
```

Aponte `PYWORKER_REPO` para ele no workergroup. O mecanismo oficial da Vast
clona o repo, instala `requirements.txt` e roda `python worker.py`.

## Criar o endpoint no Vast

Use `vast/endpoint-config.example.json` como referência (painel **Serverless**,
+ Endpoint, Advanced setup):

- `min_workers`/`max_workers`: `1`/`1` (GPU de 48 GB não escala horizontal);
- `min_load`: `1`; `inactivity_timeout`: ~600 s para permitir scale-to-zero;
- GPU: `gpu_ram >= 48`, `cuda_max_good >= 12.8`, disco `>= 100 GB`;
- Registrar `PYWORKER_REPO` e as variáveis abaixo.

> Ajuste `min_workers` para `0` só depois de validar o cold start; o boot baixa
> modelos e pode levar vários minutos na primeira vez.

## Secrets e variáveis do workergroup

Cadastre no painel/account (nunca em Git nem no agente):

| Variável | Obrigatória | Uso |
| --- | --- | --- |
| `PYWORKER_REPO` | Sim | `https://github.com/glauber-FullStackDev/ltx23-vast-pyworker` |
| `MINIO_ENDPOINT` | Sim | URL base S3-compatible, ex.: `https://minio.example.com` |
| `MINIO_BUCKET` | Sim | Bucket privado de entrada e saída |
| `MINIO_REGION` | Não | Padrão `us-east-1` |
| `MINIO_ACCESS_KEY` | Sim | Chave com permissão de gravar e assinar leitura |
| `MINIO_SECRET_KEY` | Sim | Segredo da chave MinIO |
| `MINIO_OUTPUT_PREFIX` | Não | Padrão `ltx-ia2v/results` |
| `MINIO_PRESIGN_EXPIRES_SECONDS` | Não | Padrão `86400` (24 h) |
| `MINIO_ALLOWED_HOST` | Não | Host permitido para URLs de entrada |
| `WEBHOOK_URL` | Não* | URL padrão notificada ao concluir/falhar |
| `WEBHOOK_SECRET` | Não | Assina o payload (`X-Webhook-Signature: sha256=…`) |
| `WEBHOOK_ALLOWED_HOSTS` | Não | Allow-list de hosts para `webhook.url` por job (anti-SSRF) |
| `WEBHOOK_RETRIES` | Não | Tentativas com backoff; padrão `5` |
| `WEBHOOK_STARTED` | Não | `1` para também notificar `job.started` |
| `PERSIST_JOB_STATUS` | Não | `1` (padrão) persiste `status.json` no MinIO |
| `MODEL_VOLUME_ROOT` | Não | Caminho do volume Vast; padrão `/vast-volume` |
| `BENCHMARK_RUNS` | Não | Benchmark do PyWorker; padrão `1` |
| `HF_TOKEN` | Não | Só se o Hugging Face exigir autenticação |

\* Sem `WEBHOOK_URL`, jobs podem mandar `webhook.url` por request (validado
contra `WEBHOOK_ALLOWED_HOSTS`).

Os nomes `S3_ENDPOINT`, `S3_BUCKET`, `S3_REGION`, `S3_ACCESS_KEY_ID`,
`S3_SECRET_ACCESS_KEY` também são aceitos (integração FlowScript).

## Chamar a API (SDK Vast)

O endpoint é **assíncrono**: `/submit` responde `202` com `job_id`; o render
roda em background, o backend do projeto pode assimilar o estado via **webhook**,
e o cliente pode acompanhar com `/status` através de uma sessão.

```python
import asyncio
from vastai import Serverless

async def main():
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name="ltx23-ia2v-personal-lora-vast")
        # sessão prende o job ao mesmo worker enquanto durar o render
        async with await endpoint.session(cost=100, lifetime=120) as session:
            submitted = await session.request("/submit", {
                "project_id": "uuid-do-projeto",
                "image_url": "https://minio.example.com/…/avatar.png?X-Amz-...",
                "audio_url": "https://minio.example.com/…/fala.wav?X-Amz-...",
                "prompt": "glauberavatar speaking naturally to camera",
                "job_id": None,  # opcional; sem ele o worker gera um
            }, cost=100)
            print(submitted["response"])  # {"job_id","project_id","status":"queued"}
asyncio.run(main())
```

Parâmetros opcionais idênticos ao Runpod: `width`, `height` (múltiplos de 32,
padrão `704x1280`), `duration_seconds` (≥1 e ≤30), `fps`, `audio_start_seconds`,
`seed`, `lora_strength`, `image_strength` e `webhook` (`{"url", "extra_params"}`).

### Webhook de conclusão

`POST` JSON assinado com `X-Webhook-Signature` (se `WEBHOOK_SECRET`):
`job.started` (opcional), `job.completed`, `job.failed`:

```json
{
  "event": "job.completed",
  "job_id": "job-gerado-ou-do-chamador",
  "project_id": "uuid-do-projeto",
  "status": "completed",
  "status_url": "https://minio.example.com/…/{project_id}/{job_id}/status.json?…",
  "video_url": "https://minio.example.com/…/{project_id}/{job_id}/video.mp4?…",
  "last_frame_url": "https://minio.example.com/…/{project_id}/{job_id}/last_frame.png?…",
  "parameters": {"width": 704, "height": 1280, "duration_seconds": 18, "fps": 24,
                 "audio_start_seconds": 0, "seed": 123, "lora_strength": 1.0,
                 "image_strength": 0.7},
  "execution_seconds": 145.2,
  "timestamp": "2026-09-19T00:00:00Z",
  "error": null,
  "origin": "valor-de-extra_params"
}
```

`extra_params` do chamador entram no payload, mas as chaves reservadas
(`event`, `job_id`, `project_id`, `status`, `timestamp`, etc.) nunca são
sobrescritas. Com `PERSIST_JOB_STATUS=1`, `status.json` (com state + URLs) é
gravado no MinIO a cada transição — é a trilha durável para o backend mesmo se
o webhook falhar.

## Verificação local

Os testes não exigem GPU nem dependências de terceiros:

```bash
python3 ltx-2.3-serverless-vast/tests/test_validate_webhook.py
python3 ltx-2.3-serverless-vast/tests/test_jobs_manager.py
python3 ltx-2.3-serverless-vast/tests/test_serverless_contract.py
```

Antes de produção, faça um smoke test no endpoint com imagem e áudio curtos,
confira o MP4, o PNG, a expiração das URLs e o webhook recebido.

## Diagnóstico

- **Worker preso em "Loading":** cheque o log do PyWorker e o `model.log`
  (`MODEL_LOG_FILE`); a linha `Model server ready` dispara o benchmark.
- **Worker em estado erro:** o engine reinicia; confira `boto3`/MinIO e a GPU
  de 48 GB.
- **Webhook não chega:** confira `WEBHOOK_URL`, `WEBHOOK_ALLOWED_HOSTS` e o
  `status.json` do job no MinIO.
- **Cold start lento:** modelos baixam no primeiro boot; monte o volume
  `/vast-volume` para persistir em `models/`.
- **Sem URL final:** valide as credenciais e a política `PutObject`/`GetObject`
  do bucket.