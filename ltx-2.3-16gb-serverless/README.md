# LTX 2.3 GGUF 16 GB — Worker local (API HTTP + Webhook)

Imagem Docker para rodar **em máquina local (GPU 16 GB)** o workflow LTX 2.3
IA2V Personal LoRA na variante **GGUF Q4_K_M** de `../ltx-2.3-16gb/`, com uma
**API HTTP assíncrona** e **feedback por webhook**.

- **Esqueleto** (handler, validação, ciclo de vida do ComfyUI, upload MinIO):
  `../ltx-2.3-serverless/` (worker Runpod).
- **Contrato de API/webhook**: `../ltx-2.3-serverless-vast/` (`POST /submit`,
  `POST /status`, eventos `job.started`/`job.completed`/`job.failed`,
  assinatura HMAC, espelho de status no MinIO).
- **Modelos e workflow**: `ltx-2.3-16gb/` (GGUF ~26 GiB baixados no boot para
  `$MODELS_ROOT`, nunca assados na imagem).

Nenhum dos paths de origem é alterado.

## Diferenças contratuais vs worker Vast (`ltx-2.3-serverless-vast`)

| Campo | Vast | Esta variante |
|---|---|---|
| `prompt` | obrigatório | **opcional** — sem prompt, usa o `DEFAULT_PROMPT` (token `glauberavatar`, trava de objetos, câmera estática) |
| `lora_strength` | 1.0 (máx 2) | **0.7** (máx 1) — default do worker Runpod |
| `image_strength` | 0.7 | **1.0** — default do worker Runpod |
| `duration_seconds` | mín 1 | **mín 5** (piso de lipsync) |
| `enable_prompt_enhance` | — | novo, **default `false`** (job pode enviar `true`) |
| `decode_tile_size` / `temporal_size` / `temporal_overlap` | — | novos, defaults **512 / 48 / 16** (perfil validado em 16 GB) |

Tudo o mais (envelope do webhook, `status_url`, HMAC, allowlist, retries,
`status.json` no MinIO) é idêntico ao contrato Vast.

## Como rodar

```bash
docker run -d --gpus all -p 8000:8000 \
  -v /dados/models:/data \
  -e MINIO_ENDPOINT=https://minio.example.com \
  -e MINIO_BUCKET=ltx-serverless \
  -e MINIO_ACCESS_KEY=... -e MINIO_SECRET_KEY=... \
  -e MINIO_ALLOWED_HOST=minio.example.com \
  -e API_TOKEN=troque-isto -e WEBHOOK_SECRET=troque-isto \
  ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23-16gb-serverless:latest
```

No primeiro boot o worker baixa os ~26 GiB de modelos para `$MODELS_ROOT`
(volume montado) antes de expor jobs — acompanhe por `GET /health`
(`startup_done` / `comfyui`).

## API

### `POST /submit` → 202

```json
{
  "project_id": "meu-projeto",
  "job_id": "opcional (uuid hex por default)",
  "image_url": "https://minio.example.com/input/avatar.png?...",
  "audio_url": "https://minio.example.com/input/fala.wav?...",
  "prompt": "opcional; ausente -> DEFAULT_PROMPT",
  "negative_prompt": "opcional (o negative do template 2.3 é fixo)",
  "webhook": { "url": "https://backend/hook", "extra_params": {"origin": "app"} },
  "width": 704, "height": 1280,          // %32
  "duration_seconds": 18.0,              // 5–30
  "fps": 24,                             // 8–30
  "audio_start_seconds": 0.0,            // 0–3600
  "seed": 0,                             // omitido = aleatório
  "lora_strength": 0.7,                  // 0–1
  "image_strength": 1.0,                 // 0–1
  "enable_prompt_enhance": false,        // default false; true liga o Gemma enhancer
  "decode_tile_size": 512,               // 256–1024
  "temporal_size": 48,                   // 16–4096
  "temporal_overlap": 16,                // 4+ e < temporal_size
  "base_sigmas": null, "refine_sigmas": null
}
```

Erros: `422` validação, `409` worker ocupado, `401` token inválido.

### `POST /status` (`{"job_id": "..."}`) e `GET /health`

## Camada de compatibilidade RunPod (backend `runpod-render.service`)

O backend de produção foi construído contra o proxy do RunPod serverless. As
rotas abaixo falam **esse protocolo** em paralelo ao contrato vast — nenhuma
das duas é afetada pela outra. Qualquer valor no segmento `{endpointId}` é
aceito (o worker não valida o id do endpoint).

| Backend RunPod | Rota aqui | Observação |
|---|---|---|
| `POST /{endpointId}/run` body `{"input": {...}, "webhook": "https://..."}` | idem | responde `200 {"id": "<jobId>", "status": "IN_QUEUE"}` |
| `POST /{endpointId}/runsync` | idem | espera até `RUNSYNC_WAIT_SECONDS` (default 85s) e responde o status; depois disso `IN_PROGRESS` |
| `GET /{endpointId}/status/{jobId}` | idem | `{"id", "status": IN_QUEUE\|IN_PROGRESS\|COMPLETED\|FAILED, "output": ..., "error": ...}`; 404 se desconhecido |
| `GET /{endpointId}/health` | idem | `{"jobs": {completed, failed, in_progress, queued}, "workers": {...}}` |

Detalhes de tradução:

- `webhook` chega como **string** no topo do corpo (protocolo RunPod) e é
  movido para `input.webhook.url` internamente.
- O backend RunPod não envia `project_id`; o worker usa o default
  `RUNPOD_DEFAULT_PROJECT_ID` (default `"default"`).
- O `output` (no status e no webhook) **replica o dict do worker
  `ltx-2.3-serverless`**: `{job_id, comfy_prompt_id, video_url, last_frame_url,
  image_sha, parameters{prompt, negative_prompt, duration_seconds, fps,
  audio_start_seconds, lora_strength, first_frame_strength, cfg,
  decode_tile_size, enable_prompt_enhance, base_sigmas, refine_sigmas},
  execution_seconds}` — zero mudança no backend.
- O webhook de conclusão é entregue no **formato RunPod**:
  `{"id": "<jobId>", "status": "COMPLETED"|"FAILED", "output": {...}, "error": null}`.
- Como o RunPod, jobs por essas rotas **não** recebem `job.started`.
- `API_TOKEN` continua valendo: configure o backend para enviar
  `Authorization: Bearer <API_TOKEN>` (o RunPod proxy também usa Bearer).
- Sem fila do RunPod: se o worker já está processando, `POST /run` responde
  **409** — o backend deve tratar/reintentar (o proxy RunPod enfileiraria).
- `project_id` do job (usado no caminho MinIO de saída) sai de
  `RUNPOD_DEFAULT_PROJECT_ID` quando o backend não envia.

## Webhook

Eventos `job.started` (se `WEBHOOK_STARTED=1`), `job.completed`, `job.failed`:

```json
{
  "event": "job.completed",
  "job_id": "...", "project_id": "...", "status": "completed",
  "status_url": "https://minio/.../status.json?...",
  "timestamp": "2026-10-26T12:00:00+00:00",
  "result": {
    "job_id": "...", "project_id": "...", "comfy_prompt_id": "...",
    "video_url": "https://minio/.../video.mp4?...",
    "last_frame_url": "https://minio/.../last_frame.png?...",
    "image_sha": "...",
    "parameters": { "width": 704, "height": 1280, "duration_seconds": 18.0,
      "fps": 24, "audio_start_seconds": 0.0, "seed": 42,
      "lora_strength": 0.7, "image_strength": 1.0,
      "enable_prompt_enhance": false, "decode_tile_size": 512,
      "temporal_size": 48, "temporal_overlap": 16 },
    "execution_seconds": 512.3
  },
  "error": null,
  "execution_seconds": 512.4,
  "parameters": { "width": 704, "height": 1280, "duration_seconds": 18.0, "fps": 24,
    "audio_start_seconds": 0.0, "seed": 42, "lora_strength": 0.7, "image_strength": 1.0,
    "enable_prompt_enhance": false, "decode_tile_size": 512, "temporal_size": 48, "temporal_overlap": 16 },
  "origin": "app"
}
```

Assinatura: header `X-Webhook-Signature: sha256=<HMAC-SHA256 do body>` quando
`WEBHOOK_SECRET` está definido. Hosts permitidos via `WEBHOOK_ALLOWED_HOSTS`
(fallback: host de `WEBHOOK_URL`). 5 tentativas com backoff. Com
`PERSIST_JOB_STATUS=1` o estado do job também é espelhado em `status.json` no
MinIO (fallback caso o webhook falhe).

## Variáveis de ambiente

| Variável | Obrigatória | Default | Função |
|---|---|---|---|
| `MINIO_ENDPOINT` (ou `S3_ENDPOINT`) | sim | — | endpoint S3/MinIO |
| `MINIO_BUCKET` (`S3_BUCKET`) | sim | — | bucket de resultados |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | sim | — | credenciais (`S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`) |
| `MINIO_REGION` | não | `us-east-1` | região S3 |
| `MINIO_OUTPUT_PREFIX` | não | `ltx-ia2v/results` | prefixo de saída |
| `MINIO_PRESIGN_EXPIRES_SECONDS` | não | `86400` | validade das URLs presigned |
| `MINIO_ALLOWED_HOST` | não | host do endpoint | allowlist de entrada |
| `MAX_INPUT_BYTES` | não | `104857600` | limite de download de input |
| `WEBHOOK_URL` | não | — | webhook global (job pode sobrescrever) |
| `WEBHOOK_SECRET` | não | — | ativa assinatura HMAC |
| `WEBHOOK_ALLOWED_HOSTS` | não | — | allowlist de hosts de webhook |
| `WEBHOOK_STARTED` | não | `0` | envia `job.started` |
| `WEBHOOK_RETRIES` / `WEBHOOK_TIMEOUT` | não | `5` / `10` | entrega do webhook |
| `PERSIST_JOB_STATUS` | não | `0` | espelha `status.json` no MinIO |
| `API_TOKEN` | não | — | Bearer token da API (sem env = API aberta) |
| `RUNPOD_DEFAULT_PROJECT_ID` | não | `default` | `project_id` para jobs do backend Runpod (que não envia) |
| `RUNSYNC_WAIT_SECONDS` | não | `85` | quanto `/{endpointId}/runsync` espera bloqueado |
| `MODELS_ROOT` | não | `/data` | volume persistente dos modelos |
| `LTX23_GGUF_QUANT` | não | `Q4_K_M` | quant do DiT (`Q5_K_M`, `Q6_K`) |
| `HF_TOKEN` | não | — | token HF (repos públicos não exigem) |
| `LOG_VRAM_PEAK` | não | `0` | loga pico de VRAM por job |
| `COMFY_TIMEOUT_SECONDS` | não | `21600` | timeout de execução por job |

## Parâmetros de decode (16 GB)

O `VAEDecodeTiled` (nó 316) sai da imagem com `512 / 64 / 48 / 16`
(tile/overlap/temporal/temporal_overlap) — perfil que eliminou o OOM e o
artefato de tremida temporal em GPU de 16 GB. `overlap` é fixo; os demais
aceitam override por job com mínimos seguros.

## Testes

```bash
python3 ltx-2.3-16gb-serverless/tests/test_workflow_api.py
python3 ltx-2.3-16gb-serverless/tests/test_serverless_contract.py
python3 ltx-2.3-16gb-serverless/tests/test_runpod_compat.py
```
