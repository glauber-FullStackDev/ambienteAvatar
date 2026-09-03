# LTX 2.3 IA2V Personal LoRA — Runpod Serverless

Este diretório gera uma imagem independente para o workflow **LTX 2.3 IA2V +
LoRA pessoal**. Ele não altera `../ltx-2.3/` nem substitui a imagem usada no
Vast.ai.

O worker recebe URLs pré-assinadas de imagem e áudio no MinIO, executa o
workflow e devolve URLs pré-assinadas para o MP4 e o último frame PNG.

## Pré-requisitos

- Conta Runpod, um endpoint Queue e uma GPU de 48 GB (A6000 ou A40).
- Acesso a `ghcr.io/glauber-fullstackdev` para publicar a imagem.
- MinIO acessível publicamente pelos workers Runpod, via HTTPS recomendado.
- Bucket privado, por exemplo `ltx-serverless`.
- URLs pré-assinadas de entrada que possam ser lidas pelo worker.
- Docker com suporte a build Linux/amd64.

O modelo LTX 2.3 não é incorporado na imagem. No primeiro worker ele baixa
somente os pesos necessários ao IA2V Personal LoRA (LTX 2.5 é excluído).
Use o cache de modelos/FlashBoot do Runpod para reduzir os cold starts
subsequentes.

## Build e publicação

Execute na raiz do repositório:

```bash
docker buildx build --platform linux/amd64 \
  -f ltx-2.3-serverless/Dockerfile \
  -t ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23-serverless:v1 \
  --push .
```

Não use a tag `:vast`: ela pertence ao ambiente ComfyUI interativo existente.

## Criar o endpoint Runpod

No Runpod, crie um endpoint **Serverless / Queue** usando a imagem publicada.
Use como ponto de partida
[`runpod/endpoint-config.example.json`](runpod/endpoint-config.example.json):

- `min workers`: `0`;
- `max workers`: `1`;
- GPU: A6000 ou A40, ambas de 48 GB;
- `idle timeout`: 30 segundos;
- `execution timeout`: 7200 segundos.

O worker não expõe a porta 8188. ComfyUI fica disponível apenas dentro do
container para o handler.

## Secrets e variáveis do endpoint

Cadastre estes valores em **Serverless endpoint → Environment Variables** no
Runpod. Nunca os adicione a Dockerfile, Git ou ao agente.

| Variável | Obrigatória | Uso |
| --- | --- | --- |
| `MINIO_ENDPOINT` | Sim | URL base S3-compatible, por exemplo `https://minio.example.com` |
| `MINIO_BUCKET` | Sim | Bucket privado de entrada e saída |
| `MINIO_REGION` | Não | Padrão `us-east-1` |
| `MINIO_ACCESS_KEY` | Sim | Chave com permissão de gravar resultados e assinar leitura |
| `MINIO_SECRET_KEY` | Sim | Segredo da chave MinIO |
| `MINIO_OUTPUT_PREFIX` | Não | Padrão `ltx-ia2v/results` |
| `MINIO_PRESIGN_EXPIRES_SECONDS` | Não | Padrão `86400` (24 h) |
| `MINIO_ALLOWED_HOST` | Não | Host permitido para URLs de entrada; padrão é o host de `MINIO_ENDPOINT` |
| `HF_TOKEN` | Não | Use apenas se o Hugging Face exigir autenticação para download |

O cliente/agente não recebe `MINIO_ACCESS_KEY` ou `MINIO_SECRET_KEY`. Ele deve
obter URLs pré-assinadas de upload do seu backend/MinIO e, depois do upload,
passá-las ao Runpod.

## Chamar a API

Submeta um job assíncrono. A chave Runpod permanece no chamador seguro:

```bash
curl --request POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{
    "input": {
      "image_url": "https://minio.example.com/ltx-serverless/input/avatar.png?...",
      "audio_url": "https://minio.example.com/ltx-serverless/input/fala.wav?...",
      "prompt": "glauberavatar speaking naturally to camera",
      "width": 720,
      "height": 1280,
      "duration_seconds": 18,
      "fps": 24,
      "audio_start_seconds": 0,
      "seed": 304473763956052,
      "lora_strength": 1.0,
      "image_strength": 0.7
    }
  }'
```

Parâmetros obrigatórios: `image_url`, `audio_url` e `prompt`. Os demais são
opcionais. Largura e altura precisam ser divisíveis por 32. A API mantém fixos
checkpoint, LoRAs técnicos e modelo de upscale; apenas a força do LoRA pessoal
e da imagem são ajustáveis.

O retorno inicial traz o `id` do job. Consulte até concluir:

```bash
curl --header "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID"
```

No estado concluído, `output.video_url` e `output.last_frame_url` são URLs
temporárias do MinIO; `output.execution_seconds` informa o tempo total do job
no worker.

## Verificação local

O teste de estrutura não baixa modelos nem inicia GPU:

```bash
python3 ltx-2.3-serverless/tests/test_workflow_api.py
```

Antes de produção, faça um smoke test no endpoint com uma imagem e áudio
curtos. Confira a reprodução do MP4, o PNG do último frame, a expiração das
URLs e os logs do job no Runpod.

## Diagnóstico

- **Worker não inicia:** confira o log `comfyui-serverless.log`, acesso ao
  Hugging Face e memória de GPU de 48 GB.
- **Entrada recusada:** o host de `image_url`/`audio_url` deve corresponder a
  `MINIO_ALLOWED_HOST`; confirme também a assinatura e expiração da URL.
- **Não há URL final:** verifique as credenciais e a política `PutObject`/
  `GetObject` do bucket.
- **Job expira:** aumente `execution timeout` no endpoint, não `min workers`.
