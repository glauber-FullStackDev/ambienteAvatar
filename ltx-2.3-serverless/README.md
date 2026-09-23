# LTX 2.5 IA2V — Runpod Serverless

Este diretório gera uma imagem independente para o workflow **LTX 2.5 IA2V
(padrão oficial A2V two-stage destilado) + LoRA pessoal (opcional)**.
Ele não altera `../ltx-2.3/` nem substitui a imagem usada no Vast.ai.

O worker recebe URLs pré-assinadas de imagem e áudio no MinIO, executa o
workflow e devolve URLs pré-assinadas para o MP4 e o último frame PNG.

### Arquitetura do pipeline (v2.5)

- **First frame anchor** — `LTXVImgToVideoInplace` fixa o frame 0
  (`first_frame_strength`, default `1.0`) na passada base e no refine.
- **LoRA de identidade** — `glauberavatar.safetensors` único LoRA da cadeia
  (`lora_strength`, default `0.7`; envie `0` para desligar).
- **Áudio / lipsync** — o áudio dirigente é codificado e congelado
  (noise mask zero); o LTX 2.5 gera lipsync nativo a partir dele.
- **Geração** — duas passadas destiladas (8 + 3 passos, CFG 1), base em W/2×H/2
  e refine com upscaler latente x2, conforme o padrão oficial A2V
  (`refine_sigmas` oficial `"0.85, 0.7250, 0.4219, 0.0"`).

> **Restauração (estado funcional).** O comportamento validado é o do commit
> `f9b67c2`: int8 + 8+3 passos + prompt "speaks expressively and clearly".
> Alterações de prompt que peçam fala contida ("calm", "sem expressões
> exageradas") suprimem a articulação e parecem dessincronia — o wording do
> prompt é sensível neste modelo. O experimento de refine de 9 passos
> (`"0.925, 0.85, 0.75, 0.646, 0.525, 0.403, 0.281, 0.156, 0.0"`) também
> degradou o lipsync e foi revertido; só tente via `refine_sigmas` por job,
> nunca como default.
- **Prompt enhancer** — desligado por padrão (`enable_prompt_enhance`);
  quando ligado usa o Gemma 4 E2B.
- **Decode** — `VAEDecodeTiled` com `decode_tile_size` configurável
  (default `512`; `1024` reduz emendas de tile em faces — testar se aparecerem
  manchas/regiões sujas no rosto).

> Nota: a versão anterior com IC-LoRA Ingredients (guia visual persistente e
> keyframe de referência) foi revertida — sem tokens de guia o IC-LoRA
> suprimia o movimento, e com guia forte o vídeo "voltava" ao frame inicial
> (efeito elástico). O histórico permanece no git.

O prompt padrão (usado quando o job não envia `prompt`) comanda apenas
movimento e performance — composição, cenário e enquadramento ficam sob
responsabilidade do first frame.

### DiT BF16 (experimento pausado)

A imagem publica dois templates do mesmo workflow: o padrão usa o transformer
destilado **INT8 ConvRot** e o alternativo usa o checkpoint **BF16**
(`ltx-2.5-22b-distilled-transformer-bf16.safetensors`, 39,1 GiB em VRAM).
O bootstrap detecta o checkpoint pelo nome do template em `WORKFLOW_PATH`
(e baixa apenas ele): com o template padrão o worker baixa só o INT8
(~20 GiB); com `WORKFLOW_PATH=/opt/defaults/workflows/video_ltx2_5_ia2v_bf16_api.json`
baixa só o BF16 (39,1 GiB). Para forçar a escolha, defina
`LTX25_UNET_CHECKPOINT=int8|bf16`.

> **Não defina `WORKFLOW_PATH` para o template bf16 em endpoints de
> produção.** O endpoint de exemplo (`runpod/endpoint-config.example.json`)
> usa o baseline int8 sem sobrescrita de `WORKFLOW_PATH`. O experimento BF16
> está pausado até o baseline int8 voltar a validar.

Orçamento de VRAM estimado em 48 GB: o pico ocorre na passada de refine
(704×1280) — DiT 39,1 GiB + LoRA + ativações ≈ 44–48 GiB, na fronteira do
utilável em A6000/A40. Se o ComfyUI partir para offload parcial de pesos, o
job conclui mais lentamente; o INT8 permanece como padrão de produção até a
comparação ser validada.

A instrumentação de VRAM é opt-in: defina `LOG_VRAM_PEAK=1` no endpoint para
que o worker amostra `/system_stats` durante o job e reporte no retorno do
job:

- `peak_vram_used_gb` — pico de VRAM ocupada durante o job (GiB);
- `vram_device` — nome da GPU reportada pelo ComfyUI.

## Pré-requisitos

- Conta Runpod, um endpoint Queue e uma GPU de 48 GB (A6000 ou A40).
- O bootstrap baixa apenas o checkpoint DiT usado pelo `WORKFLOW_PATH`
  (int8 ~20 GiB ou bf16 39,1 GiB) — Network Volume de 60 GB atende o modo
  int8; para alternar entre int8 e bf16 no mesmo volume, use 110 GB.
- Acesso a `ghcr.io/glauber-fullstackdev` para publicar a imagem.
- MinIO acessível publicamente pelos workers Runpod, via HTTPS recomendado.
- Bucket privado, por exemplo `ltx-serverless`.
- URLs pré-assinadas de entrada que possam ser lidas pelo worker.
- Docker com suporte a build Linux/amd64.
- `HF_TOKEN` no endpoint com termos de `Lightricks/LTX-2.5` aceitos.

Os modelos LTX 2.5 não são incorporados na imagem. No primeiro worker ele
baixa somente o set 2.5 (transformer destilado int8, Gemma 4 12B + E2B,
VAEs de vídeo e áudio, upscaler x2 e o IC-LoRA Ingredients).
Para não baixar os modelos novamente após escala para zero, anexe um Network
Volume de pelo menos 60 GB (100 GB recomendado). A imagem detecta o volume
montado pelo Runpod em `/runpod-volume` e persiste automaticamente os modelos
em `/runpod-volume/models`. FlashBoot reduz a retomada de um worker pausado,
mas não substitui o volume persistente.

## Build e publicação

Execute na raiz do repositório:

```bash
docker buildx build --platform linux/amd64 \
  -f ltx-2.3-serverless/Dockerfile \
  -t ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23-serverless:v2.5-bf16 \
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
| `S3_CONNECT_TIMEOUT_SECONDS` | Não | Padrão `20`; timeout de conexão com o storage S3 |
| `S3_READ_TIMEOUT_SECONDS` | Não | Padrão `600`; timeout para uploads de resultado |
| `S3_UPLOAD_ATTEMPTS` | Não | Padrão `3`; tentativas para enviar MP4/PNG ao storage |
| `HF_TOKEN` | Não | Use apenas se o Hugging Face exigir autenticação para download |

Os nomes equivalentes `S3_ENDPOINT`, `S3_BUCKET`, `S3_REGION`,
`S3_ACCESS_KEY_ID` e `S3_SECRET_ACCESS_KEY` também são aceitos para facilitar
a integração com o FlowScript. Prefira um único conjunto de nomes para evitar
configurações divergentes.

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
      "width": 704,
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
opcionais. Largura e altura precisam ser divisíveis por 32; o padrão é
`704x1280`. A API mantém fixos
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
- **ComfyUI recusa o workflow:** a resposta inclui o detalhe da validação no
  status do job; confira também se a imagem publicada é a `:v2.5-bf16` ou posterior.
