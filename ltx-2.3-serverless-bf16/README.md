# LTX 2.5 IA2V BF16 — Runpod Serverless (experimento isolado)

Imagem **autônoma** do worker LTX 2.5 IA2V com o transformer destilado
**BF16** (`ltx-2.5-22b-distilled-transformer-bf16.safetensors`, 39,1 GiB em
VRAM). Não compartilha código, Dockerfile, workflow CI nem download de
modelos com `../ltx-2.3-serverless/` (int8, produção) — as únicas exceções
são o binário LFS imutável do LoRA pessoal, copiado de `../ltx-2.3/assets/`,
e os custom nodes upstream clonados nas duas imagens.

Cada árvore tem seu próprio pipeline CI: push em `ltx-2.3-serverless-bf16/**`
publica somente `:v2.5-bf16`; push em `ltx-2.3-serverless/**` publica
somente `:v2.5`. Alterações no download/modelos deste lado não afetam a
imagem int8 e vice-versa.

## O que esta imagem faz diferente da int8

| | `:v2.5` (int8) | `:v2.5-bf16` (esta) |
|---|---|---|
| Pasta | `ltx-2.3-serverless/` | `ltx-2.3-serverless-bf16/` |
| Workflow CI | `publish-ltx23-serverless-image.yml` | `publish-ltx23-serverless-bf16-image.yml` |
| DiT | INT8 ConvRot (~20 GiB) | **BF16 (39,1 GiB)** |
| Download no 1º worker | só int8 | **só bf16** |
| Template baked | `video_ltx2_5_ia2v_api.json` | `video_ltx2_5_ia2v_bf16_api.json` |
| Env necessária no endpoint | nenhuma | **nenhuma** (`WORKFLOW_PATH` e checkpoint já baked) |
| Text encoder / VAEs / upscaler | idênticos | idênticos |

## Orçamento de VRAM em 48 GB

O pico ocorre na passada de refine (704×1280): DiT 39,1 GiB + LoRA + ativações
≈ 44–48 GiB — na fronteira do utilizável em A6000/A40. Se o ComfyUI partir
para offload parcial de pesos, o job conclui mais lentamente. Durações de 30 s
(577 frames) podem estourar; use int8 (`:v2.5`) para clipes longos.

A instrumentação de VRAM é opt-in: defina `LOG_VRAM_PEAK=1` no endpoint para
que o retorno do job inclua `peak_vram_used_gb` e `vram_device`.

## Build e publicação

O pipeline CI publica automaticamente no push (paths desta pasta). Manual:

```bash
docker buildx build --platform linux/amd64 \
  -f ltx-2.3-serverless-bf16/Dockerfile \
  -t ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23-serverless:v2.5-bf16 \
  --push .
```

## Endpoint Runpod

Use [`runpod/endpoint-config.example.json`](runpod/endpoint-config.example.json)
— não precisa de `WORKFLOW_PATH` nem `LTX25_UNET_CHECKPOINT`; a variante está
baked na imagem. Rollback do experimento: aponte o endpoint de volta para a
imagem `:v2.5` (int8). Network Volume recomendado: 60 GB (só o set bf16).

## Verificação local

```bash
python3 ltx-2.3-serverless-bf16/tests/test_workflow_api.py
python3 ltx-2.3-serverless-bf16/tests/test_serverless_contract.py
```

A API do worker é idêntica à da imagem int8 (mesmos parâmetros, mesmos
retornos, piso de 5 s de áudio) — consulte o
[README int8](../ltx-2.3-serverless/README.md) para a referência completa.

## Isolamento

Este pipeline é acionado apenas por mudanças em `ltx-2.3-serverless-bf16/**`.
