# Z-Image-Turbo — API e dashboard compatíveis com Qwen

Imagem Docker independente com `Tongyi-MAI/Z-Image-Turbo`, API HTTP síncrona, fila, webhook assíncrono e dashboard. O Qwen não é modificado. As duas imagens não devem ficar ligadas simultaneamente na mesma GPU de 16 GB: pare um container antes de iniciar o outro.

O checkpoint Z-Image-Turbo tem [licença Apache 2.0](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). O Prompt Enhancer, ativado por padrão, reutiliza checkpoints derivados do Qwen sujeitos à [Qwen Research License](https://huggingface.co/mozophe/Qwen-Image-2.1-PE-MTP-GGUF/blob/main/LICENSE), que exige licença separada para uso comercial. Essa restrição continua aplicável quando o enhancer está ligado.

## Iniciar no servidor

Requisitos: Linux/AMD64, Docker, NVIDIA Container Toolkit, GPU CUDA com BF16 e espaço para pesos e cache. A compatibilidade com a GPU de 16 GB e os tempos de geração e edição **ainda precisam ser validados**. Na primeira inicialização o serviço baixa o modelo; o enhancer baixa seu GGUF na primeira solicitação. Os arquivos ficam no volume `z-image-models`.

```bash
# Somente se o Qwen estiver rodando na mesma GPU:
docker stop qwen-image

cd z-image-turbo
export API_TOKEN='defina-um-token-forte'
export WEBHOOK_ALLOWED_HOSTS='api.exemplo.com'
export WEBHOOK_SECRET='defina-um-segredo-forte'
docker compose up -d --build
docker compose logs -f
```

O tunnel continua apontando para `http://localhost:8001`, a mesma porta do Qwen; a porta interna do container é `8000`. O dashboard está em `/dashboard`; `/health` retorna HTTP 503 enquanto o modelo carrega e 200 quando pronto. Pare o Qwen antes de iniciar o Z-Image para liberar a porta e a GPU. Em caso de problema de memória, confira RAM/VRAM e experimente desativar o enhancer via `PROMPT_ENHANCER_ENABLED=false`. O uso de offload sequencial é padrão e pode ser lento.

`API_TOKEN` vazio desativa autenticação. Configure-o e use HTTPS ao expor a API. O HTML do dashboard e `/health` são públicos, enquanto os dados do painel e as APIs usam Bearer quando o token está configurado.

## Publicação GHCR

O workflow `.github/workflows/publish-z-image-turbo.yml` roda testes e publica Linux/AMD64 somente quando um push para `main` altera `z-image-turbo/**`. A imagem será `ghcr.io/glauber-fullstackdev/ambienteavatar-z-image-turbo:latest`, com outra tag `sha-<SHA completo>`. A primeira publicação depende de enviar os arquivos à `main` e aguardar o workflow. Se o pacote for privado, autentique o servidor no GHCR.

```bash
docker pull ghcr.io/glauber-fullstackdev/ambienteavatar-z-image-turbo:latest
docker run -d --name z-image --gpus all --restart unless-stopped -p 8001:8000 \
  -v z-image-models:/data -e API_TOKEN="$API_TOKEN" -e CPU_OFFLOAD=sequential \
  -e WEBHOOK_ALLOWED_HOSTS="$WEBHOOK_ALLOWED_HOSTS" -e WEBHOOK_SECRET="$WEBHOOK_SECRET" \
  ghcr.io/glauber-fullstackdev/ambienteavatar-z-image-turbo:latest
docker logs -f z-image
```

## API assíncrona com webhook

A mesma rota e o mesmo envelope do Qwen são mantidos. O campo `model` aceita o nome próprio `tongyi/z-image-turbo` e o alias legado `qwen/qwen-image-2.1`. O alias não carrega o modelo Qwen: ele apenas permite manter a requisição do consumidor.

```bash
curl --max-time 30 http://localhost:8001/api/v1/images/async \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"tongyi/z-image-turbo","prompt":"Uma capivara lendo um livro","resolution":"1K","webhook":{"url":"https://api.exemplo.com/webhooks/imagem"}}'
```

Retorna HTTP 202 com `job_id`, `status: "queued"` e `status_url`. O serviço envia POST com `event: "job.completed"`, `job_id`, `status`, `result` e `error: null`; `result.data[0]` contém `b64_json` e `media_type: "image/png"`. Em falha envia `event: "job.failed"`, `result: null` e `error`. O recebedor precisa retornar 2xx e tratar duplicatas por `job_id`. Uma entrega pode chegar antes de o cliente processar o 202.

`webhook.url` deve usar HTTPS e corresponder a um dos hosts de `WEBHOOK_ALLOWED_HOSTS` (lista separada por vírgulas). `WEBHOOK_SECRET`, quando definido, assina o corpo bruto com HMAC-SHA256 no cabeçalho `X-Webhook-Signature: sha256=<hex>`. Destinos não devem redirecionar e precisam aceitar POSTs JSON de alguns megabytes. Uma falha de entrega é tentada novamente sem bloquear a GPU.

`GET /api/v1/images/jobs/{job_id}` permite consultar status, progresso, entrega e resultado. O resultado e a fila ficam em memória, limitados pelo histórico; um reinício os apaga e interrompe trabalhos e notificações. As imagens base64 ocupam RAM enquanto retidas.

## API síncrona e edição

`POST /api/v1/images` mantém o formato síncrono do Qwen. Pela Cloudflare Tunnel, use a rota assíncrona para evitar timeout durante a geração e a fila.

```bash
curl --max-time 3600 http://localhost:8001/api/v1/images \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen-image-2.1","prompt":"Uma capivara lendo um livro","resolution":"1K","aspect_ratio":"1:1","seed":42}'
```

Os campos `model`, `prompt`, `seed`, `resolution`, `aspect_ratio`, `size`, `n`, `output_format`, `stream` e `input_references` seguem as regras do Qwen. A primeira versão aceita **uma referência** por URL HTTP(S) pública ou data URL; duas ou mais retornam erro 400. Z-Image image-to-image transforma a referência e não garante edição idêntica ao Qwen. A imagem é redimensionada para as dimensões solicitadas. `n=1` e PNG são os únicos suportados; streaming não é suportado.

As proporções e as dimensões `1K`/`2K` e `size` aceitas são as mesmas do Qwen, para preservar o contrato; 2K não foi validado nesta GPU e pode esgotar memória. `GET /api/v1/images/models` anuncia o nome Z e o alias Qwen e informa o limite de uma referência; os endpoints de capacidades funcionam com ambos.

## Prompt Enhancer

Ativado por padrão, usa os mesmos modelos GGUF Q4 com MTP do Qwen, texto ou edição, mas com instruções próprias para Z-Image-Turbo e JSON interno `rewritten_prompt`. O processo llama.cpp é carregado na GPU por job e encerrado antes da difusão; os pesos do Z-Image ficam em RAM com offload. A qualidade da reescrita com as instruções adaptadas precisa ser validada em casos reais. As proporções escolhidas pelo cliente são preservadas. A seed controla a geração de imagem, não garante a mesma reescrita.

Defina `PROMPT_ENHANCER_ENABLED=false` para desativar. Uma falha no enhancer falha o job; não há troca silenciosa para o prompt original. O dashboard indica a fase `enhancing_prompt`. O volume armazena cerca de 6 GB por modelo de enhancer, mais 0,9 GB para o componente visual de edição.

## Configuração

| Variável | Padrão | Uso |
| --- | --- | --- |
| `API_TOKEN` | vazio | Token Bearer; vazio desabilita autenticação |
| `CPU_OFFLOAD` | `sequential` | `sequential` ou `model` |
| `HF_MODEL_ID` / `HF_MODEL_REVISION` | `Tongyi-MAI/Z-Image-Turbo` / `main` | Checkpoint e revisão |
| `HF_HOME` / `HF_TOKEN` | `/data/huggingface` / vazio | Cache e token Hugging Face opcional |
| `INFERENCE_STEPS` | 9 texto / 8 edição | Sobrescreve passos nas duas modalidades; deve ser compatível com `strength=0.6` na edição |
| `PROMPT_ENHANCER_ENABLED` | `true` | Reescrita automática |
| `PROMPT_ENHANCER_CONTEXT_SIZE` / `PROMPT_ENHANCER_MAX_TOKENS` | `24576` / `8192` | Contexto e limite de tokens do enhancer |
| `PROMPT_ENHANCER_GGUF_REPO` / `PROMPT_ENHANCER_GGUF_REVISION` | modelo e revisão fixados no código | Fonte dos pesos GGUF |
| `PROMPT_ENHANCER_START_TIMEOUT` / `PROMPT_ENHANCER_TIMEOUT` | `300` / `1800` | Limites em segundos para inicialização e inferência |
| `MAX_QUEUE_SIZE` / `HISTORY_SIZE` | `10` / `100` | Fila e histórico em memória |
| `MAX_REQUEST_BYTES` / `MAX_REFERENCE_BYTES` / `MAX_REFERENCE_PIXELS` | `157286400` / `10485760` / `40000000` | Limites de entrada |
| `WEBHOOK_ALLOWED_HOSTS` / `WEBHOOK_SECRET` | vazio | Hosts permitidos e segredo HMAC |
| `WEBHOOK_RETRIES` / `WEBHOOK_TIMEOUT` | `5` / `10` | Tentativas e timeout de entrega |

Os resultados, limites de tempo e qualidade podem variar com GPU, RAM e dimensões. O dashboard e o serviço mantêm seus próprios estados e não coordenam o Qwen: não execute ambos na mesma GPU ao mesmo tempo.

## Testes

```bash
python3 -m pip install -r requirements.txt httpx
python3 -m unittest discover -s tests -v
```

Os testes automatizados simulam a pipeline e não substituem testes reais de texto, edição e webhook na GPU de destino.
