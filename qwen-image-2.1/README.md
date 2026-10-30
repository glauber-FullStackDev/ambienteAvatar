# Qwen-Image-2.1 — API síncrona e dashboard

Serviço independente para geração e edição de imagens com o checkpoint oficial `Qwen/Qwen-Image-2.1`. Expõe a Image API do OpenRouter e um painel de acompanhamento, sem depender dos workers LTX.

**Licença:** os pesos oficiais estão sujeitos à [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), que permite pesquisa e avaliação não comercial; uso comercial requer licença separada.

## Requisitos e limitações

- Linux, Docker com Compose, NVIDIA Container Toolkit e GPU NVIDIA com CUDA e BF16. O alvo é 16 GB de VRAM, **ainda não validado**: offload, resolução, número de referências e memória RAM do host afetam o resultado. Se faltar VRAM, experimente `CPU_OFFLOAD=sequential`; pode ser muito mais lento. Se ainda falhar, será necessária mais VRAM ou uma solução de quantização validada.
- Espaço em disco e RAM suficientes para o modelo completo (que inclui encoder visual, transformer e VAE). O primeiro boot baixa os pesos; pode demorar e precisa de acesso ao Hugging Face. O volume `/data` conserva o cache. Uma revisão do modelo pode ser definida por `HF_MODEL_REVISION`; fixe também `DIFFUSERS_REV` no build para reproduzir uma versão validada.
- A fila e o histórico vivem na memória. Um reinício interrompe as chamadas e apaga o histórico; o cliente precisa reenviar. Se um cliente desconectar na fila, a solicitação é removida; se já estiver em execução, a GPU termina o trabalho e o resultado é descartado.
- Configure timeout do cliente, balanceador e proxy para cobrir **espera na fila e inferência**, potencialmente dezenas de minutos com offload. Chamadas simultâneas mantêm conexões abertas. Execute apenas um processo Uvicorn por GPU: múltiplos workers teriam filas independentes e carregariam o modelo várias vezes.

## Iniciar

```bash
cd qwen-image-2.1
export API_TOKEN='escolha-um-token-forte'
docker compose up -d --build
docker compose logs -f
```

Acesse `http://localhost:8001/dashboard` e informe o token. O container continua usando a porta interna `8000`, mapeada para a porta `8001` no servidor; configure o tunnel para `http://localhost:8001`. `GET /health` retorna 503 enquanto o modelo carrega ou se houver erro de inicialização, e 200 quando estiver pronto. O painel atualiza a cada 1,5 segundo; o percentual representa os passos de amostragem, não o tempo total até o PNG pronto.

## Publicação e execução pelo GHCR

O workflow `.github/workflows/publish-qwen-image-2-1.yml` testa a API e publica a imagem Linux/AMD64 quando um push para `main` inclui mudanças em `qwen-image-2.1/**`. Alterações em outras pastas ou apenas no arquivo do workflow não disparam a publicação. A primeira publicação depende de enviar os arquivos do Qwen para `main` e de o workflow terminar com sucesso. Não há disparo manual.

Neste repositório, a imagem é publicada como `ghcr.io/glauber-fullstackdev/ambienteavatar-qwen-image-2-1:latest`, além da tag imutável `sha-<SHA completo do commit>`. A visibilidade do pacote é controlada no GitHub; se for privado, autentique o servidor no GHCR com credenciais que tenham acesso ao pacote antes de executar `docker pull`.

```bash
docker pull ghcr.io/glauber-fullstackdev/ambienteavatar-qwen-image-2-1:latest
docker run -d --name qwen-image --gpus all -p 8001:8000 \
  -v qwen-models:/data -e API_TOKEN="$API_TOKEN" \
  ghcr.io/glauber-fullstackdev/ambienteavatar-qwen-image-2-1:latest
docker logs -f qwen-image
```

O download dos pesos ocorre no servidor na primeira inicialização. A publicação da imagem não valida a inferência nem comprova que ela cabe em uma GPU de 16 GB.

Também é possível executar a imagem diretamente:

```bash
docker build -t qwen-image-2.1 .
docker run --gpus all -p 8001:8000 -v qwen-models:/data \
  -e API_TOKEN="$API_TOKEN" qwen-image-2.1
```

`API_TOKEN` vazio desabilita autenticação; use um token e HTTPS via proxy em exposição fora de ambiente confiável. O HTML do painel e `/health` são públicos; dados do painel, descoberta e geração exigem `Authorization: Bearer <token>` quando configurado. O painel guarda o token na sessão do navegador.

## API

O endpoint `POST /api/v1/images` é síncrono: retorna somente após geração ou falha. Solicitações simultâneas aguardam em FIFO; fila cheia responde HTTP 429. O retorno contém `created` e `data`, com `b64_json` e `media_type: image/png`. Erros retornam `error.message` e `error.code`; o serviço não inventa custos nem contagem de tokens.

```bash
curl --max-time 3600 http://localhost:8001/api/v1/images \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen-image-2.1","prompt":"Uma capivara lendo em uma biblioteca","resolution":"1K","aspect_ratio":"1:1","seed":42}'
```

Edição com imagem de referência (HTTPS público ou data URL base64):

```bash
curl --max-time 3600 http://localhost:8001/api/v1/images \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen-image-2.1","prompt":"Troque o fundo por uma praia ao pôr do sol","input_references":[{"type":"image_url","image_url":{"url":"https://example.com/foto.png"}}]}'
```

Campos suportados: `model` e `prompt` obrigatórios; `seed`; `resolution` (`1K` padrão ou `2K`); `aspect_ratio` (`1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16`); `size` (nível ou dimensão exata da tabela); `n=1`; `output_format=png`; e até 10 `input_references`. `stream=true` e outros parâmetros não são suportados. As dimensões 2K são as recomendadas pelo modelo, e as 1K são versões reduzidas para múltiplos de 32. Todas são configurações disponíveis, não garantias de que caibam em 16 GB.

| Proporção | 1K | 2K |
| --- | --- | --- |
| 1:1 | 1024x1024 | 2048x2048 |
| 4:3 | 1216x896 | 2400x1792 |
| 3:4 | 896x1216 | 1792x2400 |
| 3:2 | 1280x832 | 2528x1696 |
| 2:3 | 832x1280 | 1696x2528 |
| 16:9 | 1376x768 | 2752x1536 |
| 9:16 | 768x1376 | 1536x2752 |

`GET /api/v1/images/models` e `GET /api/v1/images/models/qwen/qwen-image-2.1/endpoints` expõem capacidades. `GET /dashboard/api/summary` fornece o estado do painel, incluindo prompts; proteja o acesso de acordo com a sensibilidade dos dados.

## Configuração

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `API_TOKEN` | vazio | Token Bearer; vazio desabilita autenticação |
| `MAX_QUEUE_SIZE` | `10` | Máximo de solicitações aguardando; `0` não permite espera |
| `HISTORY_SIZE` | `100` | Número máximo de entradas mantidas no painel |
| `CPU_OFFLOAD` | `model` | `model` ou `sequential` |
| `INFERENCE_STEPS` | `40` | Passos de inferência, se o checkpoint não definir seu próprio cronograma |
| `HF_HOME` | `/data/huggingface` | Cache persistente dos pesos |
| `HF_MODEL_ID` | `Qwen/Qwen-Image-2.1` | Checkpoint Diffusers |
| `HF_MODEL_REVISION` | `main` | Revisão do checkpoint |
| `HF_TOKEN` | vazio | Token Hugging Face, se necessário |
| `MAX_REQUEST_BYTES` | `157286400` | Limite do corpo HTTP |
| `MAX_REFERENCE_BYTES` | `10485760` | Limite por imagem de referência |
| `MAX_REFERENCE_PIXELS` | `40000000` | Limite de pixels por referência |

Referências remotas usam conexão direta a IP público validado, recusam endereços privados e verificam cada redirecionamento. Os limites não substituem controles de rede e autenticação da implantação.

## Testes

```bash
python3 -m pip install -r requirements.txt httpx
python3 -m unittest discover -s tests -v
```

Os testes usam um pipeline simulado e não substituem uma validação de geração e edição em GPU. Para este último, execute os dois exemplos acima em uma máquina NVIDIA de 16 GB, verifique o PNG retornado, acompanhe `nvidia-smi` e observe o painel durante a execução.
