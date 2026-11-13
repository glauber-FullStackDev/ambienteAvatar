# FLUX 3 Image Klein quantizado — API local

Serviço de geração e edição de imagens com a mesma API HTTP do `qwen-image-2.1`: geração síncrona, fila, webhook, painel, parâmetros, tamanhos e resposta PNG em base64. O campo `model` aceita `black-forest-labs/flux-3-image-klein` e o alias `qwen/qwen-image-2.1`. Ambos usam FLUX.

O backend usa ComfyUI v0.41.3 em uma porta interna local e os quatro arquivos oficiais da [Black Forest Labs para ComfyUI](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy): gerador INT8 ConvRot (5,8 GB), encoder INT8 (4,9 GB), expansor de prompt W4A8 (6,7 GB) e VAE (0,3 GB). O expansor recebe o prompt e as referências antes da geração. O ComfyUI inicia com `--lowvram` e `--disable-smart-memory`, para transferir componentes agressivamente da GPU para a RAM entre operações; isso pode reduzir a velocidade e não elimina o consumo de memória para ativações. Os arquivos totalizam aproximadamente 17,7 GB; isso não representa a VRAM total necessária em execução.

O primeiro início baixa os arquivos. Acesse o repositório do modelo, aceite as condições para a conta do token e use um `HF_TOKEN` com acesso ao repositório restrito. Aceitar apenas as condições do repositório BF16 pode não liberar o repositório `flux-3-image-klein-comfy`. Os pesos têm a [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/main/LICENSE.md); verifique os termos aplicáveis.

## Executar

É necessária GPU NVIDIA CUDA compatível, Docker com NVIDIA Container Toolkit e RAM e disco suficientes. A integração quantizada não foi validada na GPU de destino; tamanho da saída, referências e outras aplicações que ocupem VRAM afetam o consumo. Reserve espaço para os pesos novos além dos arquivos BF16 já existentes no volume e da imagem Docker.

```bash
cd flux-3-image-klein
export HF_TOKEN=seu_token_hugging_face
export API_TOKEN=um_token_para_sua_api
sudo --preserve-env=HF_TOKEN,API_TOKEN docker compose up --build -d
sudo docker compose logs -f
```

A porta é `8001` no servidor e `8000` no container. O processo ComfyUI só escuta em `127.0.0.1:8188` dentro do container. Os arquivos ficam no volume `flux-3-models`. Consulte `/health`: `ready: true` indica que os pesos foram baixados e o ComfyUI iniciou; a primeira geração ainda pode exigir carregamento dos modelos.

```bash
curl http://localhost:8001/api/v1/images \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen-image-2.1","prompt":"Uma capivara lendo em uma biblioteca","resolution":"1K","aspect_ratio":"1:1","seed":42}'
```

Para edição, envie até dez objetos em `input_references`, cada um com `{"type":"image_url","image_url":{"url":"https://example.com/foto.png"}}`. URLs públicas e data URLs base64 seguem a validação da Qwen. Referências são processadas na ordem enviada; o lote passado ao expansor pode ser redimensionado para dimensões comuns pelo ComfyUI. Veja o contrato HTTP completo em [Qwen Image 2.1](../qwen-image-2.1/README.md).

As rotas `/api/v1/images/async`, `/api/v1/images/jobs/{job_id}`, `/api/v1/images/models` e `/api/v1/images/models/{model_id}/endpoints` são mantidas. O webhook exige `WEBHOOK_ALLOWED_HOSTS`. Para solicitações demoradas atrás de um proxy com limite de tempo, use a rota assíncrona com webhook; a rota síncrona depende do limite de espera do proxy.

`INFERENCE_STEPS` tem padrão 16; `COMFY_MODEL_REVISION` fixa a revisão dos arquivos do Hugging Face e `GENERATION_TIMEOUT` limita em segundos a espera pelo ComfyUI (padrão 3600). `COMFY_REV` é argumento de build e o padrão é `v0.41.3`. As variáveis antigas do Diffusers e `FLUX_UPSAMPLER_*` não são usadas. O progresso por etapa no painel não é detalhado.

O workflow `.github/workflows/publish-flux-3-image-klein.yml` testa e publica a imagem no GHCR em push para `main` quando há alterações em `flux-3-image-klein/**`.
