# FLUX 3 Image Klein — API local

Serviço independente para geração e edição com o checkpoint destilado `black-forest-labs/flux-3-image-klein`. Mantém o contrato HTTP da API `qwen-image-2.1`: geração síncrona, fila, webhook, painel, os mesmos parâmetros, tamanhos e resposta PNG em base64. O campo `model` aceita tanto `black-forest-labs/flux-3-image-klein` como o alias `qwen/qwen-image-2.1`; ambos usam FLUX internamente.

O modelo suporta texto e até dez referências. Usa o pipeline modular Diffusers. Para reduzir o consumo de memória, o prompt upsampler oficial fica desativado por padrão: envie prompts detalhados. Para ativá-lo, defina `FLUX_PROMPT_UPSAMPLER=true`; ele exige muito mais memória e também é baixado. O modelo usa a [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein); confira seus termos para o seu uso. O acesso aos pesos exige aceitar as condições no Hugging Face e fornecer um token autorizado; confira também o acesso ao repositório do upsampler.

## Executar

É necessária GPU NVIDIA com CUDA e BF16, Docker com NVIDIA Container Toolkit, RAM e espaço em disco suficientes para o modelo e seus componentes. O offload automático reduz a pressão na GPU, mas esta integração não foi validada em GPU de 16 GB: resolução, referências e upsampler podem exigir mais memória. O primeiro carregamento baixa os pesos e pode demorar.

```bash
cd flux-3-image-klein
export HF_TOKEN=seu_token_hugging_face
export API_TOKEN=um_token_para_sua_api
docker compose up --build -d
docker compose logs -f
```

A porta padrão é `8001`, configurável em `compose.yaml`. Consulte `GET /health` para verificar se o modelo carregou; o painel está em `/dashboard`. A API começa a aceitar trabalhos quando o worker estiver pronto. O cache persistente fica no volume Docker `flux-3-models`.

```bash
curl http://localhost:8001/api/v1/images \
  -H "Authorization: Bearer $API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen-image-2.1","prompt":"Uma capivara lendo em uma biblioteca","resolution":"1K","aspect_ratio":"1:1","seed":42}'
```

Para edição, inclua `input_references` com até 10 objetos `{"type":"image_url","image_url":{"url":"https://example.com/foto.png"}}`; também são aceitas data URLs base64, como na Qwen. Para associar imagens no prompt, use `<ref_image_0>`, `<ref_image_1>`, etc. Os valores de `resolution`, `aspect_ratio`, `size`, `n`, `output_format`, `seed` e `stream` seguem a validação da Qwen. Veja os exemplos e o contrato completo em [Qwen Image 2.1](../qwen-image-2.1/README.md).

As rotas `/api/v1/images/async`, `/api/v1/images/jobs/{job_id}`, `/api/v1/images/models` e `/api/v1/images/models/{model_id}/endpoints` também são mantidas. O webhook exige `WEBHOOK_ALLOWED_HOSTS`, conforme a documentação da Qwen. A listagem informa ambos os identificadores. O painel mostra a fase de amostragem sem progresso por etapa, pois o pipeline modular usado aqui não fornece o mesmo callback da Qwen.

As variáveis de fila, autenticação, limites de requisição/referência e webhook são as mesmas do compose da Qwen. `INFERENCE_STEPS` tem padrão 16; `HF_MODEL_ID` e `HF_MODEL_REVISION` configuram o checkpoint. `DIFFUSERS_REV` é um argumento de build Docker para fixar uma revisão do Diffusers; o padrão `main` pode mudar. O upsampler opcional faz parte do pipeline FLUX e não usa as variáveis `PROMPT_ENHANCER_*` da Qwen. Desativá-lo pode reduzir a qualidade, especialmente em edições. A geração ainda precisa ser validada na GPU de destino.

O workflow `.github/workflows/publish-flux-3-image-klein.yml` testa a API e publica a imagem no GHCR em push para `main` com alterações nesta pasta. A imagem é `ghcr.io/glauber-fullstackdev/ambienteavatar-flux-3-image-klein:latest`, depois que a publicação concluir.
