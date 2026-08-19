# Wan 2.2 Animate + ComfyUI no Vast.ai

Terceira imagem CUDA do projeto, isolada de LongCat e InfiniteTalk. Ela gera
somente o **video-base** a partir de:

- um video-guia gravado, usado para pose, gestos e expressoes faciais;
- uma imagem de referencia alterada por IA, usada para identidade, roupa,
  aparencia e fundo.

O workflow semeado usa o modo **animation/move**. As entradas `bg_images` e
`mask` do `WanVideoAnimateEmbeds` ficam desconectadas de proposito; isso evita o
modo replacement/mix, que manteria o fundo do video-guia. O resultado fica
pronto para ser usado como base na etapa separada do InfiniteTalk.

## Uso local

```bash
cd wan-animate
make build
make models
make verify
make up
```

Abra <http://127.0.0.1:8188> e carregue o workflow
`wan-animate-base-docker.json`, criado automaticamente em
`user/default/workflows`.

No workflow:

1. selecione o video gravado em `VHS_LoadVideo`;
2. selecione a imagem final em `LoadImage`;
3. confira a deteccao de pose e rosto;
4. ajuste o prompt para descrever a imagem e execute a fila.

O preset usa 832x480, 16 fps, quatro passos com Lightx2v, SDPA e block swap. O
arquivo final, sem a faixa de audio do video-guia, e salvo com prefixo
`wan-animate-base` em `data/output`. O audio final continua sendo
responsabilidade do InfiniteTalk. Para clipes longos, teste primeiro um trecho
curto e aumente gradualmente o numero de frames.

## Vast.ai

Use a imagem sem attestations:

```text
ghcr.io/glauber-fullstackdev/ambienteavatar-wan-animate:vast
```

Configure `Entrypoint/Args`, argumento `serve`, porta 8188, CUDA 12.8 ou
superior e um volume persistente em `/opt/ComfyUI/models`. Na primeira
inicializacao, `DOWNLOAD_MODELS_ON_START=1` baixa os pesos; depois volte para
`0` para nao verificar downloads grandes a cada boot.

Variaveis principais:

- `MODEL_VARIANT=e4m3fn` (padrao) ou `e5m2`;
- `HF_TOKEN`, se o Hugging Face exigir autenticacao;
- `COMFYUI_ARGS=--preview-method auto`;
- `DOWNLOAD_MODELS_ON_START=0` por padrao.

Use uma GPU NVIDIA de pelo menos 24 GB para o preset FP8 com offload/block
swap; 48 GB ou mais oferece mais folga. Reserve ao menos 80 GB de disco para a
imagem, cache, modelos, entradas e saidas. A inferencia nao funciona em
CPU/MPS.

## Responsabilidade desta imagem

```text
video-guia + imagem modificada
              |
              v
       Wan 2.2 Animate
              |
              v
       video-base visual
              |
              v
 InfiniteTalk (outra imagem/etapa)
```

Esta imagem nao gera o dialogo final e nao substitui a pipeline do
InfiniteTalk. Ela se limita a transferir o movimento do video para a aparencia
completa definida na imagem.
