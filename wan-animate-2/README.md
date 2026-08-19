# Wan-Animate-2 INT8 em Docker

Quarta imagem independente do projeto. Ela nao substitui nem altera a geracao
de imagem da raiz ou o ambiente [`wan-animate/`](../wan-animate/README.md), que
continua oferecendo o Wan 2.2 Animate anterior.

Este fluxo usa o suporte nativo do ComfyUI ao Wan-Animate-2 e o workflow oficial
de Motion Transfer. O modelo principal e
`wan_animate_2_int8_convrot.safetensors`; o encoder UMT5 usa FP8.

## O que entra e o que sai

- imagem final: identidade, rosto, roupa e fundo desejados;
- video-guia: movimentos, gestos e expressoes;
- prompt principal: descricao objetiva da aparencia, roupa e fundo;
- prompt de pose: descricao da acao do video-guia;
- saida: video-base visual para ser usado depois no InfiniteTalk.

O workflow semeado remove o audio do video-guia e desativa por padrao a
comparacao lado a lado do template oficial. O restante do grafo oficial,
incluindo o processamento direto do video, e preservado. O cache de contexto
continua em INT8, mas usa CPU/RAM por padrao para reduzir picos de VRAM.

## Modelos

`make models` baixa somente o conjunto recomendado pelo workflow oficial:

| Arquivo | Precisao | Tamanho |
| --- | --- | ---: |
| `wan_animate_2_int8_convrot.safetensors` | INT8 ConvRot | 16.65 GB |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | FP8 | 6.74 GB |
| `clip_vision_h.safetensors` | checkpoint oficial | 1.26 GB |
| `lightx2v_..._bf16.safetensors` | LoRA BF16 | 0.74 GB |
| `Wan2_1_VAE_bf16.safetensors` | BF16 | 0.25 GB |

O conjunto ocupa 25.65 GB. Reserve pelo menos 40 GB no volume de modelos para
downloads temporarios, cache e futuras atualizacoes. Os pesos nao fazem parte da
imagem Docker.

INT8 reduz os pesos do DiT pela metade em relacao ao BF16, mas o cache temporal
do Wan-Animate-2 ainda cresce com a resolucao e o comprimento do segmento. Para
o primeiro teste, use o tamanho e os 81 frames do workflow oficial. Se faltar
VRAM, reduza a resolucao antes de reduzir o segmento. Reserve bastante RAM para
o cache em CPU; para priorizar velocidade em uma GPU maior, troque `cache_device`
para `gpu` no subgrafo Motion Transfer.

## Uso local

```bash
cd wan-animate-2
make build
make models
make verify
make up
```

Abra <http://127.0.0.1:8188> e carregue o workflow
`wan-animate-2-int8-docker.json`, que aparece na pasta de workflows do usuario.
Troque a imagem e o video de exemplo pelos arquivos desejados. Para gerar apenas
o video-base, mantenha o subgrafo `Video Stitch` desativado.

Os dados persistentes ficam em:

```text
data/
├── cache/   # cache do Hugging Face
├── input/   # imagem e video enviados
├── models/  # checkpoints (montado em /models)
├── output/  # video-base gerado
└── user/    # configuracoes e workflows do ComfyUI
```

## Vast.ai

Use a imagem publicada:

```text
ghcr.io/glauber-fullstackdev/ambienteavatar-wan-animate-2:vast
```

Configure argumento `serve`, porta 8188 e um volume persistente em `/models`.
Na primeira inicializacao, defina `DOWNLOAD_MODELS_ON_START=1`; depois que os
25.65 GB forem verificados, volte para `0` para nao consultar os arquivos a cada
boot.

Variaveis principais:

- `DOWNLOAD_MODELS_ON_START=0` ou `1`;
- `HF_TOKEN`, se necessario;
- `COMFYUI_PORT=8188`;
- `COMFYUI_ARGS=--preview-method auto`.

## Fontes fixadas

- ComfyUI com nodes nativos do Wan-Animate-2:
  `c67885b14556cf3e4e061862925282d403d09862`;
- workflow oficial do Comfy-Org:
  `55818c64caf6d28309ddee204827e51a2c45f4dd`;
- modelos `Comfy-Org/Wan-Animate-2`:
  `ed158470869ff31fa51cf56012dac33fb00f494b`;
- ComfyUI Manager: `f39cbd56fecae0b27a446c0cd450cd591f3a8bea`.
