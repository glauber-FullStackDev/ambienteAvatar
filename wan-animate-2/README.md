# Wan-Animate-2 em Docker

Quarta imagem independente do projeto. Ela nao substitui nem altera o ambiente
[`wan-animate/`](../wan-animate/README.md), que continua oferecendo Wan 2.2
Animate no ComfyUI. Este ambiente usa o sucessor oficial Wan-Animate-2 por meio
da integracao modular do Diffusers e uma interface Gradio dedicada.

## O que entra e o que sai

- imagem final: identidade, rosto, roupa e fundo;
- video-guia: movimentos, gestos e expressoes;
- prompt: descricao objetiva da aparencia e do fundo;
- saida: video-base sem copiar o audio do video-guia.

O Wan-Animate-2 processa o video diretamente e nao usa ViTPose, YOLO ou SAM2.

## Hardware

Este modelo e substancialmente mais pesado que o Wan 2.2 Animate anterior. O
checkpoint ocupa aproximadamente 46 GB em disco. Com group offload, o consumo
documentado pelo Diffusers e aproximadamente:

| Area de saida | Memoria aproximada |
| --- | ---: |
| 480x320 | 55 GB |
| 640x480 | 63 GB |
| 800x640 | 72 GB |

Use A100 80 GB, H100 80 GB ou equivalente e pelo menos 128 GB de RAM. GPUs de
24/48 GB nao sao o alvo funcional deste fluxo atual. Reserve pelo menos 130 GB
de disco por variante baixada.

## Variantes

| `MODEL_VARIANT` | Passos | Uso |
| --- | ---: | --- |
| `base` | 40 | Padrao; melhor qualidade |
| `distilled` | 10 | Mais rapido; checkpoint separado |

Somente a variante selecionada e baixada. Para manter as duas, execute o
download uma vez com cada valor; elas ocupam cerca de 92 GB juntas.

## Uso

```bash
cd wan-animate-2
make build
make models
make verify
make up
```

Abra <http://127.0.0.1:8188>. A primeira geracao compila os blocos de atencao e
pode demorar mais que as seguintes.

O prompt deve descrever objetivamente personagem, roupa e fundo, sem narrar a
acao do video-guia. O modelo preserva a proporcao da imagem; largura e altura
definem a area-alvo.

## Vast.ai

Depois que a pipeline for publicada, use:

```text
ghcr.io/glauber-fullstackdev/ambienteavatar-wan-animate-2:vast
```

Configure o template com argumento `serve`, porta 8188, uma GPU de 80 GB e
volume persistente em `/models`. Na primeira inicializacao defina
`DOWNLOAD_MODELS_ON_START=1`; depois volte para `0`.

Variaveis principais:

- `MODEL_VARIANT=base` ou `distilled`;
- `HF_TOKEN`, se necessario;
- `DOWNLOAD_MODELS_ON_START=0`;
- `APP_PORT=8188`.

## Fontes fixadas

- Wan-Animate-2 oficial: `3ad2fef7d61d6200c9c653e0fe47be7616b323f3`;
- Diffusers com pipeline modular: `360bef807475899c2e4d7d99c2f371148a78b1a7`;
- checkpoint Base: `7d48412d7b903ff3a89f4f5a960d99e1899605a1`;
- checkpoint Distilled: `59e4141466bcb1bf9733eca1bc78be6891c9fbdf`.
