# Ambiente Avatar — LTX 2.3 GGUF 16 GB (RTX 5060 Ti)

Variante **não-serverless** do path `../ltx-2.3` para GPUs de **16 GB de VRAM**
(alvo: RTX 5060 Ti), mantendo o LoRA pessoal `glauberavatar.safetensors`
aplicado em runtime sobre pesos quantizados.

Em vez do checkpoint `ltx-2.3-22b-dev-fp8` (27,1 GiB) + Gemma fp4 (8,8 GiB) +
LoRA distilled (2,6 GiB) — impossíveis em 16 GB mesmo quantizados — este path
usa o pacote GGUF do `unsloth/LTX-2.3-GGUF`, com pesos streamados entre VRAM e
RAM pelo ComfyUI-GGUF.

## Requisitos de host

| Recurso | Valor | Por quê |
|---|---|---|
| GPU | 16 GB VRAM (RTX 5060 Ti, Blackwell/sm_120) | DiT GGUF Q4_K_M + ativações |
| **RAM** | **64 GB recomendado, 32 GB mínimo** | GGUF do DiT (~13 GiB) + Gemma QAT (~7 GiB) ficam em RAM e são dequantizados camada a camada |
| Disco | volume de ~60 GB em `/opt/ComfyUI/models` | ~26 GiB de pesos + cache/output |
| CUDA | >= 12.8 (imagem usa PyTorch 2.8.0 + CUDA 12.8) | sm_120 |

## Modelos (total ≈ 26,4 GiB)

| Arquivo | Repo/origem | Pasta de destino | Tamanho |
|---|---|---|---|
| `ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` | unsloth/LTX-2.3-GGUF (`distilled-1.1/`) | `models/diffusion_models` | 13,2 GiB |
| `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` | unsloth/gemma-3-12b-it-qat-GGUF | `models/text_encoders` | 6,9 GiB |
| `ltx-2.3-22b-distilled_embeddings_connectors.safetensors` | unsloth/LTX-2.3-GGUF | `models/text_encoders` | 2,2 GiB |
| `ltx-2.3-22b-distilled_video_vae.safetensors` | unsloth/LTX-2.3-GGUF | `models/vae` | 1,4 GiB |
| `ltx-2.3-22b-distilled_audio_vae.safetensors` | unsloth/LTX-2.3-GGUF | `models/vae` | 0,3 GiB |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | Lightricks/LTX-2.3 | `models/latent_upscale_models` | 0,9 GiB |
| `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` | Comfy-Org/ltx-2 | `models/loras` | 0,6 GiB |
| `glauberavatar.safetensors` | embutido na imagem (assets) | `models/loras` | 0,4 GiB |

Todos os repositórios são públicos — **não é preciso `HF_TOKEN`**.

Para tentar mais qualidade, suba o quant do DiT com
`LTX23_GGUF_QUANT=Q5_K_M` (15,2 GiB) ou `Q6_K` (17,8 GiB) no downloader e
ajuste o widget `unet_name` do node `UnetLoaderGGUF` no workflow — a comunidade
reporta Q6_K acima do fp8 em qualidade percebida, com peak de ~15 GB de VRAM
em Q4. O default `Q4_K_M` é o ponto de partida seguro.

## O que muda em relação ao path de 48 GB

Mesmo grafo pessoal (IA2V/A2V), mesmas seeds, mesmos sigmas e mesmo two-stage
(latent upscale x2). Apenas os loaders trocam:

| Path 48 GB | Path 16 GB | Motivo |
|---|---|---|
| `CheckpointLoaderSimple` (dev-fp8, 27 GiB) | `UnetLoaderGGUF` (distilled 1.1 Q4_K_M GGUF) | pesos quantizados streamados RAM↔VRAM |
| `LTXAVTextEncoderLoader` (Gemma fp4 + checkpoint p/ connector) | `DualCLIPLoaderGGUF` (Gemma 3 12B QAT GGUF + connectors safetensors, type `ltxv`) | o node nativo lê o checkpoint inteiro (27 GiB) para achar o connector |
| `LTXVAudioVAELoader` (áudio VAE do checkpoint) | `VAELoaderKJ` (KJNodes) com audio VAE standalone | standalone de 0,3 GiB em vez de ler 27 GiB |
| VAE de vídeo do checkpoint | `VAELoaderKJ` + `ltx-2.3-22b-distilled_video_vae.safetensors` | standalone de 1,4 GiB |
| LoRA distilled dinâmico (2,6 GiB, força 0,5) | **removido** | o DiT distilled 1.1 GGUF já embute a destilação (8 passos, CFG 1) |
| `LoraLoaderModelOnly glauberavatar` | **mantido** (força 1, sobre os pesos GGUF) | identidade visual |

O node `LTXVAudioVAELoader`/`LTXAVTextEncoderLoader` nativos fazem
`load_torch_file` do checkpoint inteiro (27 GiB em RAM) só para extrair o VAE
de áudio e o connector — inviável aqui; por isso os arquivos standalone do
repositório `unsloth/LTX-2.3-GGUF` (mesma receita do workflow oficial deles).

## Estrutura

```
ltx-2.3-16gb/
├── Dockerfile                     # ComfyUI pinado + ComfyUI-GGUF + KJNodes
├── assets/
│   ├── glauberavatar.safetensors  # LoRA pessoal (408 MB, bf16) — igual ao 48 GB
│   └── video_ltx2_3_ia2v_personal_source.json
├── custom_nodes/ComfyUI-LastFrameExport/
├── scripts/
│   ├── workflow_utils.py          # helpers de grafo (links, nodes)
│   ├── personal_workflows.py      # add_last_frame_export + build_a2v (do path 48 GB)
│   ├── gguf_convert.py            # cirurgia GGUF + validate_gguf
│   ├── build_ia2v_personal_lora_16gb.py
│   ├── build_a2v_personal_lora_16gb.py
│   ├── download_models.py         # GGUF Q4_K_M/Q5_K_M/Q6_K + Gemma QAT + VAEs
│   ├── entrypoint.py              # serve | download-models | verify
│   ├── container-entrypoint.sh
│   ├── check_dependencies.py      # valida os 2 workflows no build
│   └── smoke_core_nodes.py        # importa todos os custom nodes em --cpu
└── vast/
    ├── ltx23-16gb-template.json
    └── ltx23-16gb-jupyter-template.json
```

## Build

```bash
docker build -t ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23-16gb:vast .
```

O build já: compila os scripts, gera os dois workflows GGUF
(`*_ia2v_personal_lora_16gb.json` e `*_a2v_personal_lora_16gb.json`) a partir
do workflow pessoal fonte, valida a integridade dos grafos
(`check_dependencies.py`) e faz o smoke test de import dos custom nodes
(ComfyUI-GGUF + KJNodes + LastFrameExport).

## Vast.ai

Templates em `vast/`:

- `ltx23-16gb-template.json` — runtype args, filtro `gpu_ram >= 15`,
  `ram >= 50 GB`, disco 60 GB.
- `ltx23-16gb-jupyter-template.json` — JupyterLab + SSH, mesma imagem.

Volume: anexe em `/opt/ComfyUI/models` (~60 GB). Primeiro boot baixa ~26 GiB.

## Uso

1. Abra a porta HTTP 8188.
2. Use **LTX 2.3 IA2V + LoRA pessoal (GGUF 16 GB)** (imagem + áudio) ou
   **A2V** (só áudio).
3. O último frame sai em PNG em `output/images/last_frame/` (para continuar o
   vídeo no próximo clip).

Variáveis úteis:

- `LTX23_GGUF_QUANT`: `Q4_K_M` (default) | `Q5_K_M` | `Q6_K` — quant do DiT
  baixado pelo downloader. Se trocar, ajuste também o widget do
  `UnetLoaderGGUF` no workflow (ou reconstrua com o mesmo valor).
- `COMFYUI_ARGS` — default `--preview-method auto --reserve-vram 1.5`.
  Em casos de OOM adicione `--cache-none`.
- `DOWNLOAD_MODELS_ON_START=0` — pula o download/verificação no boot.

## Limitações e riscos conhecidos

- **Qualidade**: Q4_K_M quantiza os pesos do DiT; a identidade do LoRA pessoal
  sobre pesos Q4 precisa de validação visual (compare com o path 48 GB usando
  a mesma seed — as agendas de sigmas são idênticas, então seeds batem).
- **ComfyUI-GGUF**: pinado em `6ea2651` (PRs de metadata LTX-2 + Gemma GGUF).
  Se o loader reclamar de metadata do GGUF, atualize o commit no Dockerfile.
- **720p no estágio 2**: se estourar VRAM no latent upscale, reduza o stage 1
  (EmptyLTXVLatentVideo) e use o upscaler x2 como hoje — o two-stage já faz
  isso; alternativa é baixar a duração para 5–6 s no primeiro uso.
- O Gemma prompt-enhancer (`TextGenerateLTX2Prompt`) roda o 12B em GGUF; com
  pouco VRAM sobrando, desligue o switch "prompt enhance" — o prompt digitado
  é usado direto.
