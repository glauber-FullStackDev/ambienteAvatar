# LongCat Avatar 1.5 + ComfyUI em Docker

Este repositorio publica cinco imagens independentes. A raiz continua sendo o
ambiente LongCat; o InfiniteTalk fica em [`infinitetalk/`](infinitetalk/README.md)
e a geracao de video-base por movimento fica em
[`wan-animate/`](wan-animate/README.md). O sucessor Wan-Animate-2 em ComfyUI,
com modelo INT8 ConvRot e encoder FP8, fica separado em
[`wan-animate-2/`](wan-animate-2/README.md). Os workflows oficiais LTX 2.3 IA2V
e ID-LoRA ficam em [`ltx-2.3/`](ltx-2.3/README.md). Cada imagem possui sua propria
pipeline com filtro por caminho, portanto uma alteracao em um ambiente nao
reconstrói os outros.

Ambiente CUDA reproduzível com ComfyUI, ComfyUI Manager, LongCat Avatar,
controle de expressão, ferramentas de vídeo, Advanced Live Portrait,
LatentSync 1.6, FFmpeg e as dependências Python de execução. Os pesos não entram
na imagem: ficam em armazenamento persistente ou são baixados depois que a
instância inicia.

## Componentes incluídos

| Componente | Uso |
| --- | --- |
| `ComfyUI-LongCat-Avatar` | Geração principal do avatar por áudio |
| `LongCat Avatar Motion Smoother` | Suaviza o condicionamento no tempo e controla a força da expressão |
| `ComfyUI-VideoHelperSuite` | Carregamento, combinação e salvamento de vídeo |
| `ComfyUI-AdvancedLivePortrait` | Controle e pós-processamento facial |
| `ComfyUI-LatentSyncWrapper` | Lip-sync de acabamento com LatentSync 1.6 |
| `ComfyUI-Manager` | Administração dos custom nodes |

Todos os nodes e suas bibliotecas já fazem parte da imagem. Ainda é necessário
baixar os checkpoints usados por LongCat, LivePortrait e LatentSync.

Os checkpoints do LatentSync devem ficar em
`/opt/ComfyUI/models/latentsync/`. A imagem liga esse diretório ao caminho
interno esperado pelo wrapper, portanto os arquivos permanecem junto ao volume
normal de modelos. Preserve a estrutura `vae/`, `whisper/`,
`latentsync_unet.pt` e `stable_syncnet.pt` indicada pelo projeto do LatentSync.
O Advanced Live Portrait usa `/opt/ComfyUI/models/liveportrait/` e o detector
facial usa `/opt/ComfyUI/models/ultralytics/`.

## Requisitos

- Linux x86_64 com GPU NVIDIA. O LongCat Avatar 1.5 deste node pack não suporta
  CPU nem MPS/Apple Silicon.
- Driver NVIDIA compatível com CUDA 12.8 e NVIDIA Container Toolkit configurado.
- Recomendação prática: GPU com 24 GB ou mais de VRAM, 64 GB de RAM e pelo menos
  50 GB livres para o modo INT8 padrão. Resolução 720p e vídeos longos podem
  exigir muito mais memória.
- Docker Engine com Docker Compose v2.

> Em macOS é possível editar ou até construir a imagem, mas não executar a
> inferência CUDA. Use uma máquina Linux com NVIDIA, RunPod, Vast.ai ou servidor
> equivalente.

## Início rápido

```bash
make setup
make build
make models
make verify
make up
```

Abra <http://127.0.0.1:8188>. O workflow
`longcat-avatar1.5-docker.json` aparece na pasta de workflows do usuário. Ele é
uma cópia do exemplo do node pack ajustada para o modo escolhido e para o backend
portável `sdpa`.

Para reduzir movimentos exagerados, conecte `LongCat Avatar Motion Smoother`
entre `LongCat Avatar Audio Encode` e `LongCat Avatar Sampler`. Um ponto inicial
é `strength=0.75` e `smoothing_frames=7`; reduza `strength` gradualmente até o
movimento ficar natural. O node trabalha sobre uma cópia do condicionamento e
preserva os metadados de áudio, inclusive fluxos com mais de uma pessoa.

O node do LatentSync aparece na busca como `LatentSync1.6 Node`; o auxiliar
aparece como `Video Length Adjuster`.

Os diretórios persistentes são:

```text
data/
├── cache/   # cache do Hugging Face
├── input/   # imagens e áudios de entrada
├── models/  # pesos; não fazem parte da imagem
├── output/  # vídeos gerados
└── user/    # configurações e workflows do ComfyUI
```

## Modos de modelo

Edite `.env` antes de executar `make models`:

| `MODEL_MODE` | DiT | Característica |
| --- | --- | --- |
| `official_int8_sharded` | Oficial INT8, 4 shards | Padrão recomendado; cerca de 16 GB para o DiT |
| `official_sharded` | Oficial BF16, 6 shards | Maior qualidade/precisão original; cerca de 32 GB para o DiT |
| `single_file_safetensors` | Conversão comunitária INT8 | Arquivo único de cerca de 16 GB |

Para o texto, `TEXT_ENCODER_MODE=clip_fp8` baixa o UMT5 FP8 usado pelo workflow
de exemplo. `TEXT_ENCODER_MODE=native` baixa o tokenizer/text encoder oficial
fragmentado, que ocupa aproximadamente 22 GB adicionais. Para usar o modo
`native`, remova/desconecte o node `Load CLIP` do workflow e use o carregamento
nativo do node `LongCat Avatar Text Encode`. Ao trocar `MODEL_MODE`, apague
apenas o workflow semeado em `data/user/default/workflows/` para ele ser recriado
com a nova seleção; os modelos já baixados não são apagados.

O downloader também instala, nos nomes esperados pelo ComfyUI:

- LoRA DMD obrigatória para inferência em 8 passos;
- VAE do LongCat Video;
- Whisper large-v3 para condicionamento de áudio;
- Kim Vocal 2 para separação vocal, salvo se `INCLUDE_VOCAL_SEPARATOR=0`.

Downloads interrompidos podem ser retomados repetindo `make models`. Os
repositórios e revisões dos modelos estão fixados no script para
reprodutibilidade. Se o Hugging Face exigir autenticação, defina `HF_TOKEN` em
`.env`.

## Operação

```bash
make logs       # acompanhar inicialização e gerações
make verify     # conferir a presença dos arquivos essenciais
make down       # parar o ambiente sem apagar modelos ou saídas
```

## Imagem publicada e Vast.ai

Todo push em `main` que altera o Dockerfile, Compose, scripts ou a própria
pipeline constrói uma imagem `linux/amd64` e publica no GitHub Container
Registry com SBOM e attestação de procedência:

```text
ghcr.io/glauber-fullstackdev/ambienteavatar:1.5
ghcr.io/glauber-fullstackdev/ambienteavatar:latest
```

Para hosts do Vast.ai que não processam corretamente índices OCI com
attestations, a mesma pipeline também publica imagens `linux/amd64` sem os
manifestos auxiliares. O conteúdo, o CUDA e as dependências são os mesmos:

```text
ghcr.io/glauber-fullstackdev/ambienteavatar:vast
ghcr.io/glauber-fullstackdev/ambienteavatar:1.5-vast
```

A pipeline também pode ser executada manualmente em **Actions → Build and
publish Docker image → Run workflow**, informando uma tag adicional. Depois da
primeira publicação, configure a visibilidade do pacote como pública no GitHub
para que o Vast.ai possa puxá-lo sem credenciais.

No Vast.ai, edite o template **NVIDIA CUDA** ou crie um template a partir de
[`vast/longcat-avatar-template.json`](vast/longcat-avatar-template.json). Use a
tag `:vast` em modo `Entrypoint/Args`, argumento `serve`, porta `8188`, CUDA
`>=12.8` e pelo menos 100 GB de disco. Para o modo INT8, uma GPU de 24 GB pode
ser usada com streaming e offload; 48 GB ou mais é a opção recomendada.

Campos principais do template:

```text
Image: ghcr.io/glauber-fullstackdev/ambienteavatar
Tag: vast
Launch mode: Entrypoint/Args
Args: serve
Docker options: -p 8188:8188 -e COMFYUI_PORT=8188 -e COMFYUI_ARGS="--preview-method auto" -e DEFAULT_WEIGHT_MODE=official_int8_sharded -e MODEL_MODE=official_int8_sharded -e TEXT_ENCODER_MODE=clip_fp8 -e INCLUDE_VOCAL_SEPARATOR=1 -e DOWNLOAD_MODELS_ON_START=1 -e HF_HOME=/opt/ComfyUI/models/.cache/huggingface
Disk: 120 GB recomendado
Persistent volume: /opt/ComfyUI/models
```

Para usar JupyterLab/SSH no Vast.ai, crie o template a partir de
[`vast/longcat-avatar-jupyter-template.json`](vast/longcat-avatar-jupyter-template.json)
ou preencha:

```text
Image: ghcr.io/glauber-fullstackdev/ambienteavatar
Tag: vast
Launch mode: Jupyter-python notebook + SSH
Use JupyterLab: yes
Direct connection: yes
Jupyter directory: /opt/ComfyUI
Docker options: -p 8188:8188 -e COMFYUI_HOME=/opt/ComfyUI -e COMFYUI_MODELS=/opt/ComfyUI/models -e COMFYUI_PORT=8188 -e COMFYUI_ARGS="--preview-method auto" -e DEFAULT_WEIGHT_MODE=official_int8_sharded -e MODEL_MODE=official_int8_sharded -e TEXT_ENCODER_MODE=clip_fp8 -e INCLUDE_VOCAL_SEPARATOR=1 -e DOWNLOAD_MODELS_ON_START=1 -e HF_HOME=/opt/ComfyUI/models/.cache/huggingface
On-start script:
env >> /etc/environment || true
mkdir -p /opt/ComfyUI/models /opt/ComfyUI/input /opt/ComfyUI/output /opt/ComfyUI/user /opt/ComfyUI/models/.cache/huggingface /tmp/longcat
cd /opt/ComfyUI
nohup python /opt/avatar-scripts/entrypoint.py serve > /tmp/longcat/comfyui.log 2>&1 &
```

A imagem baixa e verifica os pesos no boot por padrão
(`DOWNLOAD_MODELS_ON_START=1`) e só inicia o ComfyUI depois que o downloader
terminar. Os downloads são grandes; depois que o volume persistente já tiver os
modelos, use `DOWNLOAD_MODELS_ON_START=0` se quiser inicializações mais rápidas.

No compose local, o serviço só publica em `127.0.0.1` por padrão. Em um
servidor remoto fora do Vast.ai, prefira um túnel SSH. Se realmente precisar
expor diretamente, altere `COMFYUI_BIND=0.0.0.0` e proteja a porta com firewall
e autenticação reversa.

## Aceleração opcional

O padrão `sdpa` funciona usando apenas o PyTorch da imagem. Para uma máquina já
conhecida, é possível construir com:

```dotenv
INSTALL_XFORMERS=1
# ou, para GPUs/toolchains compatíveis:
INSTALL_FLASH_ATTN=1
```

Depois rode `make build` novamente e selecione o backend correspondente no node
de carregamento. FlashAttention e SageAttention não são requisitos funcionais;
as extensões são sensíveis à combinação de GPU, CUDA e PyTorch. O node pack
também informa que `sageattn_3` ainda não possui o caminho completo de
cross-attention do LongCat. Para compilar FlashAttention, altere também
`PYTORCH_IMAGE` para `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`; a imagem
`runtime` padrão não inclui o compilador CUDA.

## Versões fixadas

- PyTorch 2.8.0 + CUDA 12.8 + cuDNN 9
- ComfyUI: `c1739380c6fab78e7e263cb665d04aafbfe24593`
- ComfyUI-LongCat-Avatar: `08b4daedfaed69abaf467097f8665615b2137331`
- ComfyUI Manager: `4f56cf3dfa7de5d8a8614dfe202ff8d613ba2244`
- ComfyUI-VideoHelperSuite: `4ee72c065db22c9d96c2427954dc69e7b908444b`
- ComfyUI-AdvancedLivePortrait: `3bba732915e22f18af0d221b9c5c282990181f1b`
- ComfyUI-LatentSyncWrapper: `360d5283d7276aee68b4237b1387e594e4ce640e`
- NumPy: `2.2.6`

Esses pins tornam a imagem repetível. Atualize-os de forma deliberada no
`Dockerfile`, reconstrua e valide o workflow antes de usar uma versão mais nova.
