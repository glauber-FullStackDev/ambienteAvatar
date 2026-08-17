# LongCat Avatar 1.5 + ComfyUI em Docker

Ambiente CUDA reproduzível com ComfyUI, ComfyUI Manager, o node pack
`ComfyUI-LongCat-Avatar`, FFmpeg e todas as dependências Python de execução.
Os pesos não entram na imagem: ficam em `data/models` e podem ser baixados pelo
comando fornecido.

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

A pipeline também pode ser executada manualmente em **Actions → Build and
publish Docker image → Run workflow**, informando uma tag adicional. Depois da
primeira publicação, configure a visibilidade do pacote como pública no GitHub
para que o Vast.ai possa puxá-lo sem credenciais.

No Vast.ai, edite o template **NVIDIA CUDA** e use a imagem acima em modo
`Entrypoint/Args`, argumento `serve`, porta `8188`, CUDA `>=12.8` e pelo menos
100 GB de disco. Para o modo INT8, uma GPU de 24 GB pode ser usada com streaming
e offload; 48 GB ou mais é a opção recomendada.

O serviço só publica em `127.0.0.1` por padrão. Em um servidor remoto, prefira
um túnel SSH. Se realmente precisar expor diretamente, altere
`COMFYUI_BIND=0.0.0.0` e proteja a porta com firewall e autenticação reversa.

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

Esses pins tornam a imagem repetível. Atualize-os de forma deliberada no
`Dockerfile`, reconstrua e valide o workflow antes de usar uma versão mais nova.
