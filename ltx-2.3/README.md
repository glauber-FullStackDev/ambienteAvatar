# LTX 2.3 IA2V + ComfyUI para Vast.ai

Imagem independente para executar o workflow oficial
`video_ltx2_3_ia2v.json` do ComfyUI. Ela recebe uma imagem e um audio e gera
video com audio e movimento sincronizados pelo LTX 2.3.

O ComfyUI, o workflow e as revisoes dos modelos ficam fixados. Os pesos nao
entram na imagem Docker: no primeiro boot eles sao baixados em
`/opt/ComfyUI/models`, validados por tamanho e mantidos no volume persistente.
O ComfyUI so inicia depois que os cinco arquivos estiverem completos.

## Conteudo

- ComfyUI no commit `2e47082c8ed1d1a0fe54add57f98b63433cfacbb`;
- workflow oficial no commit
  `d11b69157009227ad2a7d3a927a1eb68a3d5f281` e SHA256
  `7823a703f472d9c5e6f82c462235ff89a0fa14752ec1fd947c4422cf53e47685`;
- PyTorch 2.8.0, CUDA 12.8 e cuDNN 9;
- FFmpeg, JupyterLab e todos os nodes nativos usados pelo workflow;
- downloader retomavel via Hugging Face Hub, com repositorios e revisoes
  imutaveis.

Nao ha custom node LTX instalado. Esse template foi publicado pela propria
Comfy-Org para os nodes LTX nativos da versao fixada do ComfyUI.

## Modelos baixados na inicializacao

| Pasta em `/opt/ComfyUI/models` | Arquivo | Tamanho aproximado |
| --- | --- | ---: |
| `checkpoints/` | `ltx-2.3-22b-dev-fp8.safetensors` | 27,1 GiB |
| `text_encoders/` | `gemma_3_12B_it_fp4_mixed.safetensors` | 8,8 GiB |
| `loras/` | `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` | 2,6 GiB |
| `loras/` | `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` | 0,6 GiB |
| `latent_upscale_models/` | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 0,9 GiB |

Total exato: `42.958.104.950` bytes, aproximadamente `40,0 GiB`.

O teste normal de boot valida o tamanho exato, sem reler 40 GiB a cada
inicializacao. Para uma auditoria integral dos checksums:

```bash
/opt/ltx23-scripts/container-entrypoint.sh verify --verify-sha256
```

Se o Hugging Face limitar o download, adicione `-e HF_TOKEN=hf_...` nas opcoes
Docker do template. Downloads interrompidos podem ser retomados reiniciando a
instancia ou executando:

```bash
/opt/ltx23-scripts/container-entrypoint.sh download-models
```

## Workflow

Na primeira inicializacao, uma copia intacta aparece em:

```text
/opt/ComfyUI/user/default/workflows/video_ltx2_3_ia2v-docker.json
```

Se o arquivo ja existir ele nunca e sobrescrito, preservando ajustes feitos na
interface. O preset oficial inicia em `1280x720`, 24 FPS e 9 segundos, com
prompt enhancement local. Selecione sua imagem no `Load Image`, seu audio no
`Load Audio`, escreva a cena/prompt e enfileire o workflow. A imagem e o audio
de exemplo referenciados pelo template oficial nao sao necessarios.

Para outras resolucoes, mantenha largura e altura divisiveis por 32. O numero
de frames usado pelo modelo deve seguir a forma `8n + 1`; os controles do
workflow fazem esse ajuste a partir da duracao e do FPS.

## Vast.ai: configuracao recomendada

Use uma GPU com **48 GB de VRAM** para o preset oficial de 720p. Recomenda-se
tambem 64 GB ou mais de RAM do host, CUDA 12.8 ou superior e 100 GB de disco da
instancia. Anexe um volume persistente de no minimo 60 GB ao caminho exato
`/opt/ComfyUI/models`; 80 GB oferece uma margem melhor para cache e modelos
futuros.

Existem dois templates prontos:

- [`vast/ltx23-template.json`](vast/ltx23-template.json): somente ComfyUI,
  usando `Entrypoint/Args`;
- [`vast/ltx23-jupyter-template.json`](vast/ltx23-jupyter-template.json):
  ComfyUI mais JupyterLab e SSH.

Campos do template simples:

```text
Image: ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23
Tag: vast
Launch mode: Entrypoint/Args
Args: serve
Docker options: -p 8188:8188 -e COMFYUI_HOME=/opt/ComfyUI -e COMFYUI_MODELS=/opt/ComfyUI/models -e COMFYUI_PORT=8188 -e COMFYUI_ARGS="--preview-method auto" -e DOWNLOAD_MODELS_ON_START=1 -e HF_HOME=/opt/ComfyUI/models/.cache/huggingface
Disk: 100 GB
Persistent volume mount: /opt/ComfyUI/models
```

Para o template Jupyter, selecione `Jupyter-python notebook + SSH`, habilite
JupyterLab e conexao direta, use `/opt/ComfyUI` como diretorio e coloque no
campo **On-start script**:

```bash
env >> /etc/environment || true
mkdir -p /var/log/portal /opt/ComfyUI/models /opt/ComfyUI/input /opt/ComfyUI/output /opt/ComfyUI/user
nohup /opt/ltx23-scripts/container-entrypoint.sh serve >/tmp/ltx23-onstart.log 2>&1 &
```

No terminal do Jupyter, acompanhe download, verificacao e inicializacao com:

```bash
tail -n 200 -f /var/log/portal/comfyui.log
```

Quando aparecer `Starting server`/`To see the GUI`, abra a porta HTTP 8188 no
painel da instancia. Para verificar o processo e a API:

```bash
ps aux | grep '[m]ain.py'
curl -fsS http://127.0.0.1:8188/system_stats | python3 -m json.tool
```

## Build local

O build nao baixa os modelos, mas valida o JSON e faz smoke import dos nodes:

```bash
docker build -t ambienteavatar-ltx23:local ltx-2.3
```

Para executar com um volume nomeado:

```bash
docker run --gpus all --rm -p 8188:8188 \
  -v ltx23-models:/opt/ComfyUI/models \
  -v ltx23-user:/opt/ComfyUI/user \
  -v ltx23-input:/opt/ComfyUI/input \
  -v ltx23-output:/opt/ComfyUI/output \
  ambienteavatar-ltx23:local serve
```

## Imagem publicada

A pipeline separada publica:

```text
ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23:latest
ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23:vast
ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23:sha-<commit>
```

A tag `vast` e `linux/amd64` sem attestations auxiliares, para maximizar a
compatibilidade com o pull do Vast.ai. O pacote GHCR precisa estar publico.
