# LTX 2.3 IA2V + ComfyUI para Vast.ai

Imagem independente para executar dois workflows oficiais do ComfyUI e duas
adaptacoes prontas:

- `video_ltx2_3_ia2v.json`, para gerar video a partir de imagem, audio e prompt;
- `video_ltx2_3_id_lora.json`, para transferir a identidade vocal de um audio
  de referencia e a aparencia de uma imagem para uma nova fala descrita no
  prompt;
- `video_ltx2_3_ia2v_talkvid.json`, que preserva a narracao do IA2V e aplica o
  TalkVid 3K como reforco de identidade no primeiro estagio.
- `video_ltx2_3_ia2v_best_face.json`, que preserva a narracao original e usa o
  Best Face-ID v1.0 como referencia visual separada para reforcar o rosto.

O ComfyUI, os workflows e as revisoes dos modelos ficam fixados. Os pesos nao
entram na imagem Docker: no primeiro boot eles sao baixados em
`/opt/ComfyUI/models`, validados por tamanho e mantidos no volume persistente.
O ComfyUI so inicia depois que os sete arquivos estiverem completos.

## Conteudo

- ComfyUI no commit `2e47082c8ed1d1a0fe54add57f98b63433cfacbb`;
- workflow oficial no commit
  `d11b69157009227ad2a7d3a927a1eb68a3d5f281` e SHA256
  `7823a703f472d9c5e6f82c462235ff89a0fa14752ec1fd947c4422cf53e47685`;
- workflow ID-LoRA oficial no commit
  `04f33569dad7a1d277429bda9f35209dfa4d91cf` e SHA256
  `fcffe421129bac16b4f0655e54130d633280cdaf6949e145221e7090be42151f`;
- PyTorch 2.8.0, CUDA 12.8 e cuDNN 9;
- ComfyUI-BFSNodes no commit
  `0a2553869254eef4f3f735fdd9fea04614c3dd7e`, necessario ao condicionamento
  Best Face-ID;
- FFmpeg, JupyterLab e todos os nodes usados pelos workflows;
- downloader retomavel via Hugging Face Hub, com repositorios e revisoes
  imutaveis.

Os nodes LTX sao nativos da versao fixada do ComfyUI. O unico custom node deste
ambiente e o BFSNodes, fixado para manter o workflow Best Face-ID reproduzivel.

## Modelos baixados na inicializacao

| Pasta em `/opt/ComfyUI/models` | Arquivo | Tamanho aproximado |
| --- | --- | ---: |
| `checkpoints/` | `ltx-2.3-22b-dev-fp8.safetensors` | 27,1 GiB |
| `text_encoders/` | `gemma_3_12B_it_fp4_mixed.safetensors` | 8,8 GiB |
| `loras/` | `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` | 2,6 GiB |
| `loras/` | `ltx-2.3-id-lora-talkvid-3k.safetensors` | 1,1 GiB |
| `loras/` | `Best_FaceID_v1.0_LoRA.safetensors` | 2,3 GiB |
| `loras/` | `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` | 0,6 GiB |
| `latent_upscale_models/` | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 0,9 GiB |

Total exato: `46.582.632.638` bytes, aproximadamente `43,4 GiB`.

O teste normal de boot valida o tamanho exato, sem reler 43,4 GiB a cada
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

Na primeira inicializacao, copias intactas aparecem em:

```text
/opt/ComfyUI/user/default/workflows/video_ltx2_3_ia2v-docker.json
/opt/ComfyUI/user/default/workflows/video_ltx2_3_id_lora-docker.json
/opt/ComfyUI/user/default/workflows/video_ltx2_3_ia2v_talkvid-docker.json
/opt/ComfyUI/user/default/workflows/video_ltx2_3_ia2v_best_face-docker.json
```

Se cada arquivo ja existir ele nunca e sobrescrito, preservando ajustes feitos na
interface. O preset IA2V oficial inicia em `1280x720`, 24 FPS e 9 segundos, com
prompt enhancement local. Selecione sua imagem no `Load Image`, seu audio no
`Load Audio`, escreva a cena/prompt e enfileire o workflow. A imagem e o audio
de exemplo referenciados pelo template oficial nao sao necessarios.

### Como usar o IA2V + TalkVid

No workflow `video_ltx2_3_ia2v_talkvid-docker.json`:

1. envie a imagem no `Load Image`;
2. envie a narracao completa no `Driving Audio`;
3. descreva em ingles a cena e uma articulacao labial natural no prompt;
4. ajuste `audio_start`, duracao, resolucao, FPS e seed e execute.

O audio recortado pela duracao escolhida tem tres funcoes: alimenta o
`LTXVAudioVAEEncode` do IA2V, fornece os primeiros 5 segundos ao
`LTXVReferenceAudio` e entra diretamente no `CreateVideo`. Assim, o MP4 usa a
narracao original em vez do audio reconstruido pelo VAE. Se `audio_start` for
alterado, o trecho de referencia TalkVid comeca no mesmo ponto.

Os valores prontos seguem o workflow ID-LoRA oficial: TalkVid strength `1.0`,
identity guidance `3.0`, start `0.0` e end `1.0`. O TalkVid e a referencia de
voz atuam apenas no primeiro estagio; o segundo continua usando somente o
distilled LoRA. Essa combinacao usa nodes oficiais, mas nao e um template
oficial publicado. Comece com 6 a 8 segundos e compare com o IA2V original.

### Como usar o IA2V + Best Face-ID

No workflow `video_ltx2_3_ia2v_best_face-docker.json`:

1. envie uma foto frontal, nitida e bem iluminada no `Identity Reference`;
2. envie a narracao completa no `Driving Audio`;
3. mantenha ligado `Best Face-ID reference mode`;
4. escreva em ingles um prompt iniciado por `ref_t2v:` que descreva cabelo,
   olhos, barba, oculos, formato do rosto, enquadramento, ambiente e acao;
5. escolha duracao, resolucao, FPS e seed e execute.

A foto entra como tokens de identidade separados e nao e forcada como o
primeiro frame do video. O audio recortado alimenta o encoder IA2V, enquanto a
faixa original vai diretamente ao MP4. Os valores iniciais sao: distilled LoRA
`0.6`, Best Face-ID `1.0`, source ID `2.0`, phase `1.0`, layout `overlap`,
reference guidance `1.0` e offset temporal `0`. O reforco Best Face-ID atua no
primeiro estagio; o segundo permanece no distilled LoRA.

Este workflow e uma adaptacao experimental da receita do autor do Best Face-ID
sobre o IA2V oficial. Comece com 6 a 8 segundos. O Best Face-ID melhora a
preservacao visual da identidade, mas o sincronismo labial continua vindo do
condicionamento de audio do IA2V.

### Como usar o ID-LoRA

No workflow `video_ltx2_3_id_lora-docker.json`:

1. envie uma imagem nitida do personagem no `Load Image`;
2. envie no `Load Audio` uma amostra limpa da voz cuja identidade sera
   transferida; o node nativo recomenda aproximadamente 5 segundos;
3. preencha o prompt com as secoes abaixo e execute.

```text
[VISUAL]: A cena, aparencia, enquadramento, iluminacao e acao. Diga que a pessoa esta falando.
[SPEECH]: As palavras exatas que a pessoa deve falar.
[SOUNDS]: Tom de voz, volume, proximidade do microfone e sons do ambiente.
```

O audio enviado ao ID-LoRA e uma **referencia de identidade vocal**, nao uma
faixa que sera copiada literalmente para a saida. O texto efetivamente falado
vem de `[SPEECH]`. Para preservar um audio completo palavra por palavra seria
necessario outro tipo de pipeline, dirigido diretamente pelo audio.

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

O build nao baixa os modelos, mas valida o JSON e faz smoke import dos nodes
nativos e do BFSNodes:

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
