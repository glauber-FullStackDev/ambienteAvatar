# LTX 2.3/2.5 IA2V + ComfyUI para Vast.ai

Imagem independente para executar dois workflows oficiais do ComfyUI e quatro
adaptacoes prontas:

- `video_ltx2_3_ia2v.json`, para gerar video a partir de imagem, audio e prompt;
- `video_ltx2_3_id_lora.json`, para transferir a identidade vocal de um audio
  de referencia e a aparencia de uma imagem para uma nova fala descrita no
  prompt;
- `video_ltx2_3_ia2v_talkvid.json`, que preserva a narracao do IA2V e aplica o
  TalkVid 3K como reforco de identidade no primeiro estagio.
- `video_ltx2_3_ia2v_best_face.json`, que preserva a narracao original e usa o
  Best Face-ID v1.0 como referencia visual separada para reforcar o rosto.
- `video_ltx2_3_ia2v_ingredients.json`, que preserva a narracao original e usa
  uma reference sheet IC-LoRA Ingredients para reforcar identidade, roupa,
  props e ambiente no primeiro estagio.
- `video_ltx2_5_ia2v_distilled_8steps.json`, que combina imagem e narracao no
  transformer LTX-2.5 destilado INT8 ConvRot com a agenda oficial de 8 passos.

O ComfyUI, os workflows e as revisoes dos modelos ficam fixados. Os pesos nao
entram na imagem Docker: no primeiro boot eles sao baixados em
`/opt/ComfyUI/models`, validados por tamanho e mantidos no volume persistente.
O ComfyUI so inicia depois que os quatorze arquivos selecionados estiverem
completos.

## Conteudo

- ComfyUI no commit `a25c7bf2b8c7408d8724f4245dbe09d95992e3a1`,
  posterior ao suporte nativo ao LTX-2.5;
- workflow oficial no commit
  `d11b69157009227ad2a7d3a927a1eb68a3d5f281` e SHA256
  `7823a703f472d9c5e6f82c462235ff89a0fa14752ec1fd947c4422cf53e47685`;
- workflow ID-LoRA oficial no commit
  `04f33569dad7a1d277429bda9f35209dfa4d91cf` e SHA256
  `fcffe421129bac16b4f0655e54130d633280cdaf6949e145221e7090be42151f`;
- workflow LTX-2.5 I2V oficial no commit
  `1121504798345b1bb4e6350991f90512c4ba1ed9` e SHA256
  `0a88024394467250013ce611ee46d01cc7e73078a0899e4c80709080c5101f71`,
  usado como base da adaptacao IA2V;
- PyTorch 2.8.0, CUDA 12.8 e cuDNN 9;
- ComfyUI-BFSNodes no commit
  `0a2553869254eef4f3f735fdd9fea04614c3dd7e`, necessario ao condicionamento
  Best Face-ID;
- ComfyUI-LTXVideo no commit
  `15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d`, necessario aos nodes IC-LoRA
  Ingredients;
- patch de compatibilidade do ComfyUI-LTXVideo para `kornia>=0.8.3`, removendo
  o import quebrado de `pad` e usando `torch.nn.functional.pad`;
- FFmpeg, JupyterLab e todos os nodes usados pelos workflows;
- downloader retomavel via Hugging Face Hub, com repositorios e revisoes
  imutaveis.

Os nodes LTX IA2V/ID-LoRA sao nativos da versao fixada do ComfyUI. Os custom
nodes adicionais sao BFSNodes, para Best Face-ID, e ComfyUI-LTXVideo, para
IC-LoRA Ingredients.

O build valida tambem os imports de `torch`, `torchvision` e `torchaudio`, para
evitar publicar uma imagem com pacotes desalinhados.

## Modelos baixados na inicializacao

| Pasta em `/opt/ComfyUI/models` | Arquivo | Tamanho aproximado |
| --- | --- | ---: |
| `checkpoints/` | `ltx-2.3-22b-dev-fp8.safetensors` | 27,1 GiB |
| `text_encoders/` | `gemma_3_12B_it_fp4_mixed.safetensors` | 8,8 GiB |
| `loras/` | `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` | 2,6 GiB |
| `loras/` | `ltx-2.3-id-lora-talkvid-3k.safetensors` | 1,1 GiB |
| `loras/` | `Best_FaceID_v1.0_LoRA.safetensors` | 2,3 GiB |
| `loras/` | `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors` | 1,2 GiB |
| `loras/` | `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` | 0,6 GiB |
| `latent_upscale_models/` | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 0,9 GiB |
| `diffusion_models/` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 20,0 GiB |
| `text_encoders/` | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 14,3 GiB |
| `text_encoders/` | `gemma4_e2b_it_int8_convrot.safetensors` | 4,8 GiB |
| `vae/` | `ltx-2.5-video-vae-bf16.safetensors` | 1,4 GiB |
| `vae/` | `ltx-2.5-audio-vae-bf16.safetensors` | 0,3 GiB |
| `latent_upscale_models/` | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 0,9 GiB |

O LTX-2.5 acrescenta `44.909.870.140` bytes, aproximadamente `41,8 GiB`.
Total exato dos modelos: `92.801.281.116` bytes, aproximadamente `86,4 GiB`.
Para pular todos os pesos LTX-2.5 no boot, configure
`DOWNLOAD_LTX25_MODELS_ON_START=0`; os workflows continuam instalados, mas o
workflow 2.5 so roda depois que esses pesos forem baixados.

O teste normal de boot valida o tamanho exato, sem reler 86,4 GiB a cada
inicializacao. Para uma auditoria integral dos checksums:

```bash
/opt/ltx23-scripts/container-entrypoint.sh verify --verify-sha256
```

Antes do primeiro boot com LTX-2.5, aceite os termos em
`https://huggingface.co/Lightricks/LTX-2.5` e forneca um token com acesso por
`-e HF_TOKEN=hf_...` nas opcoes Docker do template. O token nao entra na imagem.
Sem a autorizacao, o downloader interrompe corretamente antes de iniciar o
ComfyUI. Downloads interrompidos podem ser retomados reiniciando a instancia ou
executando:

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
/opt/ComfyUI/user/default/workflows/video_ltx2_3_ia2v_ingredients-docker.json
/opt/ComfyUI/user/default/workflows/video_ltx2_5_ia2v_distilled_8steps-docker.json
```

Se cada arquivo ja existir ele normalmente e preservado, mantendo ajustes feitos
na interface. A unica excecao e uma revisao de schema de um workflow adaptado:
nesse caso o arquivo antigo recebe um backup antes da versao corrigida ser
instalada. O preset IA2V oficial inicia em `1280x720`, 24 FPS e 9 segundos, com
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

### Como usar o IA2V + IC-LoRA Ingredients

No workflow `video_ltx2_3_ia2v_ingredients-docker.json`:

1. envie a imagem inicial no `First Frame / Main Face`;
2. envie uma sheet composta no `Ingredients Reference Sheet`;
3. envie a narracao completa no `Driving Audio`;
4. escreva o prompt em duas partes: `Reference sheet:` e `Generated video:`;
5. comece com os defaults `768x448`, 5 segundos, 24 FPS e seed fixa.

A sheet precisa ser uma unica imagem em fundo preto, sem texto visivel, com
paineis limpos da pessoa, roupa, corpo, close-up frontal, perfil ou 3/4, props
fixos e ambiente. O workflow redimensiona a sheet, repete a imagem pelo numero
de frames e aplica `LTXAddVideoICLoRAGuide` no primeiro estagio com
Ingredients strength `1.0`. Depois do primeiro estagio, `LTXVCropGuides` remove
os frames da sheet antes do upscale e do decode. A narracao original vai
diretamente ao MP4 final.

Esse workflow e experimental porque combina o IA2V oficial com o IC-LoRA
Ingredients. O modelo Ingredients foi treinado para reference sheet estatica,
121 frames, 24 FPS e 768x448; mudar muito esses valores pode reduzir a
fidelidade. Para outras resolucoes, mantenha largura e altura divisiveis por 32
e componha a sheet na mesma proporcao do video para evitar distorcao.

### Como usar o LTX-2.5 IA2V destilado

No workflow `video_ltx2_5_ia2v_distilled_8steps-docker.json`:

1. envie uma foto nitida no `Identity / First Frame`;
2. envie a narracao no `Driving Audio`;
3. descreva no prompt a pessoa falando, o ambiente, a iluminacao e o movimento;
4. ajuste `audio_start`, duracao, resolucao, FPS e seed;
5. mantenha `prompt_enhance=false` e execute.

O audio recortado e codificado pelo VAE de audio 2.5 e recebe noise mask `0`,
ficando congelado enquanto condiciona o video nas duas etapas. A faixa original
recortada vai diretamente ao `CreateVideo`, sem passar pelo decoder de audio do
modelo. A primeira etapa usa a agenda destilada fixa de 8 passos; a etapa de
upscale usa mais 3 passos, ambas com CFG de video/audio em `1`.

Esse IA2V e uma adaptacao local do I2V 2.5 oficial; ainda nao existe um template
IA2V 2.5 publicado pela Comfy-Org. Comece com 5 segundos. O prompt enhancer
permanece no grafo e inicia desligado, mas seu Gemma 4 E2B e baixado quando
`DOWNLOAD_LTX25_MODELS_ON_START=1`.

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

Use uma GPU com **48 GB de VRAM** para os presets de 720p. Recomenda-se tambem
64 GB ou mais de RAM do host, CUDA 12.8 ou superior e 160 GB de disco da
instancia. Anexe um volume persistente de no minimo 100 GB ao caminho exato
`/opt/ComfyUI/models`; **120 GB** oferece margem adequada para os 86,4 GiB de
pesos, downloads parciais e cache.

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
Docker options: -p 8188:8188 -e COMFYUI_HOME=/opt/ComfyUI -e COMFYUI_MODELS=/opt/ComfyUI/models -e COMFYUI_PORT=8188 -e COMFYUI_ARGS="--preview-method auto" -e DOWNLOAD_MODELS_ON_START=1 -e DOWNLOAD_LTX25_MODELS_ON_START=1 -e HF_HOME=/opt/ComfyUI/models/.cache/huggingface -e HF_TOKEN=hf_SEU_TOKEN
Disk: 160 GB
Persistent volume: 120 GB mounted at /opt/ComfyUI/models
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
