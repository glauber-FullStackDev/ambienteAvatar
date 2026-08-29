# Guia prático — primeiro treino da sua LoRA A2V

Este guia cria uma LoRA pessoal para **LTX-2.3 Audio-to-Video**. O áudio é a condição congelada: ele guia fala, ritmo e expressão; o vídeo é o que o modelo aprende a gerar.

## Antes de começar

Você precisa de:

- acesso ao Vast.ai;
- uma conta no Hugging Face com os termos aceitos para `Lightricks/LTX-2.3` e `google/gemma-3-12b-it-qat-q4_0-unquantized`;
- um token Hugging Face de leitura;
- seus vídeos MP4 com áudio sincronizado;
- legendas para cada vídeo;
- opcionalmente, uma conta W&B para acompanhar métricas e vídeos.

Espere o workflow GitHub Actions publicar a imagem antes de criar a instância. A imagem é:

```text
ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23-a2v:vast
```

## 1. Crie armazenamento persistente no Vast

Na área **Storage / Volumes** do Vast, crie um volume de **250–300 GB**. Ao alugar a GPU, use **Rent instance using this volume** e defina o ponto de montagem:

```text
/workspace
```

O volume guarda modelos, dataset, pré-processamento e checkpoints. Sem ele, você perde esses dados se destruir a instância.

Também deixe pelo menos **50 GB** de disco de container para a imagem e arquivos transitórios.

## 2. Crie o template Vast

Na criação/edição do template, preencha:

| Campo | Valor |
| --- | --- |
| Docker image | `ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23-a2v` |
| Tag | `vast` |
| Launch mode | **Jupyter-python notebook + SSH** |
| JupyterLab | habilitado |
| Direct Jupyter e Direct SSH | habilitados |
| Jupyter directory | `/workspace` |

Não preencha `JUPYTER_TOKEN`: neste modo o Vast é quem inicia e protege o JupyterLab. A imagem usa o **On-start Script** apenas para preparar o volume e baixar os modelos, sem iniciar um segundo Jupyter.

Em **Docker Options**, cole e substitua apenas o token Hugging Face:

```bash
-p 8080:8080 \
-e DOWNLOAD_MODELS_ON_START=1 \
-e LTX_MODELS_DIR=/workspace/models \
-e HF_HOME=/workspace/.cache/huggingface \
-e WANDB_DIR=/workspace/logs/wandb \
-e WANDB_PROJECT=ltx23-personal-a2v \
-e HF_TOKEN=COLE_SEU_TOKEN_HUGGINGFACE_AQUI
```

No campo **On-start script**, cole exatamente isto:

```bash
env >> /etc/environment || true
mkdir -p /workspace/logs
nohup /usr/local/bin/bootstrap_a2v.sh > /workspace/logs/bootstrap.log 2>&1 &
```

Prefira guardar `HF_TOKEN` como variável secreta da conta Vast, em vez de salvá-lo em um template público. O `WANDB_API_KEY` será definido no terminal somente antes do treino.

Escolha uma GPU de pelo menos **80 GB de VRAM** (A100 80GB ou H100) e CUDA/driver compatível com CUDA 13.2.

## 3. Primeiro boot e Jupyter

Inicie a instância. O primeiro boot baixa o LTX-2.3 e Gemma automaticamente para `/workspace/models`; pode demorar. Não encerre a instância enquanto esse download ocorre.

No painel da instância Vast, abra **Jupyter**. Não há token da imagem para copiar: o acesso é gerenciado pelo próprio Vast. No Jupyter, abra **Terminal** e acompanhe primeiro o download:

```bash
tail -n 200 -f /workspace/logs/bootstrap.log
```

Quando aparecer `[OK] Model downloads verified`, interrompa o `tail` com `Ctrl-c` e confirme o ambiente:

```bash
check_ltx_environment.sh
```

Prossiga somente quando GPU, CUDA, checkpoint LTX e Gemma aparecerem como `[OK]`.

## 4. Organize e envie o dataset

Envie seus arquivos pelo navegador de arquivos do Jupyter:

```text
/workspace/dataset/raw/
├── dataset.json
└── videos/
    ├── clip_0001.mp4
    ├── clip_0002.mp4
    └── ...
```

Não recomprima os vídeos. Para cada MP4, crie uma caption correspondente no `dataset.json`:

```json
[
  {
    "media_path": "videos/clip_0001.mp4",
    "caption": "Uma pessoa falando diretamente para a câmera, plano médio, iluminação suave, expressão natural."
  },
  {
    "media_path": "videos/clip_0002.mp4",
    "caption": "A mesma pessoa falando para a câmera, fundo neutro, movimentos faciais sutis e sincronização labial."
  }
]
```

As captions são importantes: descreva pessoa, enquadramento, luz, fundo, roupa e comportamento visual de forma consistente.

## 5. Defina o trigger da sua LoRA

Escolha um token único. Exemplo:

```text
glauberavatar
```

Evite palavras comuns e seu nome completo. Esse token será adicionado às captions durante o preprocessamento e deve aparecer nos prompts de validação/inferência.

## 6. Preprocesse o dataset

Para seus vídeos de 89 frames a 25 FPS e 768×448, execute:

```bash
export LTX_LORA_TRIGGER=glauberavatar
export LTX_RESOLUTION_BUCKET=89x448x768
preprocess_a2v.sh
```

O formato do bucket acima é:

```text
frames x altura x largura
```

O script extrai o áudio incorporado, gera latentes de vídeo, latentes de áudio e condições de texto em:

```text
/workspace/dataset/preprocessed/
├── latents/
├── audio_latents/
└── conditions/
```

Se precisar alterar trigger, captions ou bucket depois, recrie os artefatos:

```bash
export LTX_PREPROCESS_OVERWRITE=1
preprocess_a2v.sh
```

## 7. Configure um teste de validação

Envie um áudio seu para:

```text
/workspace/validation/audio/meu_audio_4s.mp3
```

Como os clipes de treino têm 89 frames a 25 FPS, crie um WAV **estéreo** de
3,56 segundos para a validação. O encoder de áudio do LTX espera dois canais;
`-ac 2` também duplica corretamente um áudio de voz que originalmente seja mono:

```bash
ffmpeg -i /workspace/validation/audio/meu_audio_4s.mp3 \
  -t 3.56 -ar 48000 -ac 2 \
  /workspace/validation/audio/meu_teste.wav
```

## 8. Crie a configuração de treino

```bash
cp /workspace/configs/a2v_personal.yaml.example \
  /workspace/configs/a2v_personal.yaml

nano /workspace/configs/a2v_personal.yaml
```

Em `validation.samples`, use seu trigger, um prompt visual e o áudio de teste.
Cada sample pode ter seu próprio `video_dims`; portanto, horizontal e vertical
podem ser validados na mesma rodada, sem parar ou duplicar o treino:

```yaml
validation:
  samples:
    - prompt: >-
        glauberavatar, horizontal video, falando diretamente para a câmera, plano médio,
        iluminação suave de estúdio, fundo neutro, expressão natural,
        movimentos faciais sincronizados com o áudio.
      conditions:
        - type: audio_to_video
          audio: /workspace/validation/audio/meu_teste.wav
      video_dims: [960, 544, 89] # largura, altura, frames

    - prompt: >-
        glauberavatar, vertical portrait video, falando diretamente para a câmera,
        plano médio, iluminação suave de estúdio, fundo neutro e expressão natural.
      conditions:
        - type: audio_to_video
          audio: /workspace/validation/audio/meu_teste.wav
      video_dims: [544, 960, 89]

  # Fallback somente para samples sem video_dims próprio.
  video_dims: [960, 544, 89]
  frame_rate: 25.0
  interval: 300
```

Mantenha obrigatoriamente:

```yaml
model:
  training_mode: lora

training_strategy:
  name: flexible
  video:
    is_generated: true
  audio:
    is_generated: false
```

O valor inicial é `steps: 2000`. Ele é um ponto de partida, não uma obrigação.

Se não for usar W&B, altere no YAML:

```yaml
wandb:
  enabled: false
```

## 9. Inicie o treino

No terminal:

```bash
tmux new -s ltx
train_a2v.sh
```

O script valida GPU, modelos, artefatos pré-processados, áudio de validação e as semânticas A2V antes de iniciar.

Para sair sem encerrar o treino, pressione `Ctrl-b` e depois `d`.

Para voltar:

```bash
tmux attach -t ltx
```

## 10. Acompanhe os resultados

No Jupyter, acompanhe:

```text
/workspace/outputs/personal_a2v/
├── checkpoints/
├── samples/
└── training_config.yaml
```

Os vídeos de validação aparecem a cada 300 steps por padrão. Se W&B estiver habilitado, veja também o projeto `ltx23-personal-a2v` no wandb.ai.

## 11. Continue além de 2.000 steps

Para continuar até 3.000 sem perder o estado:

1. Edite `steps: 3000` em `/workspace/configs/a2v_personal.yaml`.
2. Rode:

   ```bash
   resume_a2v.sh
   ```

3. Copie o comando mostrado, semelhante a:

   ```bash
   LTX_RESUME_CHECKPOINT=/workspace/outputs/personal_a2v/checkpoints/lora_weights_step_02000.safetensors \
     train_a2v.sh
   ```

Não altere rank, módulos LoRA ou semântica A2V entre treino e retomada.

## Checklist antes de clicar em treinar

```bash
check_ltx_environment.sh
test -s /workspace/dataset/raw/dataset.json
find /workspace/dataset/raw/videos -name '*.mp4' | head
find /workspace/dataset/preprocessed/latents -name '*.pt' | head
test -s /workspace/validation/audio/meu_teste.wav
```

Se esses comandos retornarem resultados esperados, execute `train_a2v.sh` dentro do `tmux`.
