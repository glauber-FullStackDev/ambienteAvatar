# InfiniteTalk + ComfyUI no Vast.ai

Imagem CUDA separada do LongCat, baseada nos workflows I2V e V2V oficiais do
`ComfyUI-WanVideoWrapper`. Os pesos nao entram na imagem Docker: ficam no volume
persistente montado em `/opt/ComfyUI/models`.

## Uso local

```bash
cd infinitetalk
make build
make models
make verify
make up
```

Os workflows `infinitetalk-i2v-docker.json` e
`infinitetalk-v2v-docker.json` sao criados automaticamente em
`user/default/workflows`. O V2V ja vem configurado para salvar o MP4 em
`output/InfiniteTalk_V2V_*.mp4`. Nessa saida, o video original continua sendo
usado como guia, mas somente os frames gerados sao salvos; a comparacao lado a
lado do workflow oficial fica desativada. O workflow V2V e semeado a partir do
preset preservado em `defaults/workflows/infinitetalk-v2v-docker.json`, sem
reescrever os parametros iniciais. Ele abre com Q8, LoRA Lightx2v rank256,
`audio_scale=1.3`, `audio_cfg_scale=1`, 832x480, 81 frames, 4 steps e CRF 16. O
padrao `q8` acompanha esse preset; altere `MODEL_QUANTIZATION` para `q4_k_m` ou
`q6_k` antes de baixar os modelos somente se tambem for ajustar o workflow para
os arquivos correspondentes.

O downloader instala as LoRAs Lightx2v I2V rank64, rank128 e rank256. A rank256
e o padrao do workflow V2V preservado; as outras duas ficam disponiveis no node
`WanVideo LoRA Select` para comparacao com a mesma seed e os mesmos parametros.

## Vast.ai

Use a imagem:

```text
ghcr.io/glauber-fullstackdev/ambienteavatar-infinitetalk:vast
```

Depois da primeira publicacao, deixe o pacote
`ambienteavatar-infinitetalk` publico no GitHub Packages para o Vast.ai poder
baixa-lo sem credenciais.

Configure o container em modo `Entrypoint/Args`, argumento `serve`, porta 8188,
CUDA 12.8 ou superior e um volume persistente em `/opt/ComfyUI/models`. Para o
primeiro teste, defina `DOWNLOAD_MODELS_ON_START=1`; os downloads sao grandes e
o ComfyUI so inicia depois que terminarem. Nas proximas inicializacoes, volte a
variavel para `0`.

Variaveis principais:

- `MODEL_QUANTIZATION=q8`, `q6_k` ou `q4_k_m`;
- `INFINITETALK_AUDIO_SCALE=1.5` controla a forca do condicionamento de audio em workflows gerados a partir do exemplo oficial;
- `INFINITETALK_AUDIO_CFG_SCALE=1.2` controla o guidance especifico de audio em workflows gerados a partir do exemplo oficial;
- `INFINITETALK_OUTPUT_CRF=16` controla a compressao do MP4 (menor e melhor);
- `HF_TOKEN`, caso algum arquivo exija autenticacao;
- `COMFYUI_ARGS=--preview-method auto`;
- `DOWNLOAD_MODELS_ON_START=0` por padrao.

O preset Q8 se beneficia de mais VRAM. Q4 ainda e adequado para validar o fluxo
em uma GPU de 24 GB usando block swap, desde que o workflow tambem seja trocado
para os arquivos Q4. Reserve pelo menos 100 GB de disco para imagem, cache,
modelos, entradas e saidas.
