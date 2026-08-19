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
lado do workflow oficial fica desativada. Workflows V2V ja persistidos no volume
sao migrados automaticamente na proxima inicializacao. O preset V2V usa
`audio_scale=1.5`, `audio_cfg_scale=1.2` e salva em CRF 16. O reforco leve de
audio CFG faz uma passagem adicional sem condicionamento de audio e, portanto,
troca velocidade por resposta de movimento e sincronizacao. O padrao `q4_k_m` e o
ponto inicial para GPUs com menos VRAM; altere `MODEL_QUANTIZATION` para `q6_k`
ou `q8` antes de baixar os modelos se quiser priorizar qualidade.

O downloader instala as LoRAs Lightx2v I2V rank64, rank128 e rank256. A rank64
continua sendo o padrao do workflow; as outras duas ficam disponiveis no node
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

- `MODEL_QUANTIZATION=q4_k_m`, `q6_k` ou `q8`;
- `INFINITETALK_AUDIO_SCALE=1.5` controla a forca do condicionamento de audio;
- `INFINITETALK_AUDIO_CFG_SCALE=1.2` controla o guidance especifico de audio;
- `INFINITETALK_OUTPUT_CRF=16` controla a compressao do MP4 (menor e melhor);
- `HF_TOKEN`, caso algum arquivo exija autenticacao;
- `COMFYUI_ARGS=--preview-method auto`;
- `DOWNLOAD_MODELS_ON_START=0` por padrao.

O Q4 e adequado para validar o fluxo em uma GPU de 24 GB usando block swap. Q6
e Q8 se beneficiam de mais VRAM. Reserve pelo menos 100 GB de disco para imagem,
cache, modelos, entradas e saidas.
