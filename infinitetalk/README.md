# InfiniteTalk + ComfyUI no Vast.ai

Imagem CUDA separada do LongCat, baseada no workflow I2V oficial do
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

O workflow `infinitetalk-i2v-docker.json` e criado automaticamente em
`user/default/workflows`. O padrao `q4_k_m` e o ponto inicial para GPUs com
menos VRAM; altere `MODEL_QUANTIZATION` para `q6_k` ou `q8` antes de baixar os
modelos se quiser priorizar qualidade.

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
- `HF_TOKEN`, caso algum arquivo exija autenticacao;
- `COMFYUI_ARGS=--preview-method auto`;
- `DOWNLOAD_MODELS_ON_START=0` por padrao.

O Q4 e adequado para validar o fluxo em uma GPU de 24 GB usando block swap. Q6
e Q8 se beneficiam de mais VRAM. Reserve pelo menos 100 GB de disco para imagem,
cache, modelos, entradas e saidas.
