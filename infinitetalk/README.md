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

Um terceiro preset, `infinitetalk-v2v-latentsync16-docker.json`, e criado ao
lado dos dois workflows existentes. Ele parte do V2V preservado e acrescenta o
LatentSync 1.6 depois do `WanVideoDecode`, quando a saida do sampler ja voltou a
ser uma sequencia `IMAGE`, e antes do `VHS_VideoCombine`. O workflow V2V
original nao e alterado. A saida refinada usa o prefixo
`InfiniteTalk_V2V_LatentSync16`, 25 FPS e o mesmo CRF 16 do preset base.
Na inicializacao, cada preset so e copiado quando o arquivo de destino ainda
nao existe; workflows ja editados pelo usuario nao sao regravados.

O node abre com o modelo oficial LatentSync 1.6 de 512x512, seed 1247,
`lips_expression=1.5` e 20 passos. Esses sao parametros exclusivos da etapa de
acabamento; Q8, LoRA Lightx2v rank256, escalas de audio, resolucao, quantidade
de frames e sampler do InfiniteTalk permanecem iguais aos do V2V salvo.

Um quarto preset, `infinitetalk-v2v-latentsync16-stable-docker.json`, mantem
todos os anteriores e troca apenas o acabamento pelo `LatentSyncStableNode`.
Ele abre com `lips_expression=1.8`, 20 passos e controles conservadores para
reduzir manchas durante movimento de cabeca. A saida usa o prefixo
`InfiniteTalk_V2V_LatentSync16_Stable`, H.264 MP4, 25 FPS e CRF 16.

O node estabilizado expoe os seguintes grupos de ajuste:

- transformacao facial: modo (`median_gaussian`, `median` ou `gaussian`),
  janela, forca e chaves independentes para translacao, rotacao e escala;
- limites de seguranca: correcao maxima de translacao em pixels, rotacao em
  graus e escala em fracao;
- composicao: expansao ou contracao da mascara, feather e opacidade;
- protecao de movimento: limiar, sensibilidade, intensidade minima aplicada
  pelo LatentSync e suavizacao temporal;
- protecao de pose: mede o yaw com os landmarks 3D do InsightFace, entra em
  fallback acima de 25 graus e so retoma o LatentSync abaixo de 18 graus;
  durante o fallback usa integralmente o frame original do InfiniteTalk para
  impedir a sobreposicao de duas bocas;
- nucleo da boca: raio e forca preservada para manter o lip sync mesmo quando
  a borda da mascara e reduzida;
- acabamento: blur proporcional ao movimento e correspondencia de cor na
  transicao entre a regiao gerada e o rosto original;
- `debug_log`, que registra no log do ComfyUI os escores de movimento e a menor
  intensidade de composicao encontrada.

Os controles so ficam ativos dentro do `LatentSyncStableNode`. O
`LatentSyncNode` original continua com o comportamento do wrapper fixado. Para
isolar uma causa, use `0` na forca correspondente: estabilizacao, feather,
blur ou correspondencia de cor. `motion_protection=false` desliga toda a
reducao adaptativa durante movimento.

O preset Stable abre pronto com `lips_expression=1.8`, 20 passos,
`pose_protection=true`, `max_head_yaw=25`, `resume_head_yaw=18` e dois frames
de guarda em cada lado do fallback. O boot migra somente o node Stable de
workflows persistidos em schemas anteriores, preservando o restante do grafo.
Na leitura do MP4 final, falhas `Errno 11` do conversor PyAV/swscale acionam
automaticamente uma segunda leitura via FFmpeg limitada a uma thread, evitando
perder uma inferencia concluida por esgotamento temporario de recursos.

Os dois presets InfiniteTalk com LatentSync 1.6 incluem prompts positivo e
negativo voltados a manter o rosto frontal, movimentos de cabeca pequenos,
camera estavel e expressoes naturais. Os presets InfiniteTalk sem LatentSync
continuam com os prompts originais vazios. Na inicializacao, a imagem tambem
preenche esses prompts em workflows LatentSync ja semeados quando os campos
ainda estao vazios; qualquer prompt personalizado pelo usuario e preservado.

O downloader instala as LoRAs Lightx2v I2V rank64, rank128 e rank256. A rank256
e o padrao do workflow V2V preservado; as outras duas ficam disponiveis no node
`WanVideo LoRA Select` para comparacao com a mesma seed e os mesmos parametros.
Ele tambem instala no volume persistente `models/latentsync` o UNet oficial do
LatentSync 1.6, Whisper tiny, VAE do Stable Diffusion, o detector facial S3FD e
o pacote InsightFace `buffalo_l`. Assim, o node nao precisa buscar checkpoints
durante a primeira geracao.
A imagem inclui InsightFace 1.0.1 e ONNX Runtime GPU 1.26.0, fixado na ultima
serie compativel com CUDA 12.8. O smoke test de build importa o pipeline de
inferencia completo e exige o provider `CUDAExecutionProvider`.
O runtime inclui cuBLAS 12.8 e os extras CUDA/cuDNN do ONNX, carregados com
`preload_dlls()` antes de o InsightFace criar as sessoes dos modelos.
O detector usa explicitamente
`/opt/ComfyUI/models/latentsync/auxiliary/models/buffalo_l`, sem depender do
diretorio de trabalho do processo ComfyUI.

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

Todo o boot, download de modelos e log do ComfyUI e duplicado em
`/var/log/portal/comfyui.log`. Dentro do terminal da instancia Vast, acompanhe
com:

```bash
tail -n 200 -f /var/log/portal/comfyui.log
```

Com `DOWNLOAD_MODELS_ON_START=1`, o mesmo boot baixa tanto os modelos do
InfiniteTalk quanto os do LatentSync 1.6. Todos ficam no volume montado em
`/opt/ComfyUI/models`, portanto reiniciar ou atualizar a imagem nao repete os
downloads concluidos.

Variaveis principais:

- `MODEL_QUANTIZATION=q8`, `q6_k` ou `q4_k_m`;
- `INFINITETALK_AUDIO_SCALE=1.5` controla a forca do condicionamento de audio em workflows gerados a partir do exemplo oficial;
- `INFINITETALK_AUDIO_CFG_SCALE=1.2` controla o guidance especifico de audio em workflows gerados a partir do exemplo oficial;
- `INFINITETALK_OUTPUT_CRF=16` controla a compressao do MP4 (menor e melhor);
- `HF_TOKEN`, caso algum arquivo exija autenticacao;
- `COMFYUI_ARGS=--preview-method auto`;
- `DOWNLOAD_MODELS_ON_START=0` por padrao.

O LatentSync e executado depois do modelo Wan e pode elevar o pico de VRAM e o
tempo total. Para o workflow combinado, 48 GB de VRAM e a opcao recomendada;
em 24 GB, valide primeiro com 81 frames e mantenha o block swap do preset.

O preset Q8 se beneficia de mais VRAM. Q4 ainda e adequado para validar o fluxo
em uma GPU de 24 GB usando block swap, desde que o workflow tambem seja trocado
para os arquivos Q4. Reserve pelo menos 100 GB de disco para imagem, cache,
modelos, entradas e saidas. Para manter tambem os checkpoints do LatentSync e
folga para arquivos temporarios, prefira 120 GB ou mais no template Vast.ai.
