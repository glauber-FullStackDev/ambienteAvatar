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

Nos dois presets LatentSync, o audio continua passando pelo MelBand RoFormer e
pelo Wav2Vec para condicionar a animacao. Na saida, porem, o
`VHS_VideoCombine` recebe diretamente o `input_audio` original, sem usar o
audio de 16 kHz devolvido pelo LatentSync. O boot corrige essa ligacao tambem
nos workflows ja persistidos, mas somente quando a entrada de audio do combine
ainda estiver conectada ao node LatentSync; uma ligacao personalizada e
preservada.

Um quarto preset, `infinitetalk-v2v-latentsync16-stable-docker.json`, mantem
todos os anteriores e troca apenas o acabamento pelo `LatentSyncStableNode`.
Ele abre com `lips_expression=1.6`, 25 passos, `denoise_strength=1` e os
controles validados no preset de producao. A saida usa o prefixo
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

O preset Stable abre pronto com `audio_scale=1`, `audio_cfg_scale=1`,
`lips_expression=1.6`, 25 passos, `mask_expand=5`, `mask_feather=8`,
`motion_min_strength=1`, `mouth_core_radius=5`, `pose_protection=true`,
`max_head_yaw=25`, `resume_head_yaw=18` e dois frames de guarda em cada lado
do fallback. O boot migra os parametros versionados dos nodes Stable e
MultiTalk em workflows persistidos de schemas anteriores, preservando o
restante do grafo.
Na leitura do MP4 final, falhas `Errno 11` do conversor PyAV/swscale acionam
automaticamente uma segunda leitura limitada a uma thread. Para videos
`yuv420p`, o FFmpeg entrega os planos YUV sem conversao e o RGB e calculado no
Python, sem voltar a acionar o `swscale` que causou a falha. O fallback usa
`-vsync 0`, compativel com o FFmpeg antigo presente nas imagens Vast.

Os dois presets InfiniteTalk com LatentSync 1.6 incluem prompts positivo e
negativo voltados a manter o rosto frontal, movimentos de cabeca pequenos,
camera estavel e expressoes naturais. Os dois presets base sem LatentSync
continuam com os prompts originais vazios; o novo Stable sem LatentSync herda
os prompts de producao do Stable. Na inicializacao, a imagem tambem preenche
esses prompts em workflows LatentSync ja semeados quando os campos ainda estao
vazios; qualquer prompt personalizado pelo usuario e preservado.

Um quinto preset,
`infinitetalk-v2v-stable-no-latentsync-docker.json`, copia todos os parametros
de producao do Stable, mas remove integralmente o `LatentSyncStableNode`. A
saida do `WanVideoDecode` segue diretamente para o ajuste final da quantidade
de frames e para o `VHS_VideoCombine`. Ele preserva Q8, LoRA Lightx2v rank256,
prompts, sampler de quatro passos, seed 2, 480x832, `audio_scale=1`,
`audio_cfg_scale=1`, 25 FPS e CRF 16. O audio original continua ligado ao VHS.
A saida usa o prefixo `InfiniteTalk_V2V_Stable_NoLatentSync`.

## Lip Forcing 14B local

O preset separado `lipforcing14b-video-audio-docker.json` recebe um MP4 e um
arquivo de audio enviados pelo navegador. Ele nao carrega o video como batch
de imagens: os nodes de entrada devolvem caminhos validados dentro de
`/opt/ComfyUI/input`, evitando uma decodificacao e recompressao antes do modelo.

O `LipForcing14B` executa o codigo oficial fixado em um ambiente Python 3.12
isolado. Antes de iniciar, ele descarrega todos os modelos mantidos pelo
ComfyUI e limpa o cache CUDA. O preset usa BF16, seed 42, 25 FPS, decoder
`streaming_taehv`, schedule destilado `0.999 -> 0.769 -> 0`, chunks de tres
latentes e a janela de atencao treinada de sete latentes com sink 1. A saida
usa o prefixo `LipForcing14B_Final` e aparece no preview do ComfyUI.

O modo `exact_audio_duration=true` arredonda a geracao para cima ate um chunk
completo e depois corta o MP4 na duracao exata do audio. O mux final usa o
arquivo de audio selecionado pelo usuario, codificado em AAC 256 kbps; o audio
do video de referencia e ignorado.

O checkpoint publico 14B ocupa 28,6 GB. Com VAE, Wav2Vec2, TAEHV, mascara e
embedding, reserve cerca de 32 GB adicionais no volume, alem do espaco
temporario usado durante o primeiro preparo. O UMT5-XXL de 11,4 GB e baixado
temporariamente para gerar o embedding fixo de `a person talking` em CPU e e
removido depois da validacao. Isso mantem a inferencia em aproximadamente 37
GB de VRAM; o preset exige uma GPU de 48 GB ou mais. Placas de 24 ou 32 GB sao
bloqueadas, salvo quando `LIPFORCING_ALLOW_LOW_VRAM=1` for definido
explicitamente para um teste nao suportado.

O comando proprio do Lip Forcing tambem verifica e, se necessario, baixa o
`buffalo_l` usado na deteccao e no alinhamento facial. Portanto, ele pode ser
executado mesmo antes do downloader geral do InfiniteTalk.

Baixar e verificar somente os modelos do Lip Forcing pelo Terminal do Jupyter:

```bash
/opt/infinitetalk-scripts/container-entrypoint.sh download-lipforcing-models
/opt/infinitetalk-scripts/container-entrypoint.sh verify-lipforcing
```

Para baixar durante o boot, use
`DOWNLOAD_LIPFORCING_MODELS_ON_START=1`. Essa variavel e separada de
`DOWNLOAD_MODELS_ON_START`, portanto atualizar a imagem nao inicia por engano
um download adicional de aproximadamente 32 GB.

Durante a execucao, as mensagens do subprocesso recebem o prefixo
`[LipForcing14B]` e sao gravadas no mesmo
`/var/log/portal/comfyui.log`. O Lip Forcing substitui o gerador do LatentSync,
mas o projeto oficial ainda reutiliza deteccao, alinhamento, mascara e
composicao facial derivados do LatentSync; barba, dentes, perfil e movimentos
rapidos devem ser validados com um trecho curto antes da producao completa.

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
`/var/log/portal/comfyui.log`. Abra um Terminal no Jupyter da instancia Vast e
use os comandos abaixo.

Acompanhar o boot e os logs novos em tempo real:

```bash
tail -n 200 -f /var/log/portal/comfyui.log
```

Ver apenas as ultimas 300 linhas, sem manter o comando aberto:

```bash
tail -n 300 /var/log/portal/comfyui.log
```

Filtrar erros de inicializacao ou execucao:

```bash
grep -Ei 'error|exception|traceback|failed' /var/log/portal/comfyui.log | tail -n 100
```

Confirmar que o processo do ComfyUI foi iniciado:

```bash
pgrep -af '/opt/ComfyUI/main.py'
```

Confirmar que a API esta respondendo na porta interna 8188:

```bash
curl -fsS http://127.0.0.1:8188/system_stats | python3 -m json.tool
```

Durante o primeiro boot com download de modelos, o processo `main.py` ainda
pode nao aparecer. Nesse caso, o `tail -f` acima mostra qual arquivo esta sendo
baixado; o healthcheck passa somente depois que o ComfyUI termina de iniciar.

Com `DOWNLOAD_MODELS_ON_START=1`, o mesmo boot baixa tanto os modelos do
InfiniteTalk quanto os do LatentSync 1.6. Todos ficam no volume montado em
`/opt/ComfyUI/models`, portanto reiniciar ou atualizar a imagem nao repete os
downloads concluidos.

Os pesos do Lip Forcing nao fazem parte desse download padrao. Para inclui-los
no boot, defina tambem `DOWNLOAD_LIPFORCING_MODELS_ON_START=1`; o downloader e
retomavel e valida todos os arquivos antes de liberar a inferencia.

Variaveis principais:

- `MODEL_QUANTIZATION=q8`, `q6_k` ou `q4_k_m`;
- `INFINITETALK_AUDIO_SCALE=1.5` controla a forca do condicionamento de audio em workflows gerados a partir do exemplo oficial;
- `INFINITETALK_AUDIO_CFG_SCALE=1.2` controla o guidance especifico de audio em workflows gerados a partir do exemplo oficial;
- `INFINITETALK_OUTPUT_CRF=16` controla a compressao do MP4 (menor e melhor);
- `HF_TOKEN`, caso algum arquivo exija autenticacao;
- `COMFYUI_ARGS=--preview-method auto`;
- `DOWNLOAD_MODELS_ON_START=0` por padrao;
- `DOWNLOAD_LIPFORCING_MODELS_ON_START=0` por padrao;
- `LIPFORCING_ALLOW_LOW_VRAM=0` mantem o bloqueio de seguranca abaixo de 44 GiB.

O LatentSync e executado depois do modelo Wan e pode elevar o pico de VRAM e o
tempo total. Para o workflow combinado, 48 GB de VRAM e a opcao recomendada;
em 24 GB, valide primeiro com 81 frames e mantenha o block swap do preset.

O preset Q8 se beneficia de mais VRAM. Q4 ainda e adequado para validar o fluxo
em uma GPU de 24 GB usando block swap, desde que o workflow tambem seja trocado
para os arquivos Q4. Reserve pelo menos 100 GB de disco para imagem, cache,
modelos, entradas e saidas. Para manter tambem os checkpoints do LatentSync e
folga para arquivos temporarios, prefira 120 GB ou mais no template Vast.ai.
Ao manter tambem o Lip Forcing 14B, use pelo menos 160 GB; durante o primeiro
download e a criacao temporaria do embedding T5, 180 GB oferece uma margem mais
segura.
