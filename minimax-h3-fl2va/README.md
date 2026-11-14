# MiniMax H3 FL2VA — ComfyUI independente

Imagem Docker do ComfyUI com grafo de primeiro/último frame e áudio de guia. **Estado: desenvolvimento; build da imagem, importação do workflow, carregamento dos modelos e geração real ainda não validados.** Não há confirmação de funcionamento em 16 GB nem avaliação de identidade ou lip-sync. O serviço é independente e usa volumes próprios. ComfyUI e ComfyUI-GGUF têm commits fixados no Dockerfile; os pesos são baixados separadamente.

O padrão solicitado é **Turbo FL2VA de 4 passos**, LoRA `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`, Euler, scheduler simple, shift vídeo 6 / áudio 3, força 1. O autor informa treinamento a 768p; a resolução de teste 576×1024 é experimental. Há também um grafo de qualidade sem LoRA, com `res_multistep` e 20 passos. As referências oficiais são [LightX2V](https://github.com/ModelTC/Minimax-H3-Turbo), [nós H3 do ComfyUI](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py) e [guia de execução local](https://github.com/MiniMax-AI/awesome-minimax-h3-integration/blob/main/docs/run-h3-locally.md). A LoRA é FL2VA, não a Ref2VA.

## Publicação e início

O workflow `.github/workflows/publish-minimax-h3-fl2va.yml` está preparado para testar e publicar no GHCR em push para `main` com mudanças nesta pasta. A imagem deve ser reconstruída e publicada novamente para incorporar alterações. Após o push e o sucesso do GitHub Actions, no Ubuntu com NVIDIA Container Toolkit:

```bash
cd minimax-h3-fl2va
docker compose pull
docker compose up -d
docker compose logs -f
```

Imagem prevista: `ghcr.io/glauber-fullstackdev/ambienteavatar-minimax-h3-fl2va:latest`. Em outro proprietário de repositório, ajuste `H3_IMAGE`. Para build local, use `docker compose up --build -d`. Interface: `http://127.0.0.1:8190` no servidor. Para acesso remoto seguro, use túnel SSH: `ssh -L 8190:127.0.0.1:8190 glauber@192.168.1.25`. A interface não possui autenticação própria; mantenha o bind local ou proteja-a externamente.

Os volumes nomeados `h3-models`, `h3-input`, `h3-output`, `h3-user`, `h3-cache` são próprios do projeto Compose; a instalação existente não é alterada. A GPU é compartilhada: o Compose não impede concorrência com outros serviços. Execute somente uma geração por vez e confira VRAM disponível antes de iniciar.

## Modelos

Confira espaço livre **no armazenamento do Docker** antes de baixar. O único modelo principal baixado é `minimax_h3_fl2va_pruned-Q4_K.gguf` (~10,64 GiB). O download também inclui componentes necessários: encoder `qwen3vl_32b_minimax_h3-Q2_K_M.gguf` (~12,20 GiB), VAEs de vídeo e áudio oficiais e a LoRA Turbo FL2VA de 4 passos. É necessário espaço adicional para cache e arquivos temporários. O script usa `hf_hub_download` com cache persistente e retomada. Confira as licenças dos repositórios [Unsloth](https://huggingface.co/unsloth/MiniMax-H3-GGUF), [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3) e [LightX2V](https://huggingface.co/lightx2v/Minimax-h3-Turbo); forneça `HF_TOKEN` se exigido.

```bash
docker compose exec comfyui python /opt/pipeline/baixar_modelos.py
# Depois do download, reinicie para atualizar os menus de arquivos:
docker compose restart
```

O Q4_K e o encoder Q2_K_M usam ComfyUI-GGUF. A combinação com o H3, a LoRA e os 16 GB ainda não foi validada: o tamanho do arquivo não é o pico de VRAM e o offload continua necessário. O download pode ser executado com `--sem-turbo` caso se queira apenas o grafo de qualidade. Os pesos seguem a revisão `main` dos repositórios e devem ser fixados após validação. Não há download automático do modelo principal INT8.

## Executar via API local do ComfyUI

O executor roda dentro do container e envia o grafo para `/prompt`, acompanha `/history` e obtém o MP4 por `/view`. As imagens e o áudio são copiados para o diretório persistente de entrada; não presume upload de áudio pelo endpoint de imagem. São necessários arquivos reais no host:

```bash
docker compose cp primeiro.png comfyui:/tmp/primeiro.png
docker compose cp ultimo.png comfyui:/tmp/ultimo.png
docker compose cp voz.m4a comfyui:/tmp/voz.m4a
docker compose exec comfyui python /opt/pipeline/executar.py \
  --first-frame /tmp/primeiro.png --last-frame /tmp/ultimo.png \
  --audio /tmp/voz.m4a --prompt 'A pessoa fala olhando para a câmera, movimento natural.' \
  --seed 42 --output-path /opt/ComfyUI/output/resultado.mp4
docker compose cp comfyui:/opt/ComfyUI/output/resultado.mp4 ./resultado.mp4
docker compose cp comfyui:/opt/ComfyUI/output/resultado_gerado.mp4 ./resultado_gerado.mp4
docker compose cp comfyui:/opt/ComfyUI/output/resultado_audio_gerado.wav ./resultado_audio_gerado.wav
```

Acrescente `--quality` para o grafo sem LoRA; o Q4_K GGUF é usado por padrão. Opções `--width`, `--height`, `--steps`, `--seed` e `--timeout` são configuráveis; Turbo exige quatro passos. O padrão é 576×1024, 124 frames, 24 fps (~5,17 s). Áudio maior que essa duração é recusado para não cortar a fala; áudios menores recebem silêncio no final. O áudio de guia é convertido para WAV 32 kHz. O MP4 final usa a gravação original codificada em AAC, sem alterar velocidade ou tonalidade; o arquivo de diagnóstico contém o áudio gerado. O WAV de diagnóstico é extraído desse MP4 e, portanto, reflete a codificação do vídeo de diagnóstico. Isso não garante sincronia labial.

`h3_first_last_audio_ui.json` é um rascunho importável na interface para Turbo; escolha imagens e áudio em seus nós e confira conexões e modelos. A importação precisa ser verificada na versão real. Os grafos API são `h3_first_last_audio_turbo4_api.json` e `h3_first_last_audio_api.json` (qualidade). Não há suporte implementado a ajuste para exatamente 5,0 s. O primeiro frame é esticado à tela e o último sofre recorte central na implementação H3: use imagens na proporção de saída.

## Diagnóstico e pendências

`python3 diagnostico.py --comfy-url http://127.0.0.1:8190 --models /caminho/modelos > diagnostico.json` reúne informações somente de leitura. Os modelos em volumes Docker podem não estar visíveis nesse caminho no host. Revise o relatório antes de compartilhá-lo.

Faltam build e teste em GPU, verificação de `/object_info` e da importação, validação da quantização, geração de exemplo com voz em português, inspeção visual do primeiro e último frames, sincronia labial e medições de pico de VRAM/RAM e tempo. O resultado do teste estrutural do grafo não comprova essas propriedades.
