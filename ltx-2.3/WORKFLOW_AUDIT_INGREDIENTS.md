# Auditoria: LTX 2.3 IC-LoRA Ingredients

Data da auditoria: 2026-08-27.

## Fontes comparadas

- Post e comentarios: <https://www.reddit.com/r/StableDiffusion/comments/1ua0wo3/testing_ltx23_iclora_ingredients_using_wan2gpwangp/>
- Branch usada no post: <https://github.com/wing5822/Wan2GP/tree/feat/icLora-ingredients>
- Commit da branch: <https://github.com/wing5822/Wan2GP/commit/b8221c5e779a7af4943013301108c6c99b6945b4>
- ComfyUI-LTXVideo fixado pela imagem: commit
  `15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d`.
- Workflow de referencia: `example_workflows/2.3/LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json`,
  SHA256 `a9d75c1c8546d5e0125d16ac071abd473b45bbfdcfbec87b1cf7abc17dd90d32`.

## O que a branch do post realmente alterou

O commit tem apenas duas mudancas funcionais:

1. adiciona no WanGP a opcao `Set Reference Image for IC-LoRA (Ingredients)`,
   com o codigo interno `I`;
2. troca o fallback de `ltx2_ic_lora_ref_video_frames` de `1` para
   `frame_num`.

A segunda mudanca e a decisiva: a sheet deixa de ser codificada como apenas um
frame de referencia e passa a ser duplicada por toda a duracao. Isso equivale
ao par `RepeatImageBatch` + `LTXAddVideoICLoRAGuide` do workflow ComfyUI.

No teste publicado, foi usado LTX 2.3 Distilled 22B, Ingredients strength 1,
reference sheet como controle, start image I2V opcional e videos de 15
segundos. O autor informa tambem que recortou alguns artefatos iniciais na
edicao; portanto o video publicado nao e uma saida bruta perfeita do pipeline.

## Diferencas para o workflow anterior do repositorio

| Aspecto | WanGP do post | IA2V Ingredients corrigido (schema 3) | Oficial Ingredients |
| --- | --- | --- | --- |
| Duracao de teste | 15 s | 5 s | 10 s / 241 frames |
| Resolucao base | configurada no WanGP | 960x544 no schema 3 | lado curto 544, aproximadamente 960x544 com a sheet oficial |
| Start image | opcional | obrigatoria | nao usa |
| Audio | gerado pelo prompt/modelo | arquivo de audio dirige a fala e e muxado no MP4 | gerado pelo prompt/modelo |
| Reference sheet | repetida por todos os frames | repetida por todos os frames | repetida por todos os frames |
| Amostragem | pipeline WanGP | duas etapas com upscale x2 | single-stage |
| Prompt | formato oficial detalhado | `### Reference Sheet Description` / `### Target Description` | `### Reference Sheet Description` / `### Target Description` |

## Resultados da auditoria do grafo anterior

### Corretos

- A falha de apenas um frame encontrada no WanGP antigo **nao existe** no
  workflow do repositorio. O `RepeatImageBatch` recebe a contagem total de
  frames.
- O guia Ingredients recebe o latent visual antes da concatenacao com o latent
  de audio.
- `LTXVCropGuides` remove os tokens/frames de guia antes do spatial upscale e
  da decodificacao.
- A segunda etapa recebe o condicionamento ja limpo pelo `LTXVCropGuides`; o
  Ingredients nao fica aplicado por engano ao modelo da segunda etapa.
- O MP4 recebe o trecho original do `Driving Audio`, nao o audio reconstruido
  pelo VAE. Isso e coerente com a finalidade IA2V.
- IDs de links, slots de entrada/saida, modelo Ingredients e strength 1 passam
  na validacao estrutural do gerador.

### Conferencia com a avaliacao feita na sessao anterior

A sessao `Avaliação do workflow ConfiuAI` analisou um JSON anexado diferente do
workflow gerado por este repositorio. Os problemas encontrados naquele anexo
nao devem ser transportados automaticamente para esta implementacao:

- aqui o audio externo nao esta bypassado e nao existe `Any Switch` priorizando
  um empty audio latent;
- aqui nao existe o Licon I2V LoRA 0.7 ativo com titulo “BYPASSED”;
- aqui nao existe uma terceira ramificacao de render/upscale em estado hibrido;
- aqui a quantidade do `RepeatImageBatch` vem do calculo de frames, nao fica
  hardcoded em 121.

O “metade da resolucao” aparecia no schema 2 por dois motivos distintos. O
primeiro e o pipeline two-stage: ele gerava a primeira etapa em 768x448 e fazia
upscale latent x2 para aproximadamente 1536x896; trabalhar em meia resolucao na
primeira etapa e intencional. O segundo e o `reference_downscale_factor` do
IC-LoRA, aplicado internamente pelo node. Esse fator tambem e intencional. O
problema de qualidade era pre-redimensionar a sheet para 448 antes dessas
reducoes, enquanto o workflow oficial parte de lado curto 544. O schema 3 do
IA2V agora usa 960x544 na primeira etapa e lado curto 544 para a sheet.

### Problemas e riscos encontrados

1. **Pre-resize abaixo da referencia oficial no schema 2.** O fluxo anterior reduzia a sheet
   para lado curto 448; o oficial usa 544. Em seguida,
   `LTXAddVideoICLoRAGuide` redimensiona novamente conforme
   `reference_downscale_factor` lido dos metadados do LoRA. Se o fator for 2,
   essa segunda etapa reduz a grade de referencia pela metade; essa reducao
   interna e intencional, mas pre-reduzir para 448 descarta detalhe antes dela.
2. **Preset menor e nao canonico no schema 2.** 768x448 e valido e divisivel por 32, mas
   fornece menos detalhe facial que o bucket oficial proximo de 960x544.
3. **Formato de prompt diferente do treino/exemplo.** Os titulos livres do
   fluxo anterior nao reproduzem os cabecalhos usados no exemplo oficial. Isso
   nao quebra o grafo, mas pode piorar a associacao entre paineis e cena.
4. **Possivel distorcao de aspecto.** O guia usa `crop=disabled`; quando a sheet
   e o video tem proporcoes diferentes, a imagem e esticada para o alvo. A
   documentacao anterior recomenda a mesma proporcao, mas o grafo nao impede o
   erro.
5. **Contagem temporal permissiva.** O calculo `duration * fps + 1` produz
   `8n+1` nos defaults de segundos inteiros a 24 FPS, mas pode gerar contagens
   invalidas com duracao fracionaria ou outro FPS.
6. **Instrucao visual imprecisa.** A documentacao exigia “fundo preto”. A sheet
   oficial 1088x608 usa fundos claros nos paineis e apenas divisorias pretas.
7. **Combinacao experimental.** O IA2V com audio externo + Ingredients + duas
   etapas nao e o workflow Ingredients oficial. O grafo e consistente, mas a
   qualidade final precisa ser validada em GPU; os testes de build nao medem
   identidade, lip-sync, ghosting ou OOM.
8. **Pesos nao identicos ao exemplo BF16.** A imagem existente baixa DEV FP8,
   Gemma FP4 e o Distilled LoRA dinamico. Os novos presets usam esses arquivos
   para nao acrescentar dezenas de GiB. O grafo e comparavel ao oficial, mas o
   resultado nao e uma comparacao bit a bit com os pesos do JSON original nem
   com o transformer Distilled usado pelo WanGP.

## Implementacao adicionada

Foram adicionados dois workflows comparativos, o IA2V principal foi corrigido
e o comportamento anterior foi preservado em um quarto arquivo Ingredients:

- `video_ltx2_3_ingredients_official_single_stage-docker.json`: grafo oficial
  single-stage, adaptado ao checkpoint DEV FP8, Gemma FP4 e Distilled LoRA
  dinamico ja baixados pela imagem;
- `video_ltx2_3_ingredients_wangp_i2v_15s-docker.json`: o mesmo grafo, com
  `LTXVImgToVideoConditionOnly`, start image ativa, sheet repetida nos 361
  frames e 24 FPS;
- `video_ltx2_3_ia2v_ingredients_legacy_v2-docker.json`: copia regeneravel do
  schema 2 anterior, com 768x448, pre-resize 448 e prompt antigo.

O `video_ltx2_3_ia2v_ingredients-docker.json` permanece com o mesmo nome, mas
passa ao schema 3: 960x544, pre-resize 544 e prompt com as secoes oficiais. Ao
encontrar o schema 2 no volume, o seeding cria
`video_ltx2_3_ia2v_ingredients-docker.schema-v2.backup.json` antes de instalar
a correcao. Os dois workflows novos
continuam sendo instalados em nomes separados.

## Ordem de teste recomendada

1. Rode o preset oficial de 241 frames com a sheet oficial ou uma sheet na
   mesma proporcao. Ele e o controle da comparacao.
2. Rode a variante I2V com 241 frames primeiro, embora ela abra em 361; depois
   suba para 361 para comparar com o post.
3. Rode o IA2V anterior com a mesma sheet, prompt e seed quando for possivel;
   a diferenca inevitavel sera o audio externo e a segunda etapa.
4. Compare identidade facial, roupa/props, geometria relativa dos objetos,
   artefatos nos primeiros frames, sincronismo labial e pico de VRAM.
