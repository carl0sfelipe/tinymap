# Como os Tiny Maps foram feitos (pesquisa, nov/2026)

Fontes: os 15 repositórios públicos clonados, o write-up `tiny.london/how`, o diretório
`isocities.agenturl.dev` (60 mapas, 39 cidades) e as páginas "how" de cada mapa.

## Os três jeitos de fazer

### 1. IA redesenha um render do Google 3D (o "estilo tiny.london") — o que este repo faz

Pipeline comum a quase todos os mapas virais:

1. **Render**: uma câmera ortográfica única (inclinada 45°, girada 15°) sobre o
   Google Photorealistic 3D Tiles, via three.js + `3d-tiles-renderer` (ou CesiumJS) dirigido por
   Playwright. Cada "shot" é um sub-retângulo do frustum, então todos alinham no pixel.
   Escala típica 0,293 m/px.
2. **Professor (teacher)**: um modelo grande redesenha ~100 a 2.500 recortes de 1024 px em pixel art
   (Gemini 3 Pro Image / Nano Banana Pro, GPT Image 2.5). Checagens automáticas rejeitam cópias de
   foto, água inventada, áreas em branco.
3. **Aluno (student)**: LoRA num modelo aberto de edição de imagem, treinado nos pares
   render → pixel art. É ele que pinta a cidade inteira, barato e consistente.
4. **Montagem sem emenda**: tile a tile com infill (NYC) ou amostragem conjunta MultiDiffusion
   sobre um latente único (Londres).
5. **Viewer**: pirâmide Deep Zoom em WebP + OpenSeadragon.

| Mapa | Professor | Aluno | Base | Custo informado |
|---|---|---|---|---|
| Isometric NYC (código MIT) | Nano Banana Pro | Qwen-Image-Edit + LoRA "omni infill" (Oxen) | Google 3D | ~US$600 GPU; fine-tune ~US$12 |
| tiny.london | Gemini 3 Pro Image (7.277 chamadas) | Qwen-Image-2.1-Turbo int8, LoRA rank 32, 8.000 passos | Google 3D | 31 H100-h na rodada final, ~70 no total |
| Isometric Toronto | Gemini 3 Pro Image | Qwen-Image-2.1-Turbo LoRA | Google 3D | US$91,97 itemizado |
| Tiny Stockholm | gpt-image-2.5 no centro | Qwen-Image 2.1 com a LoRA do tiny.london | Google 3D | — |
| isopolis (SF) | Nano Banana | Qwen-Image-Edit-2511 LoRA | Google 3D | ~US$4 imagens + US$10-15 por treino |
| Tiny Prague (.me) | — | Qwen Image, 8 pares, 400 passos | Google 3D | — |
| Pixel Berlin | — | Qwen-Image-Edit LoRA | Google 3D | ~US$100 |
| Isometric Istanbul | GPT Image 2.5 | Qwen-Image-2.1-Turbo | modelo 3D da prefeitura | — |
| brasilia.isocity | — | Qwen-Image-Edit local numa RTX 5090 | Google 3D | — |
| HypeSeoul | — | FLUX.2 klein 4B LoRA (33 pares) | 3D próprio | — |
| Isometric SF (Bay Area) | — | Klein 4B img2img no Modal | — | — |
| Tiny Utrecht (código) | Gemini 3 Pro Image direto em 84 tiles, sem aluno | — | Google 3D | — |
| SimLeón (código) | Nano Banana Pro + juiz Gemini Flash + SSIM | — | Google 3D | US$5,75 no piloto |
| blr_mini (código) | só a etapa de render por enquanto | — | Google 3D ou Cesium ion | — |

Conclusão: **Qwen (Image-Edit ou Image-2.1) + LoRA treinada com pares de um professor grande** é o
padrão. A brasilia.isocity prova que roda local numa GPU de consumo.

### 2. Cena 3D no navegador com shader de pixel art (sem IA)

Dados abertos de prédios (LoD2, PLATEAU, OSM) renderizados ao vivo em three.js, às vezes com shader
de dithering. Gira, tem dia/noite, transporte ao vivo. Visual menos "desenhado à mão".
Código aberto: `isometric-berlin` (MIT), `berlin/london-field-atlas`, `010` Roterdã (MIT),
`iso-tash` Tashkent (MIT), `isometric-tokyo` (MIT).

### 3. Desenhado em código a partir do OSM (sem IA)

Cada prédio do OpenStreetMap extrudado e pintado em canvas/WebGL com cores de `building:colour`.
Código aberto: `isometric-london` (MIT), `leuven.social` (MIT), `edirne-isometric` (que acabou
trocando por imagem aérea do Esri inclinada).

## Repositórios abertos e o que aproveitar

| Repo | Licença | O que tem de útil |
|---|---|---|
| cannoneyed/isometric-nyc | MIT | Pipeline completo, app de geração, dataset "omni infill", LoRAs públicas no Oxen |
| Janjs/tiny-utrecht | sem licença declarada | Render de 120 linhas fiel ao tiny.london, checagens, máscara de água OSM |
| Rithvik1709/blr_mini | sem licença | Render via Cesium ion (token grátis), pirâmide, busca |
| enrique-am/simleon | sem licença | Detector de erros + correção de uma caixa só |
| JustinChavez/sf-isometric-demo | sem licença | Viewer estático de mosaico |
| Klotzkette/isometric-berlin, joshp123/010, jahongircode/iso-tash, stianfro/isometric-tokyo | MIT | Abordagem 3D |
| arikhanacademic-cpu/isometric-london, basteau/leuven.social | MIT | Abordagem em código |

Este repo não copia código dos repositórios sem licença; reimplementa as ideias publicadas.

## Lições que entraram no tinymap

- O prompt diz **como** desenhar, não **o que** (tiny.london).
- O professor copia a foto em áreas densas; um quarto de 512 px ampliado 2× é redesenhado melhor.
- Água inventada é o erro nº 1: projetar a água do OSM com a mesma câmera, citar no prompt e checar.
- Emendas somem com amostragem conjunta (janelas 1024, passo 768, grade deslocada a cada passo,
  peso em tenda) e com "pinning" das bordas já pintadas.
- Blocos em quatro fases por paridade permitem várias GPUs sem que blocos vizinhos rodem juntos.
- Pirâmide com o nível mais nítido em WebP sem perdas preserva o pixel.

## Licenças a observar

Imagens derivadas do Google Photorealistic 3D Tiles têm atribuição obrigatória e termos de uso do
Google Maps Platform. O Qwen-Image-2.1-Turbo foi publicado com licença de pesquisa/avaliação;
Qwen-Image-Edit-2511 é Apache-2.0. Confira antes de qualquer uso comercial.

## Pixel Singapore (yadongxie.com, nov/2026) — o que veio para cá

Mesma receita do tiny.london (Google 3D via Cesium, câmera 45°/15°, 0,25 m/px, shots de 2048 px,
Qwen-Image-2.1-Turbo + LoRA de estilo, MultiDiffusion com janelas 1024/768 e grade deslocada 384 px,
pinning RePaint), feito numa única RTX 5080 (pico de 14,8 GB): 4 blocos de 16.384 × 10.240 px em 6,6 h.

Trazido para o `joint()`:
- **Janelas já pintadas são puladas**: não codificam condicionamento nem chamam o transformer.
- **Faixa livre de 64 px** ao longo da borda do que já está pintado, para costurar sem parede.
- **Checkpoint do latente a cada passo** (`blocks/<id>.steps.pt`): travou, retoma do passo.
- Viewer: pixel nítido a partir de 1,25x, suavizado abaixo (evita moiré ao afastar).

Para depois (não implementado):
- **Passe de altura** no render (altura em cm nos canais R e G): faixas de trânsito sobre o asfalto
  real, carros escondidos atrás de prédios e árvores, copas que balançam, água animada só onde é
  plana. É o que faria os carrinhos da ponte andarem no tabuleiro desenhado.
- **Trânsito com IDM** (A 1,4, B 2,2, S0 2,2, T 1,3) e sprites de carro renderizados em 32 direções
  com o mesmo sol — combinaria com a contagem real da câmera da ponte.
- **Teste de saturação** para separar grama de água opaca (grama ≥ 0,6, água ≤ 0,4 na paleta deles):
  calibrar nas nossas pinturas antes de mudar o detector de água inventada.
- WebP q60 (189 MB para a cidade toda), tiles vazios não gravados e manifesto versionado no URL.

## Novos no diretório (07/11/2026: 65 mapas)

Entraram Mumbai, Paris (Postcard Roads), Roma, Madrid e Tirana; fora do diretório, Tiny Athens.
Belo Horizonte continua sem mapa.

| Mapa | O que tem de novo | Fonte |
|---|---|---|
| Isometric Mumbai | GPT Image redesenhou ~800 tiles; **LoRA no FLUX.2 [klein] 4B** (Apache-2.0), rank 32, 2.500 passos, lr 1e-4, ~1 h numa B200; **4 passos, ~2,5 s por tile de 1024 px**, guidance 1,0. Pares com **cópias do render de cor alterada** e exemplos de **preenchimento com moldura vermelha de 2 px** (faixas e miolo). | huggingface.co/Flash1405/isometric-mumbai |
| Tiny Tirana | LoRA no Qwen-Image 2.1 treinada no **Google Colab (G4)**; blocos de 8.192 px com pinning; **emendas restantes curadas com faixas de SDEdit**; recorte do mapa no **formato da cidade**; aviões ao vivo (**adsb.lol / adsb.fi** por um proxy no Cloudflare Worker); ônibus pelo GTFS. | github.com/okanerturan/tiny-tirana |
| Tiny Athens | Feito num **MacBook Air**: ~100 pinturas do Gemini, **LoRA em SD 1.5 + ControlNet canny** treinada no próprio Mac; ônibus e trólebus ao vivo (OASA). | github.com/angelopoulosk/tiny-athens |
| Postcard Roads: Paris | Mapa que é jogo: mostra os jogadores dirigindo agora. | postcardroads.com/paris |

Implementado no tinymap (sem GPU, testado em `tests/test_augment.py`):
- `lab-train --color-copies N`: cópias do render com matiz, saturação, brilho e ganho de canal
  alterados, mesma pintura-alvo. A LoRA para de copiar a cor do satélite.
- `lab-train --infill N`: exemplos de preenchimento (arte fora, maquete dentro, moldura vermelha).
  Prepara a geração tile a tile encostada em tiles prontos e o conserto por IA no estúdio.

A fazer (por valor):
1. **Backend FLUX.2 klein 4B** em `models.py` e `train.py` (arch do ai-toolkit para klein): 4 passos
   e 4B parâmetros na 3090 dá ordens de grandeza menos tempo que o Qwen 20B, e é Apache-2.0
   (comercial). Medir no laboratório contra a LoRA Qwen.
2. **Geração por preenchimento** (Mumbai): pintar cada tile novo com a moldura, usando os vizinhos
   prontos como contexto, como alternativa ao joint() para GPUs menores.
3. **Faixas de SDEdit nas emendas** (Tirana): para as emendas de desenho, não só de cor, que o
   polimento não resolve.
4. **Recorte no formato da cidade** (Tirana): máscara do limite municipal do OSM no `site`.
5. **Aviões ao vivo** (adsb.lol/adsb.fi): Foz (IGU), Minga Guazú (AGT) e Pampulha/Confins em BH.
6. **Modo leve** (Athens): SD 1.5 + ControlNet para quem não tem GPU grande; bom para a plataforma.
7. Divulgação: o mapa de mapas do tiny.london aceita PR (`tiny-world`), e há o Isometric Map Finder.

## Varredura de 08/11/2026

Diretório isometriccities.com: 80 mapas, 27 novos em novembro. Brasil novo: Brasília Isocity
(Igor Azevedo e Breno Kern), TinySP (Lucas Guerras), Tiny Orindiúva (dados abertos). BH: nenhum.

| Fonte | Técnica | No tinymap |
|---|---|---|
| Isometric SF (Justin Chavez) | Lightning 4 passos no Qwen-Edit-2511 | `teacher_lightning` |
| Isometric SF | professor com LoRA de preenchimento (45 alvos × 5) | `lab-train --infill 5` |
| Isometric SF | guarda de gasto para rodar a noite | `scripts/budget_guard.py` |
| Isometric SF | aluno FLUX.2 klein 4B, ~4x mais rápido | pendente (backend klein) |
| Isometric Toronto | filtro automático + 3 tentativas por par | já existia (`checks`, `ATTEMPTS`) |
| Isometric Toronto | checkpoint intermediário ganha do final | já existia (lab) |
| Isometric Toronto | tom global calibrado em 12 janelas | `polish --tone` |
| Isometric Toronto | camada pixel art com 256 cores acima de 2x | não precisa: nossa arte já é grade de pixel |
| Tiny Tirana | recorte na forma da cidade; cura de emenda com SDEdit | `polish --shape`; SDEdit pendente |
| Isometric Warsaw | aviso preso ao local | site + `tinymap avisos` |
| tiny.london | LoRA pública (`ktoyame/tiny-london-lora`) | só comparação no lab: licença não comercial |

## Provedores de dados (08/11/2026)

| Dado | Fonte | Uso no tinymap |
|---|---|---|
| Fotos de rua abertas | Mapillary (CC-BY-SA), KartaView | fichas de fachada (já no código) |
| Altura de prédio no Sul Global | Google Open Buildings 2.5D Temporal (Earth Engine, 4 m, 2016-2023) | alturas da maquete na Tríplice Fronteira onde OSM/Overture não têm; pendente |
| Base oficial de BH | BHMap / BHGEO da PBH (CC-BY) | nomes, lotes e equipamentos públicos para pinos e pontos de referência |
| Relevo | Copernicus DEM GLO-30 | altura do chão por ponto de referência em cidade de morro (BH); pendente |
