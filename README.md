# tinymap

Gerador de mapas isométricos em pixel art, no estilo tiny.london / isometric.nyc, rodando nas
**suas** GPUs (feito para RTX 3090 de 24 GB). Você descreve a área num YAML e o pipeline faz o resto:
render do Google 3D → pares de treino → LoRA → cidade inteira sem emendas → viewer com busca.

Primeira região: **Tríplice Fronteira** (Foz do Iguaçu, Ciudad del Este, Puerto Iguazú),
`regions/triplice-fronteira.yaml`. Canvas planejado: 45.056 × 43.008 px, 315 shots, ~1,3 gigapixel
de área útil.

Como os outros mapas foram feitos e de onde vêm as decisões: [docs/PESQUISA.md](docs/PESQUISA.md).

## Pipeline

```
YAML da região
  └─ plan ──── câmera ortográfica única (45° / 15°), canvas e lista de shots
  └─ render ── Google Photorealistic 3D Tiles → shots 2048 px que alinham no pixel
  └─ osm ───── rios (máscara + dica no prompt + checagem), lugares (busca), fronteiras
  └─ teacher ─ recortes 1024 px redesenhados + checagens automáticas → pares
  └─ train ─── LoRA no Qwen-Image-Edit (ai-toolkit, local)
  └─ generate  amostragem conjunta por blocos, 1 worker por GPU, retomável
  └─ site ──── pirâmide Deep Zoom WebP + OpenSeadragon + busca + fronteiras
```

## Instalação

```bash
git clone <este repo> tinymap && cd tinymap
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # torch com CUDA, diffusers, bitsandbytes, pyvips…
sudo apt install libvips nodejs        # pyvips precisa da libvips; Node 20+
cd render && npm install && npx playwright install chromium && cd ..
cp .env.example .env                   # GOOGLE_TILES_API_KEY ou CESIUM_ION_TOKEN
git clone https://github.com/ostris/ai-toolkit ../ai-toolkit   # para treinar a LoRA
```

O Chromium headless renderiza com a GPU (`--use-angle=vulkan`). Se a máquina não tiver display,
rode com `xvfb-run -a tinymap render …`.

## Passo a passo para a Tríplice Fronteira

**1. Piloto pequeno (≈1,5 × 1,7 km no Marco das Três Fronteiras).** Valida a câmera e a cobertura 3D.

```bash
tinymap plan   regions/triplice-fronteira.yaml --pilot
tinymap probe  regions/triplice-fronteira.yaml --pilot   # 1 km em cada landmark
tinymap render regions/triplice-fronteira.yaml --pilot --workers 2
tinymap stitch regions/triplice-fronteira.yaml --pilot   # runs/…-pilot/render_preview.jpg
```

Olhe os probes: se Ciudad del Este ou Puerto Iguazú aparecerem com prédios "derretidos" ou chapados,
a malha 3D do Google ali é fraca. Nesse caso o mapa ainda funciona (o modelo redesenha), mas vale
diminuir a área nesse lado ou aceitar um estilo mais livre ali.

**2. Pares de treino.** O jeito mais barato que ainda fica bom:

```bash
tinymap osm     regions/triplice-fronteira.yaml --pilot
# rodada 1: ~120 pares com o melhor professor que você puder (Gemini ≈ US$15),
# ou 100% local com o Qwen base / uma LoRA pública (pior, mas grátis)
tinymap teacher regions/triplice-fronteira.yaml --pilot --backend gemini --out pairs_v1 --n 120
tinymap sheets  regions/triplice-fronteira.yaml --pilot --out pairs_v1   # apague os *_art.png ruins
```

Para ficar 100% local na rodada 1: `--backend qwen` (opcionalmente `--lora` com a LoRA pública do
isometric-nyc, baixada do Oxen conforme `inference/README.md` daquele repo).

**3. LoRA v1 local** (3090, ~2-3 h para 300 pares / 3.000 passos):

```bash
tinymap train regions/triplice-fronteira.yaml --pilot --pairs pairs_v1 --name v1 --steps 3000
cd ../ai-toolkit && python run.py ../tinymap/runs/triplice-fronteira-pilot/lora/v1/train.yaml
```

**4. Rodada 2 com a própria LoRA** (local, grátis): render da cidade toda, ~800 pares, filtra, treina v2.

```bash
tinymap plan    regions/triplice-fronteira.yaml
tinymap render  regions/triplice-fronteira.yaml --workers 3
tinymap osm     regions/triplice-fronteira.yaml
tinymap teacher regions/triplice-fronteira.yaml --backend qwen --lora <v1.safetensors> --out pairs_v2 --n 800
tinymap sheets  regions/triplice-fronteira.yaml --out pairs_v2
tinymap train   regions/triplice-fronteira.yaml --pairs pairs_v2 --name v2 --steps 6000
```

**5. Teste um bloco antes de gastar a noite** e compare LoRAs:

```bash
tinymap try regions/triplice-fronteira.yaml --lora <v2.safetensors> --at -25.5925,-54.5913
```

**6. Cidade inteira em todas as GPUs**, retomável:

```bash
scripts/gpus.sh regions/triplice-fronteira.yaml <v2.safetensors>
tinymap stitch regions/triplice-fronteira.yaml --art       # prévia do que já foi pintado
tinymap unlock regions/triplice-fronteira.yaml             # se um worker morreu no meio
```

**7. Site:**

```bash
tinymap site regions/triplice-fronteira.yaml
python -m http.server -d site 8080
```

## Quanto custa / quanto demora

Estimativa a partir dos números do tiny.london (31 H100-h para 14 Gpx a 8 passos):

| Etapa | Tríplice Fronteira |
|---|---|
| Render (315 shots) | ~1-2 h, uso da Map Tiles API dentro da cota grátis mensal na maioria das contas |
| Professor rodada 1 (Gemini, 120 imagens) | ≈ US$15, ou grátis com Qwen local |
| LoRA v1 + v2 (3090) | ~2-3 h + ~5-6 h |
| Geração, 20 passos | ~30-35 h numa 3090; divide pelo número de GPUs |
| Geração, 8 passos (LoRA Lightning/Turbo) | ~12-15 h numa 3090 |

Para encurtar: `meters_per_px: 0.4` reduz a área em pixels pela metade; `steps: 8` com uma LoRA de
poucos passos (Qwen-Image-Edit Lightning) corta outros 60%.

## Detalhes técnicos

- **Câmera** (`render/camera.mjs`): uma só para render, planejamento, água, lugares e busca. Um ponto
  lat/lon cai sempre no mesmo pixel em todas as etapas.
- **Modelo** (`tinymap/models.py`): Qwen-Image-Edit-2511 em NF4 (transformer + text encoder),
  cabe em 24 GB. `--model` aceita qualquer checkpoint do `QwenImageEditPlusPipeline`.
- **Amostragem conjunta** (`QwenEdit.joint`): um latente por bloco; a cada passo, janelas de 1024 px
  com passo 768, grade alternada deslocada 384 px, previsões somadas com peso em tenda; contexto
  dos vizinhos fixado por pinning (RePaint). Decodificação também em janelas sobrepostas.
- **Multi-GPU** (`tinymap/generate.py`): blocos de 6.144 px + margem de 1.024 px, quatro fases por
  paridade, lock files. Pare e retome quando quiser.
- **Checagens** (`tinymap/checks.py`): cópia de foto (flat/soft share), água inventada fora do OSM,
  buracos brancos, deriva de geometria (correlação de bordas com o render).

## Ainda não validado em GPU

Este repositório foi escrito e testado sem GPU nem chaves: `plan` e a projeção foram executados e
conferidos (os landmarks caem no lugar certo). O resto precisa da primeira rodada na sua máquina,
e é onde espero ajustes:

- O loop de `joint()` chama o transformer do Qwen direto, com os internos do diffusers 0.42. Se uma
  versão mudar assinaturas, é ali que quebra. Rode `tinymap try` no piloto primeiro.
- O bloco `model:` do `train.yaml` pode precisar do formato exato do exemplo 24 GB do ai-toolkit
  da sua versão (comentário em `tinymap/train.py`).
- `target_height_m` e `water_height_m` são estimativas para Foz; se a máscara de rio sair deslocada
  do rio no render, ajuste `water_height_m`.

## Créditos e licença

Método de Andy Coenen (isometric.nyc, MIT) e Boris Starkov (tiny.london). Imagens derivadas de
Google Photorealistic 3D Tiles exigem a atribuição do Google; dados © OpenStreetMap contributors.
Código deste repo: MIT.
