# Plataforma: Tiny Map da sua cidade, mesmo sem prédios 3D do Google

> Gostou dos Tiny Maps de Londres, Nova York e Rio? Eles só existem onde o Google tem a cidade em 3D.
> Aqui você faz o da sua cidade, refinando junto com a IA no estúdio, e chega ainda mais perto da
> perfeição se tiver um drone.

## Posicionamento

Os mapas virais de 2026 (tiny.london, isometric.nyc, Pixel Singapore, Tiny Rio) partem do Google
Photorealistic 3D Tiles. A maior parte das cidades do mundo não tem essa malha (a Tríplice Fronteira
é o nosso caso de teste). Sem ela, a IA inventa lagos, cidades e matas, e os mapas ficam ruins.

O diferencial é o **estúdio com humano no meio da IA (human in the loop)**: a máquina faz o grosso,
correções determinísticas limpam o que dá para limpar sem inventar nada, e a pessoa decide e retoca
o resto com ferramentas feitas para isso. Drone e GPU são camadas que melhoram o resultado, não
pré-requisitos.

## Camadas (modulares)

| Camada | Precisa de | O que faz | Estado |
|---|---|---|---|
| 0. Maquete | nada (OSM + Overture + satélite) | cidade 3D aproximada, na câmera do mapa | pronto |
| 1. Estúdio | navegador | oficina: saúde do mapa, consertar e polir sem IA, caçador de defeitos, ferramentas de pixel art, histórico | pronto |
| 2. IA externa | conta no Gemini (ou outro) | exportar recorte + prompt, importar alinhado e na paleta | pronto |
| 3. GPU local | 1 RTX 3090 | geração do mapa (Qwen-Image-Edit + LoRA), laboratório de LoRA com nota objetiva | código pronto, falta rodar |
| 4. Drone | drone + piloto | malha 3D real nas áreas importantes; gabarito para medir a IA | planejado |

Cada cidade sobe de camada quando quiser: começa na 0+1 de graça, vai para 3 e 4 onde vale a pena.

## O que funciona sem GPU (testado em `tests/`)

- `tinymap demo`: cidade sintética com os defeitos da IA plantados e gabarito. Abre o estúdio sem
  dados nem chave: é o onboarding e a bancada de testes.
- `tinymap autofix --lakes`: água inventada e terra no rio repintadas a partir da maquete, com a
  tradução de cor que a própria arte usou em volta (ruas e prédios reaparecem no lugar certo).
- `tinymap polish`: grade de pixel (detecta tamanho e deslocamento), paleta do mapa (protege cores
  raras), emendas entre blocos (ajuste de cor por bloco medido contra a maquete).
- Estúdio: aba **Mapa** com nota de 0 a 100, “Consertar e polir o mapa inteiro”, polimento da área
  com prévia antes/depois; caçador acha água inventada, terra no rio, emendas e shots borrados.
- Na cidade demo, um clique leva a nota de 49 a 96. Desfazer tudo devolve os pixels exatos.

Testes: `python tests/test_estudio.py` (4 testes, ~1,5 min) e `python tests/e2e_oficina.py`
(navegador de verdade). Laboratório e geração: `tests/test_lab.py`, `tests/test_joint_helpers.py`.

## Segunda-feira (3090 e drone chegando)

1. **GPU, primeira hora** (piloto da Tríplice Fronteira):
   - `pip install -e ".[lab]"`, `tinymap try` sem LoRA num bloco do Marco: valida o `joint()` com
     pulo de janelas, faixa livre e checkpoint por passo. Se quebrar, é nos internos do diffusers.
   - `tinymap lab-freeze --pairs pairs_v1`, `lab-eval --name base` duas vezes (piso de ruído).
2. **GPU, primeira noite**: `lab-train r1` (rank 32, 3.000 passos) e `lab-eval r1 --lora .../out`
   (todos os checkpoints). O melhor vira a LoRA da geração.
3. **Geração**: `scripts/gpus.sh` no piloto; depois `tinymap autofix --lakes` e `tinymap polish`
   (ou o botão do estúdio) e a revisão humana com a nota do mapa.
4. **Drone, primeiro voo**: só depois de conferir o espaço aéreo no SARPAS. Uma área pequena e
   importante (microcentro de CDE ou cabeceira da ponte), voo vertical + inclinado na direção da
   câmera do mapa. Processar no WebODM e comparar com a maquete no estúdio (camada Satélite/Cortina).

## Roteiro de 4 semanas até o lançamento

| Semana | Produto | Qualidade | Lançamento |
|---|---|---|---|
| 1 | LoRA r1/r2 no piloto, geração do piloto, estúdio no mapa real | nota do mapa ≥ 85 no piloto; lab sem cópia | landing com o piloto |
| 2 | Tríplice Fronteira inteira; `tinymap fly plan` (missão pela câmera do mapa) | emendas e borrado zerados; lagoas revisadas | vídeo antes/depois do estúdio |
| 3 | primeira área de drone na malha; gabarito de drone no laboratório | medir quanto a maquete inventa vs o drone | convite a 3-5 cidades pequenas |
| 4 | “nova cidade” guiada: região → maquete → estúdio → publicar | checklist de publicação pela nota | lançamento mundial (inglês, espanhol, português) |

## Próximos passos de código (sem GPU, em ordem)

1. **Assistente de nova cidade**: desenhar a área num mapa, gerar `regions/<cidade>.yaml`, baixar
   OSM/Overture/satélite e abrir o estúdio.
2. **Defeito “terra na ponte / rua”**: mesma tradução maquete→arte usada no rio, aplicada onde a
   maquete tem rua e a arte tem verde ou água.
3. **Contorno**: reforçar o contorno escuro de 1 px nos prédios a partir das bordas da maquete.
4. **Nota por bloco no estúdio** (mapa de calor), para dividir a revisão entre várias pessoas.
5. **Revisão colaborativa**: várias pessoas no mesmo mapa, com histórico por autor.
6. `tinymap fly plan` (precisa do modelo do drone).
