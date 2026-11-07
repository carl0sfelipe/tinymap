# Tiny BH — Belo Horizonte

## Já existe?

Não encontrado (pesquisa de 07/11/2026): o diretório Isocities (60 mapas, 39 cidades) tem, do
Brasil, Rio, São Paulo, Brasília, Goiânia, João Pessoa e Juiz de Fora (a única de Minas). As buscas
por mapa isométrico ou pixel art de BH não trouxeram nenhum.

## Por que BH é o segundo mapa

- Contraste com a Tríplice Fronteira: lá não há malha 3D do Google e o estúdio faz o trabalho
  pesado; aqui a malha deve existir, então é o caminho "fácil" dos mapas virais, para comparar
  qualidade e custo entre os dois fluxos.
- Centro planejado (Aarão Reis, 1897): quadrícula cortada por diagonais dentro da Avenida do
  Contorno, muito reconhecível de cima.
- Relevo: o centro fica entre ~800 m (Arrudas) e ~1.100 m (Serra). Teste bom para a câmera e as
  máscaras com altura única.

## Região (`regions/belo-horizonte.yaml`)

| | Fase 1 (centro) | Piloto |
|---|---|---|
| Área | dentro do Contorno + Savassi, Lourdes, Funcionários, Santa Efigênia | Praça Sete → Praça da Liberdade |
| Tamanho | ~5,0 × 4,7 km, ~24 km² | ~1,7 × 1,9 km |
| Escala | 0,3 m/px | 0,3 m/px |
| Canvas (`tinymap plan`) | 22.528 × 18.432 px, 0,42 Gpx, 70 shots | 8.192 × 10.240 px, 16 shots |

Fase 2, região própria: Pampulha (Lagoa, Igreja São Francisco de Assis, Mineirão, Museu de Arte).

Os 10 pontos de referência (Praça Sete, Praça da Liberdade, Mercado Central, Edifício JK, Parque
Municipal, Viaduto Santa Tereza, Praça da Estação, Savassi, Igreja de Lourdes, Palácio das Artes)
foram projetados com a câmera e caem dentro do canvas. As coordenadas são aproximadas: `tinymap osm`
traz as do OpenStreetMap.

## Passo a passo (3090 local)

```bash
# 0. chave do Google (Map Tiles API) ou token Cesium ion no .env
tinymap plan   regions/belo-horizonte.yaml --pilot
tinymap probe  regions/belo-horizonte.yaml --pilot      # 1 km em cada ponto: a malha é boa?
#    se algum ponto "deslizar" em relação ao prédio, ajuste camera.target_height_m
tinymap render regions/belo-horizonte.yaml --pilot --workers 2
tinymap osm    regions/belo-horizonte.yaml --pilot
tinymap stitch regions/belo-horizonte.yaml --pilot      # prévia do render

# 1. pares de treino com revisão humana (estúdio) e laboratório de LoRA
tinymap teacher regions/belo-horizonte.yaml --pilot --backend qwen --out pairs_v1 --n 120
tinymap review  regions/belo-horizonte.yaml --pilot
tinymap lab-freeze regions/belo-horizonte.yaml --pilot --pairs pairs_v1
tinymap lab-eval   regions/belo-horizonte.yaml --pilot --name base
tinymap lab-eval   regions/belo-horizonte.yaml --pilot --name base --seed 1
tinymap lab-train  regions/belo-horizonte.yaml --pilot --pairs pairs_v1 --name r1
#    (ai-toolkit) e depois: lab-eval --name r1 --lora runs/belo-horizonte-pilot/lora/r1/out

# 2. mapa do centro
tinymap plan     regions/belo-horizonte.yaml
tinymap render   regions/belo-horizonte.yaml --workers 3
scripts/gpus.sh  regions/belo-horizonte.yaml <melhor LoRA>
tinymap autofix  regions/belo-horizonte.yaml --lakes
tinymap polish   regions/belo-horizonte.yaml
tinymap edit     regions/belo-horizonte.yaml            # estúdio: saúde do mapa e retoques
tinymap site     regions/belo-horizonte.yaml
```

Vale testar se a LoRA treinada na Tríplice Fronteira já serve em BH (`lab-eval` com ela no teste de
BH): se a nota ficar perto, não precisa de outro treino. É a primeira medida de quanto o estilo se
transfere entre cidades, que é o que a plataforma precisa para escalar.

## Custo e tempo (estimativa, a medir no piloto)

- Render: 70 shots de 2048 px pela Map Tiles API (3D Tiles); confira a cota grátis e o preço atual
  no painel do Google Cloud antes do render inteiro.
- Geração: 0,42 Gpx. Pela conta do tiny.london (31 H100-h para 14 Gpx), algumas horas de 3090.

## Riscos

- **Malha do Google**: confirmar no `probe`; árvores e prédios altos colados costumam derreter na
  malha. Se o centro vier ruim, o caminho é o mesmo da fronteira (`source: osm` + Overture + satélite).
- **Água inventada em coberturas**: piscinas e caixas d'água azuis viram "lagos" na pintura; o
  negativo já pede "swimming pools", e o `autofix` só apaga onde a malha não tem azul embaixo.
- **Relevo**: altura única da câmera em cidade de morro. Afeta só pontos de referência e máscaras,
  não o render.
- **Licença**: imagem derivada do Google 3D Tiles exige atribuição; para uso comercial, use o
  Qwen-Image-Edit-2511 (Apache) como professor e base da LoRA (já está assim no YAML).
