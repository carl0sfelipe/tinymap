# Fichas de fachada (foto da rua → janelas certas)

Para prédio alucinado ou parede lisa. A foto não vai para o desenho: vira **dado** numa ficha
(andares, janelas por andar, pé-direito, cor da parede, loja no térreo) que você confirma.

- **Região sem 3D do Google** (`source: osm`): a ficha desenha a grade de janelas, a vitrine e o
  toldo na maquete (`render/osmscene.mjs`), e a LoRA só repinta no estilo.
- **Região com 3D do Google** (BH): a malha já tem janela; a ficha confirmada e a foto vão junto
  no Exportar do estúdio, com a frase "N floors, M windows per floor..." no prompt.

## Manual (padrão, sem custo)

1. Estúdio → marque o prédio com ⬚ → aba **Fachada**.
2. "abrir o Street View aqui" (ou vá à rua), tire um print ou foto e **cole com Ctrl+V** na aba.
3. O sistema sugere andares e janelas (periodicidade das bordas). Corrija, **Confirmar ficha**.
4. **Levar fichas confirmadas para a maquete** (refaz o render se a região é `source: osm`).

Linha de comando: `tinymap fachada <região> --at lat,lon --file foto.jpg` e `tinymap fachada-aplicar <região>`.

## Automático

- **Mapillary** (fotos abertas, CC-BY-SA): `MAPILLARY_TOKEN` no `.env`. Escolhe a foto mais perto
  que olha para o prédio (rumo da câmera a até 40°).
- **Street View Static API**: `GOOGLE_MAPS_API_KEY` com a API ativada e `fachada.streetview_max_mes`
  no YAML (0 = desligado). Acha o panorama mais perto (metadados, grátis), calcula rumo, inclinação
  e campo de visão para enquadrar a fachada e baixa uma imagem de 640 px. O contador mensal fica em
  `runs/streetview_uso.json` e para no teto.
  Preço: a imagem é cobrada por requisição; confira a cota grátis e o valor no painel do Google Cloud.
  Termos: a Google Maps Platform restringe extrair dados e criar conteúdo a partir das imagens do
  Street View. Para uso comercial, prefira foto própria ou Mapillary.
