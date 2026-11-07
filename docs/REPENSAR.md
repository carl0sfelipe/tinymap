# Repensando o fluxo: fidelidade aos prédios reais (nov/2026)

## O que deu errado

A IA não tinha como ser fiel: **a entrada não continha a cidade real**. A maquete do OSM tem só os
prédios mapeados à mão, em cores chapadas, e grandes áreas lisas de verde e bege. Diante de uma área
lisa, qualquer modelo de edição inventa (lago, cidade, floresta). Prompt, dicas e checagens só
tratam o sintoma.

Medido no piloto (Marco das Três Fronteiras, ~2 × 2 km):

| Fonte | Prédios | Com altura |
|---|---|---|
| OSM (maquete atual) | 2.698 | 991 elementos com height/levels |
| Overture 2026-10-21 (OSM + Google Open Buildings + Microsoft ML) | 3.947 (+46%) | 0 height, 960 num_floors |
| Satélite (Esri World Imagery, ~0,5 m/px) | todos visíveis: telhado, cor, árvore, rua | — |

Todos os mapas virais que ficaram fiéis (NYC, Londres, Toronto) partiram de **imagem real**
(fotogrametria do Google 3D) e só trocaram o estilo. A gente partiu de um desenho esquemático.

## Novo fluxo proposto

1. **Geometria**: prédios do Overture (OSM + detecção por satélite), altura por andares/tipo como hoje.
2. **Textura real**: imagem de satélite projetada sobre a maquete (chão e telhados). O render passa a
   parecer uma fotogrametria barata: cada telhado com a cor e a forma reais, árvores onde há árvores.
   Paredes saem da extrusão. Mesma câmera, mesma projeção.
3. **Trocar o estilo sem redesenhar**: em vez de "redesenhe do zero" (força 100%, onde nasce a
   alucinação), partir do próprio render com ruído parcial (img2img / SDEdit, força ~0,4–0,6). O
   modelo só consegue mudar a aparência, não a planta. Qwen-Image-Edit-2511 (Apache-2.0, permite uso
   comercial) em vez do Qwen-Image-2.1 (só não comercial).
4. **Revisão humana continua** (tinymap review), mas comparando com o satélite e julgando estilo, não
   caçando invenção. Checagens viram "fidelidade": bordas dos prédios do render vs. arte.
5. LoRA de estilo com os pares aprovados, como antes.

## Validar antes de reconstruir (barato)

Um recorte de Puerto Iguazú (denso) e o do Marco, ponta a ponta: maquete Overture + satélite →
render → 2511 com força 0,4 / 0,6 / 1,0 → lado a lado com o satélite. ~US$ 0,50 de H100.

## Decisões do usuário

- **Fonte do satélite**: Esri World Imagery serve para testar (termos: uso não comercial com atribuição).
  Para o mapa final, uma fonte com licença clara: Google Map Tiles API (2D satellite, precisa de chave
  e faturamento; é o mesmo caminho legal do tiny.london), Mapbox, ou ortofoto da prefeitura de Foz.
- **Uso comercial (now.services)**: decide o modelo (2511 Apache) e a fonte da imagem.
