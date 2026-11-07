# Relatório da noite (07/11/2026)

## O que mudou no fluxo

1. **Maquete com a cidade real**: prédios do Overture (146.740 na região, contra ~2.700 só do OSM no
   piloto) + **satélite do Google** (Map Tiles API, z19, 33.142 tiles) colado no chão e nos telhados.
2. **Rio e ruas pintados por cima da foto**: rio do OSM em azul e ruas do Overture em cinza, e sombras
   clareadas. Isso ataca as três reclamações: "rios no lugar de ruas", "lagos no meio da cidade" e
   "casas onde não tem".
3. **Estilo**: prompt de pixel art com `true_cfg 2.5` e "lagos, poças, canais" no prompt negativo.
   Testado no piloto: com cfg 1 sai foto com contorno, com cfg 4 vira tudo grama e inventa poças.
4. **Conserto automático**: depois de cada bloco, o código acha água fora do rio e terra dentro do rio
   (comparando com o render) e redesenha só aqueles pedaços.
5. **Escala 0,4 m/px** na região inteira, o que corta o custo pela metade. O piloto continua em 0,293.

## Problemas encontrados e resolvidos

- Lentes de chão no meio do Paraná: o earcut do three.js falhava no polígono do rio (10 mil pontos +
  ilhas). Corrigido cortando a água em células de ~500 m.
- O "cadeado" por arquivo não funciona no Volume da Modal (dois workers pegaram o mesmo bloco): troquei
  por uma divisão fixa de blocos por worker e por uma regra para assumir blocos pendentes após 25 min.
- A conta da Modal tem **limite de 10 GPUs simultâneas** (pedi 15, só 10 ligaram). Relancei com 10.
- `modal app stop` sem `--yes` não para nada fora de um terminal interativo.
- **1º run do mapa perdido (~1 h de 10 H100, ~US$ 40)**: um bug no conserto (caixa maior que o bloco)
  derrubou um worker, e a Modal **preemptou** os outros no meio do bloco, sem nada salvo. Corrigido: a arte
  é salva logo depois da pintura (checkpoint), o conserto retoma de lá, e a caixa ficou limitada ao bloco.
- Blocos de borda: fora da área do mapa o render não tem nada (chão bege e azul-céu), e o modelo inventa
  cidade e canais. O site agora recorta a arte pelo polígono da área do mapa.

## Custos

| Item | US$ |
|---|---|
| **Faturamento real de novembro** (`modal billing summary`) | **194,40 medidos − 30 de crédito = 162,53** |

Quase tudo veio desta sessão. Cerca de US$ 40 foram perdidos num run sem nada salvo (bug no conserto +
preempção). A geração foi interrompida a pedido às 07:44, com 26 de 54 blocos.

## Resultado

- **26 de 54 blocos** pintados (as fases 0 e 1, faixas pares). O resto do mapa aparece como maquete com
  satélite, tanto no site quanto na oficina.
- A cidade ficou fiel: ruas, quadras, rios, ilhas e matas no lugar, em pixel art.
- Defeito que sobrou: lagoas e rios inventados, sobretudo em lavouras. O caçador de lagos da oficina
  encontra ~400 manchas (~300 ha) dentro da área.
- **Oficina do mapa** (`tinymap edit`, porta 8766): conserto à mão, apagar lagoa, exportar para o Gemini
  com prompt e referência de estilo, reimportar alinhado e ajustado à paleta, histórico com desfazer, e
  um botão para publicar no site (porta 8080).

Para terminar os 28 blocos que faltam: `modal run tinymap/modal_app.py::generate ... --gpus 10 --block 4096
--margin 512 --steps 12` retoma de onde parou (~1 h, ~US$ 40-60). Só com um teto combinado.
