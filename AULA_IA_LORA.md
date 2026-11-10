# Aula: como estamos ensinando uma IA a desenhar Belo Horizonte

> Projeto **tinymap BH**: mapa isométrico de Belo Horizonte no "estilo B"
> (cel-shaded, 12 cores, contorno preto fino, sombras sem borda).
> Esta aula explica, passo a passo e do zero, o que fizemos e por quê.

---

## 0. O problema em uma frase

Queremos transformar **imagens 3D realistas da cidade** (do Google) em **ilustrações**
no estilo B, **para a cidade inteira**, com o **mesmo estilo em todos os pedaços** e a
**mesma disposição** (cada prédio, rua e árvore no lugar certo).

Fazer isso à mão é inviável: são milhares de quadrados. A solução é **ensinar uma IA**
a fazer a conversão sozinha.

---

## 1. Conceitos básicos

### 1.1 O que é um modelo de IA de imagem

Um modelo como o **Qwen-Image-Edit** é uma função gigante com **bilhões de números**
(os *pesos* ou *parâmetros*). Ele recebe uma imagem e um texto (o *prompt*) e devolve
uma imagem nova.

Esses pesos foram ajustados num treino enorme (milhões de imagens, semanas de GPUs
caríssimas). Por isso o modelo já "sabe" desenhar, mas desenha **do jeito dele**, não
exatamente no nosso estilo B.

### 1.2 Difusão: como a imagem nasce

Os modelos modernos de imagem são **modelos de difusão**:

1. Começam de uma imagem de **puro ruído** (chuvisco de TV).
2. A cada **passo**, a rede prevê "qual parte disso é ruído" e tira um pouco.
3. Depois de N passos sobra uma imagem limpa.

O texto (prompt) e a imagem de entrada (o render) **guiam** essa limpeza: em cada passo
a rede tenta chegar numa imagem que combine com eles.

- **Mais passos** em geral dão mais qualidade, mas levam mais tempo.
- O **Lightning** é um "atalho" treinado que faz o modelo chegar ao resultado em
  **4 passos** em vez de ~30–50. Por isso cada imagem nossa leva ~15 s numa RTX 3090.
- **true_cfg** é o quanto o modelo "obedece" o prompt versus solta a criatividade.
- A **semente** (*seed*, ex.: 3110) fixa o ruído inicial: mesma semente e mesma entrada
  dão a mesma imagem. Por isso usamos sempre a mesma nos testes, para comparar de forma
  justa.

### 1.3 Modelo de edição (image-to-image)

O **Qwen-Image-Edit-2511** não desenha do nada: ele **edita** uma imagem. Damos o render
do Google e pedimos "redesenhe isso como ilustração". Isso é ótimo para nós, porque a
disposição dos prédios vem do render.

### 1.4 Quantização: caber na GPU

O modelo inteiro em precisão normal não cabe nos 24 GB da RTX 3090. Por isso usamos
versões **quantizadas**:

- **NF4** (4 bits) para gerar imagens: cada número ocupa 4 bits em vez de 16. Perde um
  pouco de precisão, mas cabe.
- **qfloat8** (8 bits) no treino da LoRA.

É como salvar uma foto em JPEG em vez de RAW: menor, quase igual.

### 1.5 GPU, VRAM e o "OOM"

A GPU é o processador que faz as contas da IA. A memória dela (**VRAM**, 24 GB na 3090)
precisa guardar o modelo e os cálculos. Dois trabalhos de IA ao mesmo tempo estouram a
memória (**CUDA OOM**, *out of memory*). Por isso a regra **"um job de GPU por vez"**.
Já perdemos um lote (`pairs_bc2`) assim.

---

## 2. O que é LoRA (a parte principal)

### 2.1 O problema de re-treinar o modelo inteiro

Para o modelo aprender o estilo B, poderíamos ajustar **todos** os bilhões de pesos
(*fine-tuning completo*). Mas isso:

- precisa de várias GPUs grandes;
- gera um arquivo de dezenas de GB;
- com poucos exemplos (dezenas), o modelo **"decora"** os exemplos e esquece o resto
  (*overfitting*).

### 2.2 A ideia da LoRA

**LoRA = Low-Rank Adaptation** (adaptação de posto baixo).

Dentro do modelo há muitas **matrizes** de pesos grandes, por exemplo 3072 × 3072
(≈ 9,4 milhões de números cada). A LoRA **congela** essas matrizes e aprende só uma
**correção pequena** para cada uma:

```
W_novo = W_original  +  B × A
         (congelado)    (aprendido)
```

- **A** tem tamanho r × 3072 e **B** tem tamanho 3072 × r.
- **r** é o *rank* (posto), por exemplo 16 ou 32.
- Com r = 16: 2 × 3072 × 16 ≈ **98 mil** números em vez de 9,4 milhões, ou seja,
  **~1%**.

**Analogia:** o modelo é um pintor experiente. Não vamos reeducá-lo do zero; vamos dar
a ele um **caderninho de anotações** ("nesse cliente: contorno preto fino, cores
chapadas, sombra sem borda"). O caderninho é a LoRA: pequeno, barato e removível.

Vantagens:

- treina em **1 GPU** de 24 GB;
- o arquivo final tem dezenas a centenas de MB (`belo-horizonte_bh_bd70.safetensors`);
- dá para **ligar e desligar**, e até **dosar** (`--lora-scale 0.7` = "use 70% do
  caderninho");
- como mexe em pouca coisa, aprende o **estilo** sem esquecer como desenhar.

### 2.3 O que é um "par" de treino

A LoRA aprende por **exemplos de entrada → saída desejada**. Cada **par** é:

| entrada | saída desejada |
|---|---|
| `fXXXX_render.png` (render Google 3D) | `fXXXX_art.png` (ilustração no estilo B, aprovada) |

O treino mostra o par ao modelo milhares de vezes e ajusta A e B para que, dado o render,
a saída fique **parecida com a arte aprovada**.

**Qualidade dos pares = qualidade da LoRA.** Se aprovamos pares tremidos, a LoRA aprende
a tremer. Por isso gastamos tanto tempo escolhendo pares bons.

### 2.4 Como o treino funciona por dentro

Em cada **passo** do treino:

1. Pega um par (render, arte).
2. Pega a arte, **suja com ruído** num nível aleatório.
3. Pede ao modelo (com a LoRA) para prever o ruído, sabendo do render e do prompt.
4. Compara a previsão com o ruído verdadeiro. O tamanho do erro é a **loss** (perda).
5. Usa cálculo (gradiente) para ajustar **só A e B** um pouquinho na direção que reduz
   o erro. O tamanho desse ajuste é o **learning rate** (`lr: 1.0e-04`).

No nosso log aparece:

```
belo-horizonte_bh_bd70: 93% | 1493/1600 [5:13:06, 12.49s/it, lr: 1.0e-04 loss: 7.806e-02]
```

- **1600 passos** no total, **~12,5 s por passo** na 3090, **~5,6 h** no total.
- A **loss** oscila (0,07 a 0,15) porque cada passo usa um par e um nível de ruído
  diferentes. O que importa é a tendência média.
- **Checkpoints** foram salvos nos passos 400, 800 e 1200. Se a versão final tiver
  *overfitting* (decorado demais), usamos uma anterior.

### 2.5 `lab-freeze` e `lab-train`

No tinymap:

- **`tinymap lab-freeze`** "congela" o conjunto de pares: copia e organiza os pares
  escolhidos no formato que o treinador espera. Assim o treino é reproduzível, porque
  sabemos exatamente com o que treinamos.
- **`tinymap lab-train`** chama o **ai-toolkit** (o programa de treino de LoRA) com o
  modelo base, os pares, o *rank*, o *learning rate*, o número de passos e a quantização
  qfloat8.

---

## 3. Passo a passo do que fizemos

### Passo 1: render 3D (a entrada)

Com o **Google Photorealistic 3D Tiles** renderizamos Belo Horizonte com uma câmera
**isométrica** fixa (mesmo ângulo e mesma escala). A cidade é cortada em **recortes**
(*frames*) de 512 ou 1024 px: `f0010_512`, `f0015_1024` etc.

### Passo 2: o "professor" (Qwen sem LoRA) gera candidatos

Como ainda não temos LoRA, usamos o Qwen "cru" com um **prompt detalhado** (o *prompt
BC*: 12 cores, 2 tons por parede, contorno de 2 px, sombra sem borda, não inventar
água...). Ele é o **professor**: gera 1 a 3 **candidatos** por recorte (`_try1`,
`_try2`...).

Os **checks automáticos** (`checks.json`, `report.txt`) já reprovam erros óbvios:

- `photo-copy` / `render-copy`: só copiou a foto, não desenhou;
- `invented-water` / `land-on-water`: inventou lagos ou rios;
- `blank`: imagem vazia.

Lotes gerados: `pairs_bc` (60), `pairs_bc2` (90), depois `pairs_bd` (90) e `pairs_bd2`
(120).

### Passo 3: tentativas de "endireitar" as linhas (o que NÃO funcionou)

Queríamos contorno perfeitamente reto. Tentamos três fontes de geometria:

1. **Vetorizar pelo RGB do render.** Ruim: a foto tem textura demais.
2. **Profundidade/normal da malha do Google.** A malha 3D do Google vem de
   fotogrametria e é **ruidosa**, então o contorno saiu picotado.
3. **Prédios do OpenStreetMap extrudados** (caixas perfeitas). As caixas **não batem**
   com a foto: as posições e alturas do OSM são diferentes das do Google. As linhas
   ficavam no lugar errado.

**Lição:** um contorno perfeito em cima de uma geometria errada é pior do que um contorno
imperfeito na geometria certa.

### Passo 4: deixar mais chapado (o que funcionou)

Testamos 4 variantes nos mesmos 6 recortes e com a mesma semente:

| Variante | O que é | Resultado |
|---|---|---|
| A | prompt atual | base |
| B | prompt reforçado ("flat solid colour, ruler-drawn edges...") | ajuda pouco |
| C | B + 8 passos (Lightning-8) | **pior**: mais pintado, 2× mais lento |
| **D** | **B + pós-processamento** | **vencedora** |

O **pós-processamento D** é processamento clássico de imagem (sem IA, roda na CPU em
< 1 s):

1. **Filtro bilateral**: suaviza cores **sem borrar bordas** (só mistura pixels
   vizinhos que já têm cor parecida).
2. **Mean-shift** (`pyrMeanShiftFiltering`): agrupa regiões de cor parecida em
   **blocos uniformes**, o que dá o efeito "chapado".
3. **Quantização em 48 cores sem dither**: limita a paleta e elimina o chuvisco.

Métricas (média dos 6 recortes): o índice de imagem chapada (`flat`) subiu de 0,24 para
0,39 e o de tremido (`soft`) caiu de 0,28 para 0,07.

### Passo 5: janelas regulares (tentado, abandonado)

A ideia era detectar a grade de janelas das torres e redesenhá-las como **cópias
idênticas**. Duas versões falharam: a detecção confundia telhado, chão e sombra com
fachada. Uma imagem 2D chapada não diz com segurança "isto é uma parede vertical".
Também testamos o Opus desenhando em código (SVG): ficou perfeitamente reto, mas
**infiel** ao lugar e caro. Ambos foram descartados.

### Passo 6: revisão humana (HITL, *human in the loop*)

O `tinymap review` mostra cada recorte com os candidatos. Você aprova, refaz ou descarta.
Fizemos uma **versão para celular** (S25 Ultra): deslizar entre as opções, tocar e
aprovar. A aprovação grava `review.json` e o `_art.png` (que é o "gabarito" do par).

Você aprovou **24 pares** (4 em `pairs_bc`, 8 em `pairs_bc2`, 12 em `pairs_bd`).

### Passo 7: nota automática (imitar o seu gosto)

Para não depender só de revisão manual, criamos um **classificador**:

1. **Features** (medidas numéricas) por candidato, comparando com o render: quão
   chapado, quão tremido, quantas linhas retas longas (**transformada de Hough**), se o
   contorno é escuro e fino, quantas cores, quão fiel às bordas do render (**SSIM**,
   **IoU** de bordas) e se copiou a foto.
2. **Regressão logística**: um modelo simples que aprende **um peso por feature**
   ("chapado conta +, tremido conta −") a partir dos seus 24 aprovados (positivos) e
   dos outros candidatos dos mesmos recortes (negativos).
3. **Validação deixando um de fora (LOO, *leave-one-out*)**: tira um aprovado, treina
   com o resto e vê se o modelo escolheria o mesmo. Acertou **20 de 24 (83%)**.
4. **Limiar 0,63**: só aprova automaticamente quem passa dessa nota.

Um bug ensinou algo: na primeira versão **todos os pesos saíram zero**, porque as
features não estavam padronizadas e os pares não vistos contavam como "reprovados".
Na segunda versão, corrigida, o script funcionou.

**Cuidado observado:** no lote `pairs_bd2` ele aprovou 110 de 120, com notas quase 1,0.
É sinal de **saturação**: o modelo, treinado com poucos exemplos, fica confiante demais
em dados um pouco diferentes. Por isso não usamos todos: pegamos só os 35 de maior nota.

### Passo 8: treino da LoRA `bh_bd70`

Conjunto final: **70 pares**.

- 24 aprovados por você;
- 11 escolhidos automaticamente em `pairs_bd` e `pairs_bc2`;
- 35 de maior nota em `pairs_bd2`.

Treino: 1600 passos, cerca de 5,6 h na RTX 3090, **cerca de US$ 1,60**.
Saída: `belo-horizonte_bh_bd70.safetensors` mais os checkpoints 400, 800 e 1200.

### Passo 9: teste da LoRA (em andamento)

Geramos 6 recortes que **não estavam no treino** (para não "colar na prova") e
comparamos:

1. Render Google 3D;
2. Qwen sem LoRA (prompt B + pós D);
3. Qwen + LoRA `bh_bd70`.

Se a LoRA for melhor, ela vira o **novo professor**. A partir daí, gerar a cidade
inteira fica mais consistente, e um novo ciclo de pares melhores pode treinar uma LoRA
ainda melhor (**bootstrapping**).

---

## 4. O ciclo completo (resumo visual)

```
 Render Google 3D ──► Qwen + prompt (professor) ──► candidatos
                                                      │
                                     checks automáticos (água, cópia)
                                                      │
                                     pós D (chapado: bilateral + mean-shift + quant.)
                                                      │
                               revisão humana (celular) + nota automática
                                                      │
                                              pares aprovados
                                                      │
                                lab-freeze ──► lab-train (ai-toolkit, LoRA)
                                                      │
                                        LoRA bh_bd70 = novo professor
                                                      │
                                     (repete o ciclo com qualidade maior)
                                                      ▼
                                       mapa inteiro de Belo Horizonte
```

---

## 5. Custos e infraestrutura

| Item | Valor |
|---|---|
| Máquina vast.ai (Omã, RTX 3090) | US$ 0,28/h |
| Gerar 1 candidato (512 px, Lightning-4) | ~15 s, cerca de US$ 0,001 |
| Pós D | < 1 s de CPU (grátis na prática) |
| Treino LoRA (70 pares, 1600 passos) | ~5,6 h, cerca de US$ 1,60 |
| Banda de download | US$ 53/TB: evitar baixar modelos de novo |

Regras de segurança que seguimos:

- **nunca** parar ou reiniciar a máquina (perde a GPU);
- **nunca** `pkill -f` (mata o próprio shell): matar só pelo PID exato;
- **um job de GPU por vez**;
- trabalhar em **cópias**, nunca no `/workspace/tinymap` original.

Acesso: SSH por **túnel Cloudflare** (`cloudflared access ssh`), porque a sessão na nuvem
só sai por HTTPS. As imagens ficam publicadas por outro túnel
(`editing-pdas-hayes-candle.trycloudflare.com`).

---

## 6. Glossário rápido

- **Prompt**: o texto que guia a IA.
- **Prompt negativo**: o que evitar (o Lightning o ignora).
- **Difusão**: gerar imagem tirando ruído passo a passo.
- **Lightning**: atalho para gerar em 4 passos.
- **Seed (semente)**: fixa a aleatoriedade, para que testes sejam comparáveis.
- **Quantização (NF4, qfloat8)**: guardar números com menos bits para caber na GPU.
- **VRAM / OOM**: memória da GPU; *out of memory* = estourou.
- **LoRA**: correção pequena e treinável (B × A) sobre um modelo congelado.
- **Rank (r)**: tamanho da correção; maior = mais capacidade e mais risco de decorar.
- **Par**: entrada (render) + saída desejada (arte aprovada).
- **Loss**: erro do modelo num passo de treino.
- **Learning rate**: tamanho de cada ajuste.
- **Checkpoint**: foto do estado do treino num certo passo.
- **Overfitting**: decorar os exemplos e piorar em casos novos.
- **HITL**: *human in the loop*, a pessoa revisa e corrige.
- **Regressão logística**: classificador simples que soma features com pesos.
- **LOO**: validação deixando um exemplo de fora por vez.
- **Bilateral / mean-shift**: filtros clássicos que suavizam sem borrar bordas.
- **Hough**: técnica que acha linhas retas numa imagem.
- **SSIM / IoU**: medidas de semelhança entre imagens ou regiões.

---

## 7. Anexo: o que aprendemos com o tiny.london (lido em 10/11/2026)

Fonte: https://tiny.london/how/ (Boris Starkov, outubro 2026). Os números são declarados
pelo autor.

**Pipeline:**

1. **Render:** uma câmera ortográfica única para Londres inteira, com 3.570 fotos de
   2.048 px que se alinham ao pixel; mover o *frustum* em vez de mover a câmera.
2. **Professor:** o Gemini 3 Pro Image redesenha cerca de 2.500 quadros de 1.024 px.
   Testes automáticos comparados com o render e com o OSM rejeitam 31%. **Nenhuma
   rotulagem manual.**
3. **LoRA:** rank 32, 8.000 passos, lr 1e-4, 6.326 exemplos, ai-toolkit, sobre o
   **Qwen-Image-2.1-Turbo** em int8: 3h50 numa H100. A v1 deles, com 86 pares, também
   saiu fraca.
4. **Geração sem emenda (MultiDiffusion):** **um latente só** para o mapa inteiro. A cada
   passo, o modelo roda em janelas de 1.024 px com passo de 768 px e as previsões são
   somadas com um peso em "tenda" (zero na borda da janela). Nos passos ímpares a grade
   fica deslocada em 384 px. Pixels prontos ficam **presos** (*pinning*, método RePaint),
   e o novo cresce a partir deles. Os blocos de 12.288 px rodam em 4 fases pela paridade
   de linha e coluna.
5. **Água:** o OSM indica onde há água. Objetos inventados na água são removidos com uma
   textura de água próxima mais uma **correção de Poisson** (sem emenda).

**Lições para nós:**

- O prompt deve dizer **como** desenhar, nunca **o quê**: "desenhe exatamente o que o
  render mostra; não acrescente nada; onde há pouco detalhe, mantenha simples".
- Incluir uma frase por quadro gerada do OSM: "WATER IN THIS VIEW: none...". Isso
  acaba com lagos inventados.
- Teste de cópia da foto: fração de pixels com vizinhança 3×3 chapada. Desenho bom fica
  entre 0,42 e 0,55; cópia, entre 0,18 e 0,35.
- Diversidade do treino: embeddings DINOv2, 64 grupos por aparência, de 8 a 30 exemplos
  por grupo, mais lugares difíceis e quadros com zoom de 2×.
- A LoRA precisa de **centenas a milhares** de pares escolhidos automaticamente. Com
  38 a 86 pares, ela fica fraca.
- O pós-processamento (nosso filtro D) deve rodar **uma vez no mapa montado**, não em
  cada janela, para não criar emenda.
- Salvar estados atômicos e o latente a cada passo, para retomar sem perder trabalho.
- O Turbo amostra em 8 passos sem CFG, cerca de 4× mais rápido que o Edit com guidance.
