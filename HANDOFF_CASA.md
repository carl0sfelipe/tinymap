# Handoff: tinymap BH, continuar na máquina de casa (RTX 3090)

Cole o texto abaixo num Claude Code aberto na sua máquina de casa.

---

Contexto: projeto tinymap BH, mapa isométrico de Belo Horizonte no "estilo B"
(cel-shaded, contorno preto fino, cores chapadas, sombra sem borda). Respostas curtas em
português simples: 1 frase, até 3 pontos e 1 ação. Toda imagem de comparação leva
RÓTULO em cada quadro (origem, modelo, prompt).

Leia antes: AULA_IA_LORA.md (conceitos, histórico e lições do tiny.london) neste repo
(carl0sfelipe/tinymap, branch claude/bold-fermi-786pbl).

## Máquinas

1. **Omã (vast.ai, RTX 3090, US$ 0,28/h)**
   - SSH pelo túnel:
     `ssh -i <sua chave> -o ProxyCommand="cloudflared access ssh --hostname manager-divisions-undertake-fraser.trycloudflare.com" root@oma`
   - Ou direto, se a sua rede permitir: `ssh -p 41357 root@82.178.70.231`.
   - Leia /workspace/HANDOFF.md lá (regras e armadilhas).
   - Imagens públicas: /workspace/srv/<pasta> -> https://editing-pdas-hayes-candle.trycloudflare.com/<pasta>/
   - O saldo acaba por volta das 3h de quarta, 11/11 (Brasília), se o Carlos não
     depositar mais.
2. **Casa (RTX 3090, 24 GB):** onde você está agora.

## Regras

- NUNCA parar, reiniciar ou destruir a máquina de Omã. NUNCA usar `pkill -f` (matar só
  pelo PID exato).
- Um job de GPU por vez em cada máquina (dois juntos dão CUDA OOM).
- Não editar o original /workspace/tinymap em Omã: usar a cópia
  /workspace/cur/chapado/repo.
- Banda em Omã custa US$ 53/TB: nada de baixar modelos grandes de lá. Em casa, baixe do
  Hugging Face.
- Avisar o Carlos do custo antes de qualquer treino.

## Estado (10/11, ~6h Brasília)

- **Receita aprovada pelo Carlos** (folha
  https://editing-pdas-hayes-candle.trycloudflare.com/nota/folha_04_bd2_auto.png):
  - Qwen-Image-Edit-2511 em NF4 com Lightning-4 passos, SEM LoRA, semente 3110;
  - prompt B: o prompt BC de regions/belo-horizonte-bc.yaml mais "flat solid colour
    fills, perfectly straight ruler-drawn edges, no wobble, no sketchy lines, no painterly
    texture";
  - recorte 512 px (ampliado para 1024, pintado e reduzido para 512);
  - pós D: `cv2.bilateralFilter(d=7, sigmaColor=40, sigmaSpace=7)`, depois
    `pyrMeanShiftFiltering(sp=8, sr=16, maxLevel=1)`, depois quantização em 48 cores
    MEDIANCUT sem dither.
- **Piloto pronto:** runs/belo-horizonte-pilot/art_piloto_bd2. Prévia:
  https://editing-pdas-hayes-candle.trycloudflare.com/piloto/mapa_piloto.jpg
  (~3,3 h, 758 janelas).
- **Mapa todo rodando em Omã** (tmux "mapa", log /workspace/mapa.log, termina com
  FIM_MAPA):
  - região regions/belo-horizonte-mapa.yaml, saída runs/belo-horizonte;
  - canvas 22.528 × 18.432 px; render feito (70 fotos); geração em 12 blocos de ~441
    janelas, começada por volta das 5h45 (Brasília);
  - estimativa: ~16–17 h só em Omã;
  - progresso em /workspace/srv/mapa/progress.json e previa.jpg;
  - um cursor-agent (tmux "noite") supervisiona; relatório em
    /workspace/cur/piloto/NOITE.md.
- **LoRA bh_bd70** (70 pares, 38 de fato no treino): o Carlos achou PIOR que sem LoRA.
  Não usar.
- **Revisão no celular** (usuário carlos):
  - pairs_bc: https://loud-knew-secret-polls.trycloudflare.com
  - pairs_bc2: https://girlfriend-therapy-cursor-weddings.trycloudflare.com
  - pairs_bd: https://exactly-specs-gotta-birmingham.trycloudflare.com
  - a senha está em /workspace/senha_proxy.py, nos argumentos do processo; não a
    escreva em lugar público.

## Tarefas para a máquina de casa

1. **Montar o ambiente:**
   - Linux ou WSL2 com CUDA, Python 3.11+ e ~60 GB livres;
   - clonar o tinymap: copie a pasta /workspace/cur/chapado/repo de Omã com rsync, sem
     a pasta runs, porque ela tem as mudanças do generate;
   - `pip install -e .` e as dependências do repo;
   - baixar do Hugging Face o Qwen/Qwen-Image-Edit-2511, a versão NF4 usada em Omã
     (veja em tinymap/models.py qual repo e quant) e a LoRA Lightning de 4 passos;
   - rodar `tinymap doctor`.
2. **Dividir a geração do mapa com Omã:**
   - o generate trabalha em blocos e fases; blocos da mesma fase não se tocam, então
     cada máquina pode rodar blocos diferentes;
   - copiar de Omã runs/belo-horizonte/{view.json,render/,…}, umas centenas de MB,
     barato;
   - combinar com Omã quais blocos cada máquina faz: por exemplo, Omã faz as fases 0 e
     2, e a casa faz 1 e 3, respeitando a ordem, porque as fases seguintes usam a
     margem dos blocos prontos;
   - devolver os blocos prontos com rsync para Omã ou para a casa e montar tudo num
     lugar só;
   - verificar se o gerador de Omã já pegou um bloco antes (locks), para não duplicar.
3. **Ao terminar:**
   - aplicar o pós D UMA vez no mapa montado, se a adaptação atual estiver fazendo por
     janela (veja a lição do tiny.london);
   - rodar `tinymap autofix --lakes` para a água inventada;
   - rodar `tinymap site` para gerar a pirâmide Deep Zoom e o visualizador;
   - publicar a prévia rotulada.
4. **Depois (opcional, avisar o custo antes):** testar o método do tiny.london
   (MultiDiffusion num latente só, prompt "como e não o quê", frase de água do OSM)
   num pedaço, e mais tarde uma LoRA com centenas ou milhares de pares escolhidos
   automaticamente, de preferência no Qwen-Image-2.1-Turbo.
