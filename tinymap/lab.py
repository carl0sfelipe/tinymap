"""Laboratório de LoRA: medir cada treino contra um conjunto de teste fixo, como em
marceloeduardo.com/pixel-lora-training (MiniDF, Brasília). Roda na 3090 local.

A ideia: "parece melhor" não serve para comparar LoRAs. Cada treino ("run") pinta os MESMOS 32
recortes que nunca entram no treino, e recebe uma nota objetiva:

  nota   LPIPS(pintura da LoRA, pintura aprovada do professor)  — menor é melhor
  cópia  LPIPS(pintura da LoRA, render de entrada) < 0,30        — copiou a maquete: descartado
  bordas distância de bordas entre a pintura e o render          — ruas e prédios fora do lugar:
         descartado se pior que a do melhor run em mais de 0,03
  ruído  o run base repetido com outra semente: a diferença entre as duas notas é o piso de ruído;
         um run só vira o novo melhor se bater o melhor por mais que esse piso

Fluxo (tudo local):

    tinymap lab-freeze <region> --pilot --pairs pairs_v1      # congela 32 recortes de teste
    tinymap lab-eval   <region> --pilot --name base            # nota do modelo sem LoRA
    tinymap lab-eval   <region> --pilot --name base --seed 1   # repetição: mede o piso de ruído
    tinymap lab-train  <region> --pilot --pairs pairs_v1 --name r1 --rank 32 --steps 3000
    tinymap lab-eval   <region> --pilot --name r1 --lora runs/.../r1/out/...   # cada checkpoint
    tinymap lab-report <region> --pilot                        # lab/index.html: gráfico e folhas

O teste fica em runs/<nome>/lab/testset.json e NUNCA muda depois de criado: mudar o teste invalida
a comparação com todos os runs anteriores. O treino (tinymap train / lab-train) exclui esses ids.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .region import Run

FRAME = 1024
COPY_MIN = 0.30      # LPIPS até o render abaixo disto = cópia da maquete
EDGE_SLACK = 0.03    # bordas piores que as do melhor run em mais que isto = geometria perdida
DEFAULT_NOISE = 0.0145  # piso medido no MiniDF; trocado pelo nosso assim que houver repetição do base


def lab_dir(run: Run) -> Path:
    d = run.dir / "lab"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── conjunto de teste ────────────────────────────────────────────────────────

def freeze_testset(run: Run, pairs_dirs: list[Path], n: int = 32, seed: int = 0) -> list[dict]:
    """Escolhe n pares APROVADOS (com <id>_art.png) para teste, metade de recortes de detalhe
    (caixa menor = mais zoom) e metade normais. Grava uma vez; se já existe, devolve o existente."""
    path = lab_dir(run) / "testset.json"
    if path.exists():
        return json.loads(path.read_text())
    pares = []
    for pd in pairs_dirs:
        frames = {f["id"]: f for f in json.loads((Path(pd) / "frames.json").read_text())} \
            if (Path(pd) / "frames.json").exists() else {}
        for art in sorted(Path(pd).glob("*_art.png")):
            fid = art.name[:-8]
            render = art.with_name(f"{fid}_render.png")
            if not render.exists():
                continue
            box = frames.get(fid, {}).get("box")
            size = (box[2] - box[0]) if box else Image.open(render).width
            pares.append({"id": f"{Path(pd).name}/{fid}", "art": str(art), "render": str(render), "size": size})
    if len(pares) < n:
        raise SystemExit(f"só {len(pares)} pares aprovados; o teste precisa de {n} e o treino do resto. "
                         f"Aprove mais na revisão (tinymap review) ou use --n menor.")
    rng = np.random.default_rng(seed)
    corte = np.median([p["size"] for p in pares])
    detalhe = [p for p in pares if p["size"] < corte] or pares
    normal = [p for p in pares if p["size"] >= corte]
    meia = n // 2
    escolhidos = list(rng.choice(len(detalhe), min(meia, len(detalhe)), replace=False))
    teste = [dict(detalhe[i], kind="detalhe") for i in escolhidos]
    usados = {p["id"] for p in teste}
    resto = [p for p in normal if p["id"] not in usados]
    for i in rng.choice(len(resto), min(n - len(teste), len(resto)), replace=False):
        teste.append(dict(resto[i], kind="normal"))
    path.write_text(json.dumps(teste, indent=1, ensure_ascii=False))
    print(f"teste congelado: {len(teste)} recortes ({sum(t['kind'] == 'detalhe' for t in teste)} de detalhe) "
          f"de {len(pares)} aprovados -> {path}")
    return teste


def calibrate(run: Run, lp=None) -> dict:
    """Distâncias do PRÓPRIO professor no teste: até a maquete (onde deve ficar a trava de cópia) e
    bordas (quanto o professor já mexe na geometria). Grava lab/teacher.json e avisa se a trava de
    cópia estiver acima do que o professor faz — aí ela descartaria até a pintura de referência."""
    teste = json.loads((lab_dir(run) / "testset.json").read_text())
    lp = lp or Lpips()
    cop, edg = [], []
    for t in teste:
        r = Image.open(t["render"]).convert("RGB").resize((FRAME, FRAME), Image.LANCZOS)
        a = Image.open(t["art"]).convert("RGB").resize((FRAME, FRAME), Image.LANCZOS)
        cop.append(lp(a, r))
        edg.append(edge_distance(a, r))
    info = {"copy_median": float(np.median(cop)), "copy_p10": float(np.percentile(cop, 10)),
            "edge_median": float(np.median(edg)), "copy_min": copy_min(run)}
    (lab_dir(run) / "teacher.json").write_text(json.dumps(info, indent=1))
    print(f"professor até a maquete: mediana {info['copy_median']:.3f} (10% mais perto: {info['copy_p10']:.3f}); "
          f"bordas {info['edge_median']:.3f}")
    if info["copy_p10"] < info["copy_min"]:
        print(f"AVISO: a trava de cópia ({info['copy_min']}) está acima de 10% das pinturas do professor. "
              f"Ponha `lab: {{copy_min: {round(info['copy_p10'] * 0.8, 2)}}}` na região.")
    return info


def test_ids(run: Run) -> set[str]:
    """ids 'pasta/recorte' do teste, para o treino excluir."""
    p = run.dir / "lab" / "testset.json"
    return {t["id"] for t in json.loads(p.read_text())} if p.exists() else set()


# ── métricas ─────────────────────────────────────────────────────────────────

def _edges(img: Image.Image, size: int = 512) -> np.ndarray:
    """Mapa binário das bordas mais fortes (15%), afinado; reduz o efeito de textura e dithering."""
    g = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.2)),
                   dtype=np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    return mag > np.quantile(mag, 0.85)


def edge_distance(drawn: Image.Image, render: Image.Image, tol: int = 3) -> float:
    """1 - F1 das bordas com tolerância de `tol` px (em 512 px). 0 = mesmas ruas e prédios no lugar."""
    a, b = _edges(drawn), _edges(render)
    k = 2 * tol + 1
    dil = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(k))) > 127  # noqa: E731
    da, db = dil(a), dil(b)
    prec = (a & db).sum() / max(1, a.sum())
    rec = (b & da).sum() / max(1, b.sum())
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return float(1 - f1)


class Lpips:
    """LPIPS (rede AlexNet), na GPU se houver. Carregado só quando usado."""

    def __init__(self, device: str = "cuda"):
        import lpips
        import torch
        self.torch = torch
        self.dev = device if torch.cuda.is_available() else "cpu"
        self.net = lpips.LPIPS(net="alex", verbose=False).to(self.dev).eval()

    def __call__(self, a: Image.Image, b: Image.Image, size: int = 512) -> float:
        t = self.torch
        def tens(im):
            x = np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float32) / 127.5 - 1
            return t.from_numpy(x).permute(2, 0, 1)[None].to(self.dev)
        with t.no_grad():
            return float(self.net(tens(a), tens(b)).item())


def score_frame(out: Image.Image, ref: Image.Image, render: Image.Image, lp) -> dict:
    return {"score": lp(out, ref), "copy": lp(out, render), "edge": edge_distance(out, render)}


def summarize(per: list[dict]) -> dict:
    keys = ("score", "copy", "edge")
    s = {k: float(np.mean([p[k] for p in per])) for k in keys}
    for kind in ("detalhe", "normal"):
        sub = [p["score"] for p in per if p.get("kind") == kind]
        if sub:
            s[f"score_{kind}"] = float(np.mean(sub))
    return s


# ── decisão e registro ───────────────────────────────────────────────────────

def load_runs(run: Run) -> list[dict]:
    p = lab_dir(run) / "runs.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def noise_floor(runs: list[dict]) -> tuple[float, str]:
    """Diferença entre repetições do mesmo run (mesmo nome, sementes diferentes). Fica com a maior."""
    por_nome: dict[str, list[float]] = {}
    for r in runs:
        por_nome.setdefault(r["name"], []).append(r["summary"]["score"])
    difs = [max(v) - min(v) for v in por_nome.values() if len(v) >= 2]
    return (max(difs), "medido") if difs else (DEFAULT_NOISE, "padrão do MiniDF (repita o base para medir)")


def copy_min(run: Run) -> float:
    """Trava de cópia: region lab.copy_min, senão 0,30 (MiniDF)."""
    return float(run.region.get("lab", {}).get("copy_min", COPY_MIN))


def decide(summary: dict, previous: list[dict], noise: float, cmin: float = COPY_MIN) -> tuple[str, str]:
    """('descartado'|'melhor'|'empate'|'pior', motivo) contra os runs válidos anteriores."""
    if summary["copy"] < cmin:
        return "descartado", f"copiou a maquete (distância {summary['copy']:.3f} < {cmin})"
    validos = [r for r in previous if r["status"] not in ("descartado", "repetição")]
    if not validos:
        return "melhor", "primeiro run válido"
    best = min(validos, key=lambda r: r["summary"]["score"])
    if summary["edge"] > best["summary"]["edge"] + EDGE_SLACK:
        return "descartado", (f"bordas {summary['edge']:.3f} piores que as do melhor ({best['name']}, "
                              f"{best['summary']['edge']:.3f}) em mais de {EDGE_SLACK}")
    ganho = best["summary"]["score"] - summary["score"]
    if ganho > noise:
        return "melhor", f"bate {best['name']} por {ganho:.4f} (ruído {noise:.4f})"
    if ganho > -noise:
        return "empate", f"dentro do ruído de {best['name']} ({ganho:+.4f})"
    return "pior", f"{-ganho:.4f} pior que {best['name']}"


def record(run: Run, name: str, lora: str | None, config: dict, per: list[dict], seed: int) -> dict:
    runs = load_runs(run)
    summary = summarize(per)
    noise, _ = noise_floor(runs + [{"name": name, "summary": summary}])
    if any(r["name"] == name for r in runs):
        # repetição (outra semente): serve só para medir o piso de ruído, não compete
        status, motivo = "repetição", f"mede o ruído de {name}: piso agora {noise:.4f}"
    else:
        status, motivo = decide(summary, runs, noise, copy_min(run))
    entry = {"name": name, "lora": lora, "seed": seed, "t": time.strftime("%Y-%m-%d %H:%M"), "config": config,
             "summary": summary, "status": status, "why": motivo, "per": per}
    with open(lab_dir(run) / "runs.jsonl", "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"{name}: nota {summary['score']:.4f}  cópia {summary['copy']:.3f}  bordas {summary['edge']:.3f}"
          f"  -> {status.upper()} ({motivo})")
    return entry


# ── avaliação ────────────────────────────────────────────────────────────────

def evaluate(run: Run, name: str, paint, lora: str | None = None, config: dict | None = None,
             seed: int = 0, lp=None) -> dict:
    """paint(render_1024, seed) -> imagem. Pinta os recortes de teste, mede e registra o run."""
    teste = json.loads((lab_dir(run) / "testset.json").read_text())
    out_dir = lab_dir(run) / "out" / (f"{name}_s{seed}" if seed else name)
    out_dir.mkdir(parents=True, exist_ok=True)
    lp = lp or Lpips()
    per = []
    for i, t in enumerate(teste):
        render = Image.open(t["render"]).convert("RGB").resize((FRAME, FRAME), Image.LANCZOS)
        ref = Image.open(t["art"]).convert("RGB").resize((FRAME, FRAME), Image.LANCZOS)
        fn = out_dir / (t["id"].replace("/", "__") + ".png")
        if fn.exists():
            out = Image.open(fn).convert("RGB")
        else:
            t0 = time.time()
            out = paint(render, seed + i).convert("RGB").resize((FRAME, FRAME), Image.LANCZOS)
            out.save(fn)
            print(f"  [{i + 1}/{len(teste)}] {t['id']} {time.time() - t0:.0f}s", flush=True)
        per.append({"id": t["id"], "kind": t["kind"], "out": str(fn), **score_frame(out, ref, render, lp)})
    entry = record(run, name, lora, config or {}, per, seed)
    sheet(run, entry)
    return entry


def checkpoints(lora: str | None) -> list[str | None]:
    """--lora pode ser um arquivo ou a pasta de saída do ai-toolkit (avalia todos os checkpoints,
    em ordem de passo). None = modelo base."""
    if not lora:
        return [None]
    p = Path(lora)
    if p.is_dir():
        return [str(f) for f in sorted(p.rglob("*.safetensors"), key=lambda f: (len(f.stem), f.stem))]
    return [str(p)]


class LocalPainter:
    """QwenEdit carregado UMA vez na 3090 (NF4); troca a LoRA entre checkpoints sem recarregar o
    modelo. Usa as mesmas configurações da geração do mapa (prompt, passos, true_cfg, negativo)."""

    def __init__(self, run: Run, model_id: str = "Qwen/Qwen-Image-Edit-2511", quant: str = "nf4", gpu: int = 0):
        from .models import ModelCfg, QwenEdit
        self.m = QwenEdit(ModelCfg(model_id=model_id, quant=quant, device=f"cuda:{gpu}"))
        self.st, self.g = run.region["style"], run.region["generate"]
        self.base = {"model": model_id, "quant": quant, "steps": self.g["steps"], "true_cfg": self.g.get("true_cfg", 1.0)}
        self.lora: str | None = None

    def use(self, lora: str | None, scale: float = 1.0) -> dict:
        p = self.m.p
        if self.lora:
            p.delete_adapters(["style"])
        if lora:
            p.load_lora_weights(lora, adapter_name="style")
            p.set_adapters(["style"], adapter_weights=[scale])
        self.lora = lora
        return {**self.base, "lora_scale": scale if lora else None}

    def __call__(self, img: Image.Image, seed: int) -> Image.Image:
        return self.m.edit(img, self.st["prompt"], steps=self.g["steps"], true_cfg=self.g.get("true_cfg", 1.0),
                           negative=self.st.get("negative"), seed=seed)


# ── folhas e relatório ───────────────────────────────────────────────────────

def sheet(run: Run, entry: dict, thumb: int = 256) -> Path:
    """Uma linha por recorte: render | professor | este run, com a nota."""
    teste = {t["id"]: t for t in json.loads((lab_dir(run) / "testset.json").read_text())}
    per = entry["per"]
    cols = 2
    W, H = cols * 3 * thumb, ((len(per) + cols - 1) // cols) * (thumb + 18)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for k, p in enumerate(per):
        x, y = (k % cols) * 3 * thumb, (k // cols) * (thumb + 18)
        t = teste[p["id"]]
        for j, src in enumerate((t["render"], t["art"], p["out"])):
            img.paste(Image.open(src).convert("RGB").resize((thumb, thumb)), (x + j * thumb, y + 18))
        d.text((x + 4, y + 2), f"{p['id'][-28:]}  nota {p['score']:.3f}  cópia {p['copy']:.2f}  bordas {p['edge']:.2f}",
               fill="black")
    out = lab_dir(run) / "sheets" / f"{entry['name']}_s{entry['seed']}.jpg"
    out.parent.mkdir(exist_ok=True)
    img.save(out, quality=85)
    return out


def report(run: Run) -> Path:
    """lab/index.html: progresso da nota por run, melhor, descartados e as folhas de teste."""
    runs = load_runs(run)
    noise, origem = noise_floor(runs)
    tj = lab_dir(run) / "teacher.json"
    prof = json.loads(tj.read_text()) if tj.exists() else None
    runs = runs  # repetições aparecem na tabela e no gráfico em cinza-claro
    W, H, P = 760, 260, 40
    pts = [(i, r["summary"]["score"], r) for i, r in enumerate(runs)]
    svg = ""
    if pts:
        ys = [p[1] for p in pts]
        lo, hi = min(ys) - 0.01, max(ys) + 0.01
        X = lambda i: P + (W - 2 * P) * (i / max(1, len(pts) - 1))  # noqa: E731
        Y = lambda v: H - P - (H - 2 * P) * (v - lo) / max(1e-9, hi - lo)  # noqa: E731
        best, linha = 9.0, []
        for i, v, r in pts:
            if r["status"] not in ("descartado", "repetição"):
                best = min(best, v)
            linha.append(f"{X(i):.1f},{Y(best if best < 9 else v):.1f}")
        svg += f'<polyline points="{" ".join(linha)}" fill="none" stroke="#2a9d8f" stroke-width="2"/>'
        for i, v, r in pts:
            cor = {"descartado": "#bbb", "repetição": "#ddd", "melhor": "#e76f51"}.get(r["status"], "#264653")
            svg += (f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="5" fill="{cor}">'
                    f'<title>{html.escape(r["name"])}: {v:.4f} ({r["status"]})</title></circle>')
        svg += (f'<text x="4" y="{Y(hi) + 4:.0f}" font-size="11">{hi:.3f}</text>'
                f'<text x="4" y="{Y(lo):.0f}" font-size="11">{lo:.3f}</text>')
    linhas = "".join(
        f"<tr class='{r['status']}'><td>{html.escape(r['name'])}</td><td>{r['seed']}</td>"
        f"<td>{r['summary']['score']:.4f}</td><td>{r['summary'].get('score_detalhe', 0):.4f}</td>"
        f"<td>{r['summary'].get('score_normal', 0):.4f}</td><td>{r['summary']['copy']:.3f}</td>"
        f"<td>{r['summary']['edge']:.3f}</td><td>{r['status']}</td><td>{html.escape(r['why'])}</td>"
        f"<td><a href='sheets/{html.escape(r['name'])}_s{r['seed']}.jpg'>folha</a></td>"
        f"<td><code>{html.escape(json.dumps(r['config'], ensure_ascii=False))}</code></td></tr>"
        for r in runs)
    page = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Laboratório de LoRA</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:14px system-ui;margin:16px;max-width:1100px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}}
tr.descartado{{color:#999}}tr.melhor{{font-weight:600}}code{{font-size:11px}}svg{{max-width:100%;height:auto}}
.wrap{{overflow-x:auto}}</style>
<h1>Laboratório de LoRA — {html.escape(run.name)}</h1>
<p>Nota = LPIPS até a pintura aprovada do professor nos {len(json.loads((lab_dir(run) / 'testset.json').read_text())) if (lab_dir(run) / 'testset.json').exists() else 0} recortes de teste (menor é melhor).
Descartado se copiar a maquete (cópia &lt; {copy_min(run)}) ou perder a geometria (bordas &gt; melhor + {EDGE_SLACK}).
Piso de ruído: <b>{noise:.4f}</b> ({origem}).
{f"Professor até a maquete: {prof['copy_median']:.3f} (mediana); bordas do professor {prof['edge_median']:.3f}." if prof else "Rode lab-freeze de novo para calibrar a trava de cópia."}</p>
<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Nota por run">{svg}</svg>
<p><span style="color:#e76f51">●</span> novo melhor <span style="color:#264653">●</span> válido
<span style="color:#bbb">●</span> descartado — linha: melhor até ali</p>
<div class="wrap"><table><tr><th>run</th><th>seed</th><th>nota</th><th>detalhe</th><th>normal</th><th>cópia</th>
<th>bordas</th><th>estado</th><th>por quê</th><th></th><th>config</th></tr>{linhas}</table></div></html>"""
    out = lab_dir(run) / "index.html"
    out.write_text(page)
    return out
