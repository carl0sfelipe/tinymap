"""Polimento determinístico da arte: sem GPU, sem modelo, mesma entrada → mesma saída.

O modelo de imagem entrega "quase pixel art": pixels de 2-4 px com bordas moles, ruído do VAE, cores
que derivam de um bloco para o outro. Aqui ficam as correções que não precisam de IA:

  grade    descobre o tamanho real do pixel desenhado (e o deslocamento da grade) e encaixa cada célula
           numa cor só: bordas duras de novo, ruído some.
  paleta   uma paleta só para o mapa inteiro (tirada da própria arte); cada pixel vai para a cor mais
           próxima. Junta tons quase iguais e dá a cara de "jogo".
  emendas  acha as divisas entre blocos de geração onde a cor muda de repente e corrige o lado de lá
           com uma transferência de cor que some aos poucos (sem tocar no desenho).
  borrado  mede quanto da arte ainda é gradiente mole (candidato a refazer).

    tinymap polish regions/demo.yaml --grid --palette --seams [--dry-run]

Tudo grava pelo histórico da oficina: cada shot alterado pode ser desfeito.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

# ── grade de pixel ───────────────────────────────────────────────────────────


def _cells(a: np.ndarray, k: int, ox: int, oy: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Recorta a imagem na grade (k, ox, oy) e devolve as células (n_y, n_x, k*k, C)."""
    h, w = a.shape[:2]
    ny, nx = (h - oy) // k, (w - ox) // k
    b = a[oy:oy + ny * k, ox:ox + nx * k]
    return b.reshape(ny, k, nx, k, -1).transpose(0, 2, 1, 3, 4).reshape(ny, nx, k * k, -1), (ny, nx)


def grid_error(a: np.ndarray, k: int, ox: int, oy: int) -> float:
    """Quanto a imagem difere de 'cada célula k×k tem uma cor só' (desvio médio dentro das células)."""
    c, _ = _cells(a, k, ox, oy)
    return float(np.abs(c - np.median(c, axis=2, keepdims=True)).mean())


def _fold(profile: np.ndarray, k: int) -> np.ndarray:
    n = len(profile) // k * k
    return profile[:n].reshape(-1, k).mean(0)


def detect_grid(img: Image.Image, kmax: int = 8, sample: int = 768) -> dict:
    """Tamanho do pixel desenhado e deslocamento da grade.

    Numa pixel art ampliada k vezes, as bordas só acontecem entre células: a força das bordas somada
    por coluna (e por linha) é periódica com período k, com pico no deslocamento da grade. O borrado
    do modelo espalha o pico mas não muda o período. Para cada k, dobra o perfil em k posições e mede
    o pico relativo; escolhe o k mais forte (empate → o menor, porque 6 também "vê" a grade de 3).
    k=1 = não há grade (arte já nítida, ou foto)."""
    g = np.asarray(img.convert("L"), np.float32)
    h, w = g.shape
    y0, x0 = max(0, h // 2 - sample // 2), max(0, w // 2 - sample // 2)
    g = g[y0:y0 + sample, x0:x0 + sample]
    # só bordas de verdade (> 8 níveis): o ruído do VAE espalha energia em todas as posições e apaga o período
    dx, dy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    px = np.where(dx > 8, dx, 0).sum(0)   # borda entre a coluna x e x+1
    py = np.where(dy > 8, dy, 0).sum(1)
    scores = {}
    for k in range(2, kmax + 1):
        fx, fy = _fold(px, k), _fold(py, k)
        sx = (fx.max() - fx.mean()) / (fx.mean() + 1e-6)
        sy = (fy.max() - fy.mean()) / (fy.mean() + 1e-6)
        # as duas direções precisam concordar: textura listrada numa direção só não é grade
        scores[k] = (min(sx, sy), int((fx.argmax() + 1 + x0) % k), int((fy.argmax() + 1 + y0) % k))
    kbest = max(scores, key=lambda k: scores[k][0])
    top = scores[kbest][0]
    k = min(kk for kk in scores if scores[kk][0] >= 0.9 * top and kbest % kk == 0)
    if top < 0.25:
        return {"k": 1, "ox": 0, "oy": 0, "contrast": round(float(top), 3)}
    # o pico do perfil pode cair um pixel ao lado com o borrado; com k conhecido, o deslocamento certo
    # é o que deixa as células mais uniformes (erro de célula mínimo)
    gg = g[..., None]
    ox, oy = min(((a, b) for b in range(k) for a in range(k)), key=lambda o: grid_error(gg, k, *o))
    return {"k": k, "ox": int((ox + x0) % k), "oy": int((oy + y0) % k), "contrast": round(float(scores[k][0]), 3)}


def snap_grid(img: Image.Image, k: int, ox: int = 0, oy: int = 0) -> Image.Image:
    """Cada célula k×k vira a cor mediana dela (bordas duras, sem ruído). As sobras nas bordas da
    imagem (menores que uma célula) usam a célula vizinha."""
    if k <= 1:
        return img.convert("RGB")
    a = np.asarray(img.convert("RGB"), np.float32)
    h, w = a.shape[:2]
    # estende para cobrir as sobras antes de ox/oy e depois da última célula
    pad_t, pad_l = (k - oy) % k, (k - ox) % k
    ph = -(-(h + pad_t) // k) * k
    pw = -(-(w + pad_l) // k) * k
    big = np.pad(a, ((pad_t, ph - h - pad_t), (pad_l, pw - w - pad_l), (0, 0)), mode="edge")
    c, (ny, nx) = _cells(big, k, 0, 0)
    med = np.median(c, axis=2)
    out = np.repeat(np.repeat(med, k, axis=0), k, axis=1)[pad_t:pad_t + h, pad_l:pad_l + w]
    return Image.fromarray(out.round().clip(0, 255).astype(np.uint8))


# ── paleta ───────────────────────────────────────────────────────────────────


def build_palette(images: list[Image.Image], n: int = 64, thumb: int = 384) -> list[tuple[int, int, int]]:
    """Paleta do mapa: corte mediano sobre um mosaico das miniaturas já encaixadas na grade (sem o
    ruído do modelo, que gasta cores da paleta à toa). Determinístico."""
    if not images:
        return []
    tiles = []
    for im in images:
        im = im.convert("RGB")
        gd = detect_grid(im)
        tiles.append(snap_grid(im, gd["k"], gd["ox"], gd["oy"]).resize((thumb, thumb), Image.NEAREST))
    cols = int(np.ceil(np.sqrt(len(tiles))))
    mos = Image.new("RGB", (cols * thumb, int(np.ceil(len(tiles) / cols)) * thumb))
    for i, t in enumerate(tiles):
        mos.paste(t, ((i % cols) * thumb, (i // cols) * thumb))
    q = mos.quantize(n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = q.getpalette()[: 3 * n]
    used = sorted(set(np.asarray(q).ravel().tolist()))
    return [tuple(pal[3 * i: 3 * i + 3]) for i in used]


def apply_palette(img: Image.Image, palette: list[tuple[int, int, int]], max_dist: float = 28.0,
                  max_channel: int = 18) -> Image.Image:
    """Cada pixel vai para a cor mais próxima da paleta (distância com peso perceptivo 3-4-2), sem
    pontilhado: o pontilhado que já está desenhado continua, ruído novo não aparece.
    Cor sem vizinha próxima na paleta (> max_dist, ou algum canal mudando mais que max_channel) fica
    como está: um telhado azul raro não vira verde só porque a paleta não o tem."""
    if not palette:
        return img.convert("RGB")
    a = np.asarray(img.convert("RGB"), np.int32)
    flat = a.reshape(-1, 3)
    p = np.asarray(palette, np.int32)
    wgt = np.array([3, 4, 2], np.int32)
    # memo por cor única: imagens de mapa têm poucas cores distintas depois da grade
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    mapped = uniq.copy()
    for i in range(0, len(uniq), 4096):
        u = uniq[i:i + 4096]
        d = (((u[:, None, :] - p[None, :, :]) ** 2) * wgt).sum(-1)
        j = d.argmin(1)
        near = (np.sqrt(d[np.arange(len(u)), j] / wgt.sum()) <= max_dist) & \
            (np.abs(u - p[j]).max(1) <= max_channel)
        mapped[i:i + 4096][near] = p[j[near]]
    out = mapped[inv.ravel()].reshape(a.shape)
    return Image.fromarray(out.astype(np.uint8))


def palette_spread(img: Image.Image) -> int:
    """Número de cores distintas (em 5 bits por canal): cai muito depois de grade + paleta."""
    a = np.asarray(img.convert("RGB")) >> 3
    return int(len(np.unique(a.reshape(-1, 3), axis=0)))


# ── emendas entre blocos ─────────────────────────────────────────────────────


def _stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = a.reshape(-1, 3).astype(np.float32)
    return f.mean(0), f.std(0) + 1e-3


def seam_score(left: np.ndarray, right: np.ndarray) -> float:
    """Diferença de cor entre duas faixas encostadas (média por canal, peso perceptivo)."""
    ml, _ = _stats(left)
    mr, _ = _stats(right)
    return float(np.sqrt((((ml - mr) ** 2) * np.array([0.3, 0.59, 0.11]) * 3).sum()))


def find_seams(canvas_crop, size: tuple[int, int], block: int, band: int = 48, seg: int = 256,
               thr: float = 6.0, has_art=None, render_crop=None) -> list[dict]:
    """Percorre as divisas verticais e horizontais dos blocos de geração em segmentos de `seg` px e
    devolve os trechos onde a cor muda de repente (score > thr). canvas_crop(box) -> Image.
    Com render_crop, desconta a diferença que a própria maquete já tem ali (borda de rio, de parque):
    só conta como emenda o salto que a arte tem a mais."""
    def score(box, split, axis):
        a = np.asarray(canvas_crop(box).convert("RGB"), np.float32)
        sa = seam_score(a[:, :split], a[:, split:]) if axis == "v" else seam_score(a[:split], a[split:])
        if render_crop is None:
            return sa
        r = np.asarray(render_crop(box).convert("RGB"), np.float32)
        sr = seam_score(r[:, :split], r[:, split:]) if axis == "v" else seam_score(r[:split], r[split:])
        return max(0.0, sa - sr)
    W, H = size
    out = []
    for x in range(block, W, block):
        for y in range(0, H, seg):
            y1 = min(H, y + seg)
            if has_art and not (has_art(x - 1, y) and has_art(x, y)):
                continue
            s = score((x - band, y, x + band, y1), band, "v")
            if s > thr:
                out.append({"kind": "emenda", "axis": "v", "at": x, "score": round(s, 1),
                            "box": [x - band, y, x + band, y1]})
    for y in range(block, H, block):
        for x in range(0, W, seg):
            x1 = min(W, x + seg)
            if has_art and not (has_art(x, y - 1) and has_art(x, y)):
                continue
            s = score((x, y - band, x1, y + band), band, "h")
            if s > thr:
                out.append({"kind": "emenda", "axis": "h", "at": y, "score": round(s, 1),
                            "box": [x, y - band, x1, y + band]})
    return out


def fix_seam(img: Image.Image, axis: str, at: int, width: int = 384, band: int = 48, taper: int = 0) -> Image.Image:
    """Corrige uma divisa dentro de `img` (coordenadas locais): o lado depois da divisa (direita ou
    baixo) recebe a transferência de cor (média e desvio por canal) que o deixa igual ao lado antes,
    com força 1 na divisa caindo a 0 em `width` px. Nada muda de lugar; só a cor.
    taper: a força também cai a 0 nas duas pontas do trecho, em `taper` px (sem quina de retângulo)."""
    a = np.asarray(img.convert("RGB"), np.float32).copy()
    if axis == "h":
        a = a.transpose(1, 0, 2)
    ref_m, ref_s = _stats(a[:, max(0, at - band):at])
    cur_m, cur_s = _stats(a[:, at:at + band])
    end = min(a.shape[1], at + width)
    t = 1 - (np.arange(end - at, dtype=np.float32) / max(1, end - at))  # 1 na divisa → 0
    t = (t * t * (3 - 2 * t))[None, :, None]  # suave (smoothstep)
    if taper:
        n = a.shape[0]
        u = np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n)) / taper, 0, 1).astype(np.float32)
        t = t * (u * u * (3 - 2 * u))[:, None, None]
    seg = a[:, at:end]
    moved = (seg - cur_m) / cur_s * ref_s + ref_m
    a[:, at:end] = seg + (moved - seg) * t
    if axis == "h":
        a = a.transpose(1, 0, 2)
    return Image.fromarray(a.round().clip(0, 255).astype(np.uint8))


def block_transforms(crop, size: tuple[int, int], block: int, band: int = 64,
                     has_art=None) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Ganho e deslocamento de cor por bloco de geração que deixam TODAS as divisas iguais ao mesmo
    tempo (mínimos quadrados, a média dos blocos fica onde está). É a correção certa para a causa das
    emendas: cada bloco sai do modelo com uma deriva de cor própria.
    Devolve {(bx, by): (ganho[3], deslocamento[3])}; aplicar: novo = ganho * cor + deslocamento."""
    W, H = size
    nbx, nby = -(-W // block), -(-H // block)
    idx = {(bx, by): i for i, (bx, by) in enumerate((x, y) for y in range(nby) for x in range(nbx))}
    pairs = []  # (bloco a, bloco b, média a, média b, desvio a, desvio b) ao longo da divisa
    for by in range(nby):
        for bx in range(nbx):
            for nx, ny, axis in ((bx + 1, by, "v"), (bx, by + 1, "h")):
                if nx >= nbx or ny >= nby:
                    continue
                if axis == "v":
                    x = nx * block
                    y0, y1 = by * block, min(H, (by + 1) * block)
                    if has_art and not (has_art(x - 1, (y0 + y1) // 2) and has_art(x, (y0 + y1) // 2)):
                        continue
                    a = np.asarray(crop((x - band, y0, x + band, y1)).convert("RGB"), np.float32)
                    A, B = a[:, :band], a[:, band:]
                else:
                    y = ny * block
                    x0, x1 = bx * block, min(W, (bx + 1) * block)
                    if has_art and not (has_art((x0 + x1) // 2, y - 1) and has_art((x0 + x1) // 2, y)):
                        continue
                    a = np.asarray(crop((x0, y - band, x1, y + band)).convert("RGB"), np.float32)
                    A, B = a[:band], a[band:]
                (ma, sa), (mb, sb) = _stats(A), _stats(B)
                pairs.append((idx[(bx, by)], idx[(nx, ny)], ma, mb, sa, sb))
    n = len(idx)
    gains = np.ones((n, 3))
    offs = np.zeros((n, 3))
    if not pairs:
        return {k: (gains[i], offs[i]) for k, i in idx.items()}
    for ch in range(3):
        # 1) log-ganho: log g_a + log s_a = log g_b + log s_b, com soma dos log-ganhos = 0
        M = np.zeros((len(pairs) + 1, n))
        y = np.zeros(len(pairs) + 1)
        for r, (i, j, ma, mb, sa, sb) in enumerate(pairs):
            M[r, i], M[r, j] = 1, -1
            y[r] = np.log(sb[ch]) - np.log(sa[ch])
        M[-1, :] = 1
        lg = np.linalg.lstsq(M, y, rcond=None)[0]
        g = np.exp(np.clip(lg, -0.3, 0.3))
        # 2) deslocamento: g_a m_a + o_a = g_b m_b + o_b, com soma = 0
        y2 = np.zeros(len(pairs) + 1)
        for r, (i, j, ma, mb, sa, sb) in enumerate(pairs):
            y2[r] = g[j] * mb[ch] - g[i] * ma[ch]
        o = np.linalg.lstsq(M, y2, rcond=None)[0]
        gains[:, ch], offs[:, ch] = g, o
    return {k: (gains[i], offs[i]) for k, i in idx.items()}


def block_transforms_by_render(art_crop, render_crop, has_art, size: tuple[int, int], block: int,
                               thumb: int = 256) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Deriva de cor de cada bloco medida contra a maquete, não contra o vizinho.

    A maquete é a mesma em todo o mapa; o que muda de bloco para bloco é como o modelo a traduziu em
    cor. Para cada bloco, ajusta arte ≈ G_b·maquete + O_b por canal (mínimos quadrados em todos os
    pixels com arte); o mapa inteiro dá a relação média G, O. A correção leva cada bloco para a média:
        novo = (G/G_b)·arte + (O − (G/G_b)·O_b)
    Usa o bloco todo, então não se engana com o que existe de cada lado da divisa (rio, parque)."""
    W, H = size
    stats = {}
    allx, ally = [], []
    for by in range(-(-H // block)):
        for bx in range(-(-W // block)):
            box = (bx * block, by * block, min(W, (bx + 1) * block), min(H, (by + 1) * block))
            tw = max(16, thumb * (box[2] - box[0]) // block)
            th = max(16, thumb * (box[3] - box[1]) // block)
            art = art_crop(box).resize((tw, th), Image.BOX)
            a = np.asarray(art, np.float32)
            mask = a[..., 3] > 250 if a.shape[-1] == 4 else np.ones(a.shape[:2], bool)
            if mask.mean() < 0.2:
                continue
            r = np.asarray(render_crop(box).convert("RGB").resize((tw, th), Image.BOX), np.float32)[mask]
            a = a[..., :3][mask]
            stats[(bx, by)] = (r, a)
            allx.append(r)
            ally.append(a)
    if not stats:
        return {}

    def fit(x, y):
        g, o = np.ones(3), np.zeros(3)
        for ch in range(3):
            A = np.stack([x[:, ch], np.ones(len(x))], 1)
            g[ch], o[ch] = np.linalg.lstsq(A, y[:, ch], rcond=None)[0]
        return g, o
    G, O = fit(np.concatenate(allx), np.concatenate(ally))
    out = {}
    for k, (x, y) in stats.items():
        gb, ob = fit(x, y)
        ratio = np.clip(G / np.where(np.abs(gb) < 1e-3, 1e-3, gb), 0.75, 1.33)
        out[k] = (ratio, O - ratio * ob)
    return out


def apply_transform(img: Image.Image, gain: np.ndarray, off: np.ndarray) -> Image.Image:
    a = np.asarray(img.convert("RGB"), np.float32) * gain + off
    return Image.fromarray(a.round().clip(0, 255).astype(np.uint8))


# ── borrado ──────────────────────────────────────────────────────────────────


def softness(img: Image.Image) -> float:
    """Fração de vizinhos com diferença pequena mas não nula (3-12 níveis): gradiente mole.
    Pixel art limpa fica perto de 0; foto ou arte borrada passa de 0,3."""
    a = np.asarray(img.convert("L"), np.int16)
    d = np.concatenate([np.abs(np.diff(a, axis=0)).ravel(), np.abs(np.diff(a, axis=1)).ravel()])
    return float(((d >= 3) & (d <= 12)).mean())


# ── recorte da oficina e mapa inteiro ────────────────────────────────────────


def polish_image(img: Image.Image, ops: list[str], palette=None, k: int | None = None) -> tuple[Image.Image, dict]:
    """Aplica as operações num recorte (RGB). Devolve a imagem e o que foi feito."""
    info: dict = {"antes": {"cores": palette_spread(img), "mole": round(softness(img), 3)}}
    out = img.convert("RGB")
    if "grid" in ops:
        gd = detect_grid(out) if not k else {"k": k, "ox": 0, "oy": 0, "contrast": None}
        if k:  # escala dada à mão: ainda acha o deslocamento certo
            g = np.asarray(out.convert("L"), np.float32)[..., None]
            ox, oy = min(((x, y) for y in range(k) for x in range(k)), key=lambda o: grid_error(g, k, *o))
            gd.update(ox=ox, oy=oy)
        out = snap_grid(out, gd["k"], gd["ox"], gd["oy"])
        info["grade"] = gd
    if "palette" in ops and palette:
        out = apply_palette(out, palette)
        info["paleta"] = len(palette)
    info["depois"] = {"cores": palette_spread(out), "mole": round(softness(out), 3)}
    return out, info


def map_palette(run, n: int = 64) -> list[tuple[int, int, int]]:
    """Paleta do mapa, guardada em runs/<nome>/palette.json (apague para recalcular)."""
    p = run.dir / "palette.json"
    if p.exists():
        return [tuple(c) for c in json.loads(p.read_text())]
    shots = sorted((run.dir / "art").glob("r*_c*.png"))[::2] or sorted((run.dir / "art").glob("r*_c*.png"))
    pal = build_palette([Image.open(s) for s in shots[:64]], n)
    p.write_text(json.dumps(pal))
    return pal


def run_all(run, region_file: str, grid: bool, palette: bool, seams: bool, dry_run: bool = False,
            only: list[str] | None = None) -> dict:
    """Mapa inteiro, shot a shot, pelo histórico da oficina (desfazível)."""
    from .mapedit import MapEdit
    ed = MapEdit(run, region_file)
    s = run.view["shot_px"]
    pal = map_palette(run) if palette else None
    tot: dict = {"shots": 0, "emendas": 0, "ids": []}
    ops = [o for o, on in (("grid", grid), ("palette", palette)) if on]
    if ops:
        for p in sorted((run.dir / "art").glob("r*_c*.png")):
            if only and p.stem not in only:
                continue
            r, c = (int(v[1:]) for v in p.stem.split("_"))
            box = (c * s, r * s, (c + 1) * s, (r + 1) * s)
            im = Image.open(p).convert("RGBA")
            alpha = im.getchannel("A")
            new, info = polish_image(im.convert("RGB"), ops, pal)
            print(f"{p.stem}: cores {info['antes']['cores']} → {info['depois']['cores']}, "
                  f"mole {info['antes']['mole']} → {info['depois']['mole']}"
                  + (f", pixel {info['grade']['k']} px" if "grade" in info else ""), flush=True)
            tot["shots"] += 1
            if not dry_run:
                out = new.convert("RGBA")
                out.putalpha(alpha)
                b = io.BytesIO()
                out.save(b, "PNG", compress_level=1)
                tot["ids"].append(ed.apply(box, b.getvalue(), "auto: polimento (" + ", ".join(ops) + ")")["id"])
    if seams:
        # 1) ajuste de cor por bloco inteiro (resolve as derivas de uma vez, inclusive nos cantos)
        block = run.region.get("generate", {}).get("block_px", 6144)
        tot["emendas"] = len(seam_items(ed))
        tr = block_transforms_by_render(lambda b: ed.crop("art", b), lambda b: ed.crop("render", b),
                                        None, tuple(run.view["canvas_px"]), block)
        for p in sorted((run.dir / "art").glob("r*_c*.png")):
            r, c = (int(v[1:]) for v in p.stem.split("_"))
            if ((c * s) // block, (r * s) // block) not in tr:
                continue
            g_, o_ = tr[((c * s) // block, (r * s) // block)]
            if np.allclose(g_, 1, atol=0.01) and np.allclose(o_, 0, atol=1.0):
                continue
            if dry_run:
                continue
            im = Image.open(p).convert("RGBA")
            out = apply_transform(im.convert("RGB"), g_, o_).convert("RGBA")
            out.putalpha(im.getchannel("A"))
            b = io.BytesIO()
            out.save(b, "PNG", compress_level=1)
            tot["ids"].append(ed.apply((c * s, r * s, (c + 1) * s, (r + 1) * s), b.getvalue(),
                                       "auto: cor do bloco (emendas)")["id"])
        # 2) sobras locais: várias passadas: no canto de 4 blocos, corrigir a divisa horizontal mexe na vertical já
        # corrigida; repete a busca até não sobrar emenda (ou 3 passadas)
        for passada in range(3):
            # depois do ajuste por bloco, só sobra emenda local forte; as fracas (< 12) são mais
            # conteúdo do que cor, e corrigir cria mancha
            found = [it for it in seam_items(ed) if it["score"] >= 12]
            if not found or dry_run:
                for it in found:
                    print(f"emenda {it['axis']} em {it['at']} ({it['box']}): {it['score']}", flush=True)
                break
            for it in found:
                tot["ids"].append(_fix_one(ed, run, it))
            print(f"passada {passada + 1}: {len(found)} trechos de emenda corrigidos", flush=True)
        tot["emendas_restantes"] = len(seam_items(ed)) if not dry_run else tot["emendas"]
    if not dry_run:
        (run.dir / "polish_last.json").write_text(json.dumps(tot["ids"]))
    return tot


def _fix_one(ed, run, it: dict, w: int = 384, taper: int = 128) -> str:
    """Corrige um trecho de emenda pelo histórico da oficina; devolve o id para desfazer. A caixa
    passa `taper` px das pontas do trecho, e a correção some aos poucos nelas."""
    x0, y0, x1, y1 = it["box"]
    W, H = run.view["canvas_px"]
    if it["axis"] == "v":
        box = (it["at"] - 64, max(0, y0 - taper), min(W, it["at"] + w), min(H, y1 + taper))
    else:
        box = (max(0, x0 - taper), it["at"] - 64, min(W, x1 + taper), min(H, it["at"] + w))
    art = ed.crop("art", box)
    alpha = art.getchannel("A")
    fixed = fix_seam(art.convert("RGB"), it["axis"], 64, width=w, taper=taper).convert("RGBA")
    fixed.putalpha(alpha)
    b = io.BytesIO()
    fixed.save(b, "PNG", compress_level=1)
    return ed.apply(box, b.getvalue(), f"auto: emenda {it['axis']} em {it['at']}")["id"]


def undo_last(run, region_file: str) -> int:
    from .mapedit import MapEdit
    ed = MapEdit(run, region_file)
    p = run.dir / "polish_last.json"
    ids = json.loads(p.read_text()) if p.exists() else []
    for hid in reversed(ids):
        ed.undo(hid)
    p.unlink(missing_ok=True)
    return len(ids)


def seam_items(ed, thr: float = 10.0) -> list[dict]:
    """Para o caçador de defeitos da oficina. Abaixo de 10 a diferença não se vê a olho (medido na
    cidade demo depois do ajuste por bloco); o ajuste por bloco usa a medida inteira, não esta lista."""
    run = ed.run
    block = run.region.get("generate", {}).get("block_px", 6144)
    s = ed.s
    return find_seams(lambda b: ed.crop("mixed", b), tuple(run.view["canvas_px"]), block,
                      has_art=lambda x, y: (ed.art / f"r{y // s}_c{x // s}.png").exists(),
                      render_crop=lambda b: ed.crop("render", b), thr=thr)


