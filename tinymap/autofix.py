"""Conserto automático da água inventada, aprendido das edições à mão (nov/2026).

As 18 edições feitas na oficina seguiram duas regras, sempre com cor chapada:
  - água (azul) pintada em cima de uma rua da maquete  -> asfalto do mapa
  - laguinho / verde azulado no meio da grama          -> a grama em volta
Aqui as duas regras rodam no mapa todo. A maquete diz o que existe embaixo (rua, rio de verdade, piscina,
telhado azul), a arte diz onde o modelo pintou água. Cada shot alterado entra no histórico da oficina,
então dá para desfazer um por um.

    tinymap autofix regions/triplice-fronteira.yaml [--dry-run]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROAD = (108, 116, 116)     # cinza usado nas edições à mão
GRASS = (116, 204, 100)    # verde usado nas edições à mão
MIN_PX = 40                # manchas menores que isso ficam (ruído, reflexo)
LAKE_PX = 1500             # fora da rua, mancha compacta (meia-largura > 10 px) e maior que isso é lago: fica


def _blue(a: np.ndarray) -> np.ndarray:
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (b - r > 40) & (b > g - 35)            # azul e turquesa (água pintada)


def _teal(a: np.ndarray) -> np.ndarray:
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (g - r > 40) & (b - r >= 15) & ~_blue(a)  # verde azulado (borda de laguinho)


def _render_water(ren: np.ndarray) -> np.ndarray:
    r, g, b = ren[..., 0], ren[..., 1], ren[..., 2]
    return (b - r > 60) & (b - g > 15)


def _render_road(ren: np.ndarray) -> np.ndarray:
    r, g, b = ren[..., 0], ren[..., 1], ren[..., 2]
    return (np.abs(r - g) < 8) & (np.abs(g - b) < 8) & (r > 100) & (r < 150)


def _render_bluish(ren: np.ndarray) -> np.ndarray:  # piscina, telhado azul de verdade (azul forte e claro)
    r, g, b = ren[..., 0], ren[..., 1], ren[..., 2]
    # a sombra da mata no satélite é azulada (85,119,127) e não conta; piscina (100,200,220) e telhado azul contam
    return (b - r > 70) & (b > 140) & ~_render_water(ren)


def fix(art: np.ndarray, ren: np.ndarray, area: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """art, ren: HxWx3 int16. Devolve a arte consertada e estatísticas.

    Regras (validadas contra as edições à mão):
      1. mancha azul em cima de rua (>= 50% sobre a rua da maquete) -> asfalto, inteira
      2. mancha azul fora da rua, pequena (< LAKE_PX) ou fina (canal/risco, meia-largura <= 10 px) -> grama
      3. lago grande e compacto fora da rua -> fica (provavelmente existe; decisão humana)
      4. verde azulado só na borda (até 4 px) das manchas consertadas -> mesma cor do conserto
    """
    from scipy import ndimage as nd
    out = art.copy()
    real = nd.binary_dilation(_render_water(ren), iterations=12)          # rio/lago de verdade + 12 px de margem
    road = _render_road(ren)
    bluish = _render_bluish(ren)
    blue_all = _blue(art)
    blue = blue_all & ~real
    if area is not None:
        blue &= area
    teal = _teal(art) & ~real
    # junta pedaços vizinhos (lago com textura/ilhas fica quebrado em centenas de manchas) SEM tirar a água real,
    # para a decisão ser pelo conjunto: o que encosta em água real é extensão dela e fica
    cand = nd.binary_closing(blue_all | (_teal(art) & nd.binary_dilation(blue_all, iterations=4)), iterations=5)
    if area is not None:
        cand &= area
    lab, n = nd.label(cand)
    st = {"manchas": 0, "rua_px": 0, "grama_px": 0, "puladas": 0, "lagos_mantidos": 0}
    if not n:
        return out, st
    sizes = nd.sum(cand, lab, range(1, n + 1))
    objs = nd.find_objects(lab)
    H, W = art.shape[:2]
    for k, (sl, size) in enumerate(zip(objs, sizes), 1):
        if size < MIN_PX:
            continue
        ys, xs = sl
        y0, y1, x0, x1 = max(0, ys.start - 8), min(H, ys.stop + 8), max(0, xs.start - 8), min(W, xs.stop + 8)
        m = lab[y0:y1, x0:x1] == k
        if real[y0:y1, x0:x1][m].mean() > 0.05:        # encosta em rio/lago de verdade: é extensão dele, fica
            st["lagos_mantidos"] += 1
            continue
        if bluish[y0:y1, x0:x1][m].mean() > 0.3:      # embaixo tem azul de verdade (piscina, telhado): deixa
            st["puladas"] += 1
            continue
        on_road = road[y0:y1, x0:x1][m].mean()
        if on_road >= 0.5:
            color, key = ROAD, "rua_px"
        else:
            half = nd.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1].max()
            if size >= LAKE_PX and half > 10:
                st["lagos_mantidos"] += 1
                continue
            color, key = GRASS, "grama_px"
        m = m & (blue[y0:y1, x0:x1] | teal[y0:y1, x0:x1])                   # pinta só o que é água de fato
        grow = nd.binary_dilation(m, iterations=2)                          # borda clara da mancha
        halo = nd.binary_dilation(m, iterations=4) & teal[y0:y1, x0:x1]     # verde azulado colado nela
        paint = (grow | halo) & ~real[y0:y1, x0:x1]
        if area is not None:
            paint &= area[y0:y1, x0:x1]
        out[y0:y1, x0:x1][paint] = color
        st["manchas"] += 1
        st[key] += int(paint.sum())
    return out, st


def area_mask(run, r: int, c: int) -> np.ndarray:
    from PIL import ImageDraw
    s = run.view["shot_px"]
    m = Image.new("L", (s, s), 0)
    ImageDraw.Draw(m).polygon([(x - c * s, y - r * s) for x, y in run.view["area_px"]], fill=255)
    return np.asarray(m) > 127


def run_all(run, region_file: str, dry_run: bool = False, only: list[str] | None = None) -> dict:
    """Roda em todos os shots com arte. Grava pelo histórico da oficina (desfazível) se não for dry_run."""
    from .mapedit import MapEdit
    ed = MapEdit(run, region_file)
    s = run.view["shot_px"]
    tot = {"shots": 0, "manchas": 0, "rua_px": 0, "grama_px": 0, "puladas": 0, "lagos_mantidos": 0, "ids": []}
    for p in sorted((run.dir / "art").glob("r*_c*.png")):
        if only and p.stem not in only:
            continue
        r, c = (int(v[1:]) for v in p.stem.split("_"))
        box = (c * s, r * s, (c + 1) * s, (r + 1) * s)
        im = Image.open(p).convert("RGBA")
        alpha = np.asarray(im)[..., 3] > 0
        art = np.asarray(im.convert("RGB"), np.int16)
        ren = np.asarray(ed.rc.crop(box), np.int16)
        new, st = fix(art, ren, area_mask(run, r, c) & alpha)
        if not st["manchas"]:
            continue
        for k in ("manchas", "rua_px", "grama_px", "puladas", "lagos_mantidos"):
            tot[k] += st[k]
        tot["shots"] += 1
        print(f"{p.stem}: {st['manchas']} manchas · {st['rua_px']} px → rua · {st['grama_px']} px → grama", flush=True)
        if not dry_run:
            out = Image.fromarray(new.astype(np.uint8)).convert("RGBA")
            out.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
            from io import BytesIO
            b = BytesIO()
            out.save(b, "PNG", compress_level=1)
            tot["ids"].append(ed.apply(box, b.getvalue(), "auto: água inventada → rua/grama")["id"])
    if not dry_run:
        (run.dir / "autofix_last.json").write_text(json.dumps(tot["ids"]))
    return tot


def validate(run) -> None:
    """Compara a regra com as edições à mão: roda no 'antes' e mede quanto bate com o 'depois'."""
    from .region import render_canvas
    rc = render_canvas(run)
    H = run.dir / "art_history"
    for d in sorted(H.iterdir()):
        m = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        if m.get("note") != "edição à mão":
            continue
        A = np.asarray(Image.open(d / "antes.png").convert("RGBA"), np.int16)
        B = np.asarray(Image.open(d / "depois.png").convert("RGB"), np.int16)
        had = A[..., 3] > 0
        human = (np.abs(A[..., :3] - B).sum(2) > 0) & had
        if human.sum() < 100:
            continue
        ren = np.asarray(rc.crop(tuple(m["box"])), np.int16)
        auto, _ = fix(A[..., :3].copy(), ren, had)
        robot = (np.abs(A[..., :3] - auto).sum(2) > 0) & had
        inter = (human & robot).sum()
        same_col = (np.abs(auto - B).sum(2) < 30)[human & robot].mean() if inter else 0
        print(f"{d.name}: você mudou {int(human.sum())} px; a regra mudou {int(robot.sum())} px; "
              f"em comum {inter / human.sum():.0%} do seu; cor igual em {same_col:.0%}; "
              f"a regra mudou além {(robot & ~human).sum() / max(1, robot.sum()):.0%}")


def undo_last(run, region_file: str) -> int:
    """Desfaz o último lote do autofix (só os IDs que ele mesmo criou, do mais novo para o mais antigo)."""
    from .mapedit import MapEdit
    ed = MapEdit(run, region_file)
    p = run.dir / "autofix_last.json"
    ids = json.loads(p.read_text()) if p.exists() else []
    n = 0
    for hid in reversed(ids):
        if (ed.hist / hid).exists() and not (ed.hist / hid / "desfeito").exists():
            ed.undo(hid)
            n += 1
    print(f"{n} edições automáticas desfeitas")
    return n
