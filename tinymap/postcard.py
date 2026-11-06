"""Cartões isométricos (1200x630) de uma loja, imóvel ou bairro, para og:image e redes sociais.

Cada página do now.services ganha uma imagem que nenhum concorrente tem: o próprio quarteirão
desenhado em pixel art, com o pino. Compartilhada no WhatsApp, a prévia já é o mapa.

    tinymap postcard  <region.yaml> --at -25.5097,-54.6110 --name "Microcentro"
    tinymap postcards <region.yaml> --pinos https://now.services/api/mapa/pinos.json
        -> site/postais/<handle>.jpg  (servido em https://map.now.services/postais/<handle>.jpg)

Imóvel (precisão APROXIMADA) sai sem pino, com um contorno da célula: o ponto exato nunca aparece.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import geo
from .region import ROOT, Run, art_canvas

W, H = 1200, 630
ZOOM = 1  # 1 px do mapa = 1 px do cartão: ~350 m de largura a 0,293 m/px


def _font(size: int):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make(run: Run, lat: float, lon: float, name: str, precise: bool = True, out: Path | None = None) -> Path:
    aff = geo.load(run)
    x, y = geo.to_px(aff, lat, lon)
    cw, ch = W // ZOOM, H // ZOOM
    box = (int(x - cw / 2), int(y - ch * 0.55), int(x - cw / 2) + cw, int(y - ch * 0.55) + ch)
    img = art_canvas(run).crop(box).resize((W, H), Image.NEAREST)
    d = ImageDraw.Draw(img)
    px, py = (x - box[0]) * ZOOM, (y - box[1]) * ZOOM
    if precise:
        # pino em pixel art: losango amarelo com borda preta e sombra
        s = 14
        d.polygon([(px, py - 3 * s), (px + s, py - 2 * s), (px, py - s), (px - s, py - 2 * s)], fill="#ffd34e", outline="black", width=3)
        d.line([(px, py - s), (px, py)], fill="black", width=4)
        d.ellipse([px - 8, py - 3, px + 8, py + 3], fill=(0, 0, 0, 90))
    else:
        r = 150 / run.view["meters_per_px"] / 2 * ZOOM  # célula de ~150 m
        d.rectangle([px - r, py - r / 2, px + r, py + r / 2], outline="#ffd34e", width=4)
    # faixa com o nome e a marca
    band = 84
    d.rectangle([0, H - band, W, H], fill=(14, 26, 34))
    d.text((32, H - band + 18), name[:48], font=_font(40), fill="#f3ead2")
    mark = "map.now.services"
    f = _font(26)
    d.text((W - 32 - d.textlength(mark, font=f), H - band + 30), mark, font=f, fill="#ffd34e")
    out = out or run.path("postais", "".join(c if c.isalnum() else "-" for c in name.lower())[:60] + ".jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, quality=86, optimize=True)
    return out


def batch(run: Run, pinos_url: str) -> None:
    with urllib.request.urlopen(pinos_url, timeout=60) as r:
        pinos = json.load(r)["pinos"]
    out_dir = ROOT / "site" / "postais"
    for p in pinos:
        path = make(run, p["lat"], p["lon"], p["nome"], precise=p["precisao"] == "EXATA", out=out_dir / f"{p['handle']}.jpg")
        print(path)
    print(f"{len(pinos)} cartões em {out_dir}")
