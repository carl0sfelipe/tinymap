"""Finished art shots -> Deep Zoom pyramid (WebP) + search index + borders for the viewer in site/.

Uses libvips (pyvips), which streams, so a 1.5 gigapixel canvas never has to fit in RAM.
"""

from __future__ import annotations

import json
import shutil

from .region import ROOT, Run


BG = (14, 26, 34)  # fundo do viewer (site/index.html --bg)


def _masked(run: Run):
    """O que o site mostra em cada shot planejado: o render (maquete + satélite) por baixo, a arte por
    cima (com transparência onde a oficina criou arte parcial), recortado pelo polígono da área do mapa
    (fora dele o modelo inventava cidade e canais). Grava em art_site/ e devolve a pasta."""
    from PIL import Image, ImageDraw
    v = run.view
    s = v["shot_px"]
    out = run.dir / "art_site"
    out.mkdir(exist_ok=True)
    poly = [tuple(p) for p in v["area_px"]]
    for c, r in v["shots"]:
        ren, art = run.dir / "render" / f"r{r}_c{c}.png", run.dir / "art" / f"r{r}_c{c}.png"
        if not ren.exists() and not art.exists():
            continue
        im = Image.open(ren).convert("RGBA") if ren.exists() else Image.new("RGBA", (s, s), BG + (255,))
        if art.exists():
            im.alpha_composite(Image.open(art).convert("RGBA"))
        m = Image.new("L", im.size, 0)
        ImageDraw.Draw(m).polygon([(x - c * s, y - r * s) for x, y in poly], fill=255)
        Image.composite(im.convert("RGB"), Image.new("RGB", im.size, BG), m).save(out / f"r{r}_c{c}.png")
    return out


def build(run: Run, version: str = "v1", quality: int = 90) -> None:
    import pyvips

    v = run.view
    s = v["shot_px"]
    black = pyvips.Image.black(s, s, bands=3)
    src = _masked(run)
    tiles = []
    for r in range(v["rows"]):
        for c in range(v["cols"]):
            p = src / f"r{r}_c{c}.png"
            tiles.append(pyvips.Image.new_from_file(str(p), access="sequential")[0:3] if p.exists() else black)
    img = pyvips.Image.arrayjoin(tiles, across=v["cols"])
    out = ROOT / "site" / "maps" / run.name / version
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    img.dzsave(str(out / "map"), tile_size=512, overlap=0, suffix=f".webp[Q={quality}]", layout="dz")

    def load(name):
        p = run.dir / "osm" / name
        return json.loads(p.read_text()) if p.exists() else {}

    places = [{"n": p["name"], "k": p.get("kind"), "x": p["px"][0], "y": p["px"][1]}
              for p in load("places_px.json").get("points", [])]
    borders = [f["rings"][0] for f in load("borders_px.json").get("features", [])]
    from . import geo
    aff = geo.load(run)
    bridges = [{"slug": b["slug"], "name": b["name"],
                "br": geo.to_px(aff, b["br_end"]["lat"], b["br_end"]["lon"]),
                "py": geo.to_px(aff, b["py_end"]["lat"], b["py_end"]["lon"])}
               for b in run.region.get("bridges", [])]
    sat = load("sat.json")
    attribution = (f"Imagem derivada de satélite do Google ({sat['copyright']})" if sat.get("copyright")
                   else "Imagem derivada de Google Photorealistic 3D Tiles")
    attribution += " · Dados © OpenStreetMap contributors" + (" · Overture Maps" if run.region.get("buildings") == "overture" else "")
    meta = {"title": run.region.get("title", run.name), "width": v["canvas_px"][0], "height": v["canvas_px"][1],
            "attribution": attribution,
            "dzi": f"maps/{run.name}/{version}/map.dzi", "places": places, "borders": borders,
            "geo": aff, "bridges": bridges, "live": run.region.get("now_services", {})}
    (ROOT / "site" / "maps" / run.name / "meta.json").write_text(json.dumps(meta))
    (ROOT / "site" / "config.json").write_text(json.dumps({"map": f"maps/{run.name}/meta.json"}))
    print(f"pyramid -> {out}; open with: python -m http.server -d site 8080")
