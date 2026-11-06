"""Finished art shots -> Deep Zoom pyramid (WebP) + search index + borders for the viewer in site/.

Uses libvips (pyvips), which streams, so a 1.5 gigapixel canvas never has to fit in RAM.
"""

from __future__ import annotations

import json
import shutil

from .region import ROOT, Run


def build(run: Run, version: str = "v1", quality: int = 90) -> None:
    import pyvips

    v = run.view
    s = v["shot_px"]
    black = pyvips.Image.black(s, s, bands=3)
    tiles = []
    for r in range(v["rows"]):
        for c in range(v["cols"]):
            p = run.dir / "art" / f"r{r}_c{c}.png"
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
    meta = {"title": run.region.get("title", run.name), "width": v["canvas_px"][0], "height": v["canvas_px"][1],
            "dzi": f"maps/{run.name}/{version}/map.dzi", "places": places, "borders": borders,
            "geo": aff, "bridges": bridges, "live": run.region.get("now_services", {})}
    (ROOT / "site" / "maps" / run.name / "meta.json").write_text(json.dumps(meta))
    (ROOT / "site" / "config.json").write_text(json.dumps({"map": f"maps/{run.name}/meta.json"}))
    print(f"pyramid -> {out}; open with: python -m http.server -d site 8080")
