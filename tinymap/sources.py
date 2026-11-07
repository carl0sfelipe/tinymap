"""Fontes de dados reais para a maquete: prédios do Overture e imagem de satélite.

Overture Maps (buildings) junta OSM + Google Open Buildings + Microsoft ML Buildings: no piloto da
Tríplice são 3.947 prédios contra 2.698 só do OSM. A imagem de satélite (Google Map Tiles API, 2D
"satellite") é projetada sobre o chão e os telhados da maquete, para o render ter a cidade real.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

from .region import Run

OVERTURE_RELEASE = os.environ.get("TINYMAP_OVERTURE", "2026-10-21.0")


def overture_buildings(bbox: dict) -> list[dict]:
    """Prédios do Overture na caixa {south, west, north, east}, no formato de osm.fetch_scene."""
    import duckdb

    from .osm import altura_predio

    c = duckdb.connect()
    c.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2';")
    src = (f"read_parquet('s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=buildings/type=building/*',"
           " hive_partitioning=1)")
    rows = c.execute(f"""
        SELECT id, ST_AsGeoJSON(geometry), height, num_floors, min_height, min_floor, class, subtype,
               roof_color, facade_color
        FROM {src}
        WHERE bbox.xmin > ? AND bbox.xmax < ? AND bbox.ymin > ? AND bbox.ymax < ?""",
                     [bbox["west"], bbox["east"], bbox["south"], bbox["north"]]).fetchall()
    out = []
    for oid, gj, h, floors, mh, mf, cls, sub, roof, facade in rows:
        g = json.loads(gj)
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"] if g["type"] == "MultiPolygon" else []
        tags = {"height": str(h) if h else None, "building:levels": str(floors) if floors else None,
                "building": cls or sub or "", "min_height": str(mh) if mh else None,
                "building:min_level": str(mf) if mf else None}
        hh, mhh = altura_predio(tags, hash(oid) & 0xFFFF)
        for k, poly in enumerate(polys):
            # GeoJSON é [lon, lat]; a maquete usa [lat, lon]. Só o anel externo (pátios internos são raros aqui).
            ring = [[lat, lon] for lon, lat in poly[0]]
            out.append({"id": f"o{oid[:12]}{k}", "rings": [ring], "h": hh, "min_h": mhh,
                        "colour": facade, "roof_colour": roof})
    return out


# ── Satélite: Google Map Tiles API (2D, mapType satellite) ────────────────────

def _tile_xy(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def google_satellite(run: Run, bbox: dict, z: int = 19, pad: float = 0.005) -> Path:
    """Baixa os tiles de satélite da caixa (com folga) e grava osm/sat.jpg + osm/sat.json.

    z=19 dá ~0,27 m/px na latitude de Foz, perto dos 0,293 m/px do mapa. Precisa de
    GOOGLE_TILES_API_KEY (.env) com a Map Tiles API ativa. Atribuição obrigatória: sat.json guarda
    o texto de copyright devolvido pela API."""
    key = os.environ.get("GOOGLE_TILES_API_KEY") or _env_key()
    if not key:
        raise RuntimeError("falta GOOGLE_TILES_API_KEY no .env (Map Tiles API ativa no Google Cloud)")
    req = urllib.request.Request(f"https://tile.googleapis.com/v1/createSession?key={key}",
                                 data=json.dumps({"mapType": "satellite", "language": "pt-BR", "region": "BR"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        session = json.load(r)["session"]
    s, w, n, e = bbox["south"] - pad, bbox["west"] - pad, bbox["north"] + pad, bbox["east"] + pad
    x0, y0 = map(math.floor, _tile_xy(n, w, z))
    x1, y1 = map(math.floor, _tile_xy(s, e, z))
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    print(f"satélite z{z}: {cols} x {rows} = {cols * rows} tiles")
    cache = run.path("osm", "sat_tiles", ".keep").parent

    def get(t):
        tx, ty = t
        f = cache / f"{z}_{x0 + tx}_{y0 + ty}.jpg"
        if f.exists() and f.stat().st_size:
            return
        url = f"https://tile.googleapis.com/v1/2dtiles/{z}/{x0 + tx}/{y0 + ty}?session={session}&key={key}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read()
                tmp = f.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(f)
                return
            except Exception as ex:  # noqa: BLE001
                if attempt == 4:
                    raise RuntimeError(f"tile {x0 + tx},{y0 + ty}: {ex}") from ex
                time.sleep(2 * (attempt + 1))

    from concurrent.futures import ThreadPoolExecutor
    todo = [(tx, ty) for ty in range(rows) for tx in range(cols)]
    with ThreadPoolExecutor(32) as ex:  # ~1,3 tile/s em série; 32 em paralelo
        for i, _ in enumerate(ex.map(get, todo), 1):
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} tiles", flush=True)
    # atribuição (obrigatória nos termos do Google)
    vp = f"https://tile.googleapis.com/tile/v1/viewport?session={session}&key={key}&zoom={z}&north={n}&south={s}&east={e}&west={w}"
    try:
        with urllib.request.urlopen(vp, timeout=30) as r:
            copyright = json.load(r).get("copyright", "")
    except Exception:  # noqa: BLE001
        copyright = "Google"
    return save_chunks(run, cache, z, x0, y0, cols, rows, "google", copyright)


CHUNK = 4096  # textura WebGL segura; a maquete usa um pedaço por textura


def tone(img: Image.Image) -> Image.Image:
    """Clareia as sombras do satélite (mata e beirais quase pretos o modelo lia como água) e
    devolve um pouco de cor. Curva: gama 0,75 + leve saturação."""
    from PIL import ImageEnhance
    lut = [round(255 * (i / 255) ** 0.75) for i in range(256)]
    return ImageEnhance.Color(img.point(lut * 3)).enhance(1.15)


def save_chunks(run: Run, cache: Path, z: int, x0: int, y0: int, cols: int, rows: int, provider: str,
                copyright: str) -> Path:
    """Monta os pedaços de 4096 px (osm/sat_r{i}_c{j}.jpg) direto dos tiles, sem a mosaica inteira
    na memória (a região toda em z19 tem 37 mil x 58 mil px), + osm/sat.json e uma prévia."""
    per = CHUNK // 256
    W, H = cols * 256, rows * 256
    chunks = []
    prev = Image.new("RGB", (cols * 32, rows * 32))
    for i in range(0, rows, per):
        for j in range(0, cols, per):
            tw, th = min(per, cols - j), min(per, rows - i)
            part = Image.new("RGB", (tw * 256, th * 256))
            for ty in range(th):
                for tx in range(tw):
                    t = Image.open(cache / f"{z}_{x0 + j + tx}_{y0 + i + ty}.jpg").convert("RGB")
                    part.paste(t, (tx * 256, ty * 256))
            part = tone(part)
            name = f"sat_r{i // per}_c{j // per}.jpg"
            part.save(run.path("osm", name), quality=92)
            prev.paste(part.resize((tw * 32, th * 32)), (j * 32, i * 32))
            chunks.append({"file": name, "px": j * 256, "py": i * 256, "w": part.width, "h": part.height})
    meta = {"z": z, "x0": x0, "y0": y0, "w": W, "h": H, "chunks": chunks, "provider": provider, "copyright": copyright}
    out = run.path("osm", "sat.json")
    out.write_text(json.dumps(meta, indent=1))
    prev.save(run.path("osm", "sat_preview.jpg"), quality=85)
    print(f"satélite {W} x {H} px em {len(chunks)} pedaços; atribuição: {copyright}")
    return out


def _env_key() -> str | None:
    from .region import ROOT
    p = ROOT / ".env"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.strip().startswith("GOOGLE_TILES_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def overture_roads(bbox: dict) -> list[dict]:
    """Ruas do Overture (transportation/segment) no formato de osm.fetch_scene."""
    import duckdb

    from .osm import LARGURA_VIA
    c = duckdb.connect()
    c.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2';")
    src = (f"read_parquet('s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=transportation/type=segment/*',"
           " hive_partitioning=1)")
    rows = c.execute(f"""SELECT ST_AsGeoJSON(geometry), subtype, class FROM {src}
        WHERE bbox.xmin > ? AND bbox.xmax < ? AND bbox.ymin > ? AND bbox.ymax < ? AND subtype IN ('road', 'rail')""",
                     [bbox["west"], bbox["east"], bbox["south"], bbox["north"]]).fetchall()
    out = []
    for gj, sub, cls in rows:
        g = json.loads(gj)
        if g["type"] != "LineString":
            continue
        line = [[lat, lon] for lon, lat in g["coordinates"]]
        if sub == "rail":
            out.append({"line": line, "w": 3, "kind": "rail"})
        else:
            out.append({"line": line, "w": LARGURA_VIA.get(cls or "", 5), "kind": "road",
                        "major": cls in ("motorway", "trunk", "primary", "secondary")})
    return out


def overture_scene(run: Run, bbox: dict, water_ll: Path | None = None) -> Path:
    """Maquete da região sem Overpass: prédios e ruas do Overture + água do OSM (osm/water_ll.json
    de `tinymap osm`, que já cobre a região inteira)."""
    pad = 0.003
    box = {"south": bbox["south"] - pad, "west": bbox["west"] - pad, "north": bbox["north"] + pad, "east": bbox["east"] + pad}
    t0 = time.time()
    buildings = overture_buildings(box)
    roads = overture_roads(box)
    water = []
    if water_ll and Path(water_ll).exists():
        water = merge_water([{"rings": f["rings"], "holes": f.get("holes", [])}
                             for f in json.loads(Path(water_ll).read_text())["features"]])
    scene = {"bbox": bbox, "buildings": buildings, "roads": roads, "green": [], "water": water}
    out = run.path("osm", "scene_ll.json")
    out.write_text(json.dumps(scene))
    print(f"maquete: {len(buildings)} prédios, {len(roads)} vias, {len(water)} águas ({time.time() - t0:.0f} s)")
    return out


def merge_water(feats: list[dict], gap_deg: float = 0.0002) -> list[dict]:
    """Junta os polígonos de água e fecha as frestas entre eles. O OSM divide o Paraná na fronteira e as
    bordas não se encostam: no render aparecia uma lente de chão no meio do rio (até ~40 m de largura)."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union
    polys = []
    for f in feats:
        for ring in f["rings"]:
            if len(ring) < 4:
                continue
            pg = Polygon([(lon, lat) for lat, lon in ring], [[(lon, lat) for lat, lon in h] for h in f.get("holes", []) if len(h) >= 4])
            polys.append(pg.buffer(0))
    u = unary_union(polys).buffer(gap_deg).buffer(-gap_deg * 0.8)  # fecha frestas, devolve ~4 m de margem
    out = []
    for pg in (u.geoms if isinstance(u, MultiPolygon) else [u]):
        if pg.area < 1e-9:
            continue
        ll = lambda cs: [[lat, lon] for lon, lat in cs]  # noqa: E731
        out.append({"rings": [ll(pg.exterior.coords)], "holes": [ll(i.coords) for i in pg.interiors]})
    return out
