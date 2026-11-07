"""OpenStreetMap layers projected with the render camera: water (mask + teacher hints + checks),
places (viewer search) and national borders (viewer overlay)."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from .region import Run, node

# Servidores tentados em ordem (o principal costuma dar 504 em hora de pico).
# TINYMAP_OVERPASS=url1,url2 troca a lista; TINYMAP_OVERPASS_TIMEOUT é o teto por pedido, em segundos.
OVERPASS = os.environ.get("TINYMAP_OVERPASS", "https://overpass-api.de/api/interpreter,"
                          "https://overpass.kumi.systems/api/interpreter,"
                          "https://overpass.private.coffee/api/interpreter").split(",")


def _overpass(q: str) -> dict:
    data = urllib.parse.urlencode({"data": q}).encode()
    teto = float(os.environ.get("TINYMAP_OVERPASS_TIMEOUT", 180))
    for attempt in range(2):
        for url in OVERPASS:
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": "tinymap/0.1"})
                with urllib.request.urlopen(req, timeout=teto) as r:
                    return json.load(r)
            except Exception as e:  # noqa: BLE001
                print(f"overpass {urllib.parse.urlparse(url).netloc} failed ({e})")
        time.sleep(15 * (attempt + 1))
    raise RuntimeError("overpass unreachable")


def _rings_of(el: dict) -> tuple[list, list]:
    if el["type"] == "way":
        return [[[p["lat"], p["lon"]] for p in el["geometry"]]], []
    outer, inner = [], []
    for m in el.get("members", []):
        if "geometry" not in m:
            continue
        ring = [[p["lat"], p["lon"]] for p in m["geometry"]]
        (inner if m.get("role") == "inner" else outer).append(ring)
    return _join(outer), _join(inner)


def _join(parts: list) -> list:
    """Join open way segments of a multipolygon into closed rings."""
    rings, open_ = [], [p[:] for p in parts]
    while open_:
        cur = open_.pop(0)
        while cur[0] != cur[-1]:
            for i, p in enumerate(open_):
                if p[0] == cur[-1]:
                    cur += p[1:]
                elif p[-1] == cur[-1]:
                    cur += p[::-1][1:]
                else:
                    continue
                open_.pop(i)
                break
            else:
                break
        rings.append(cur)
    return rings


def water_height(run: Run) -> float:
    """Altura da lâmina d'água na projeção. Na maquete OSM a água é desenhada no chão (target_height_m),
    não na altura real do rio; usar water_height_m ali deslocava a máscara ~75 m (≈250 px) para baixo."""
    if run.region.get("source") == "osm":
        return run.region["camera"]["target_height_m"]
    return run.region.get("water_height_m", 0)


def fetch(run: Run) -> None:
    b = run.region["bbox"]
    pad = 0.01
    bbox = f"{b['south'] - pad},{b['west'] - pad},{b['north'] + pad},{b['east'] + pad}"
    water = _overpass(f"""[out:json][timeout:240];
      (nwr["natural"="water"]({bbox}); nwr["waterway"="riverbank"]({bbox}); nwr["water"="river"]({bbox});
       nwr["natural"="bay"]({bbox}););
      out geom;""")
    feats = []
    for el in water["elements"]:
        if el["type"] == "node":
            continue
        outer, inner = _rings_of(el)
        if outer:
            feats.append({"id": f"{el['type'][0]}{el['id']}", "kind": "water", "rings": outer, "holes": inner})
    json.dump({"h": water_height(run), "features": feats}, open(run.path("osm", "water_ll.json"), "w"))
    node("project.mjs", run.dir, run.path("osm", "water_ll.json"), run.path("osm", "water_px.json"))

    places = _overpass(f"""[out:json][timeout:240];
      (node["place"~"suburb|neighbourhood|quarter|town|city|village"]({bbox});
       nwr["tourism"~"attraction|museum|viewpoint"]["name"]({bbox});
       nwr["amenity"~"place_of_worship|university|hospital|marketplace"]["name"]({bbox});
       nwr["shop"="mall"]["name"]({bbox}); nwr["man_made"="bridge"]["name"]({bbox});
       nwr["aeroway"="aerodrome"]["name"]({bbox}););
      out center tags;""")
    pts = []
    for el in places["elements"]:
        t = el.get("tags", {})
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        name = t.get("name:pt") or t.get("name")
        if lat and name:
            pts.append({"name": name, "lat": lat, "lon": lon, "kind": t.get("place") or t.get("tourism")
                        or t.get("amenity") or t.get("shop") or t.get("man_made") or t.get("aeroway")})
    for lm in run.region.get("landmarks", []):
        pts.append({**lm, "kind": "landmark"})
    json.dump({"points": pts}, open(run.path("osm", "places_ll.json"), "w"))
    node("project.mjs", run.dir, run.path("osm", "places_ll.json"), run.path("osm", "places_px.json"))

    borders = _overpass(f"""[out:json][timeout:240];
      way["boundary"="administrative"]["admin_level"="2"]({bbox}); out geom;""")
    lines = [{"id": f"w{el['id']}", "kind": "border", "rings": [[[p["lat"], p["lon"]] for p in el["geometry"]]]}
             for el in borders["elements"]]
    json.dump({"h": water_height(run), "features": lines}, open(run.path("osm", "borders_ll.json"), "w"))
    node("project.mjs", run.dir, run.path("osm", "borders_ll.json"), run.path("osm", "borders_px.json"))
    print(f"water {len(feats)}  places {len(pts)}  border segments {len(lines)}")


class Water:
    def __init__(self, run: Run):
        p = run.dir / "osm" / "water_px.json"
        self.feats = json.loads(p.read_text())["features"] if p.exists() else []

    def mask(self, box: tuple[int, int, int, int]) -> Image.Image:
        x0, y0, x1, y1 = box
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(m)
        for f in self.feats:
            for r in f["rings"]:
                if len(r) > 2:
                    d.polygon([(x - x0, y - y0) for x, y in r], fill=255)
        for f in self.feats:
            for r in f["holes"]:
                if len(r) > 2:
                    d.polygon([(x - x0, y - y0) for x, y in r], fill=0)
        return m


def water_sentence(mask: Image.Image) -> str:
    a = np.asarray(mask) > 127
    share = a.mean()
    if share < 0.002:
        return "WATER IN THIS VIEW: none. Nothing here uses the water colour."
    h, w = a.shape
    names = [["top left", "top", "top right"], ["left", "centre", "right"], ["bottom left", "bottom", "bottom right"]]
    where = [names[r][c] for r in range(3) for c in range(3)
             if a[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3].mean() > 0.01]
    return (f"WATER IN THIS VIEW: about {share * 100:.0f}% of the view, in the {', '.join(where)}: only the river "
            "the render shows there. Nothing else uses the water colour.")


# ── Maquete 3D a partir do OSM (onde não há Google 3D Tiles) ─────────────────

# Altura padrão por tipo quando o OSM não tem height nem building:levels. Calibrado "a olho"
# para a fronteira: casas térreas e sobrados em Foz, galerias de 3-5 andares no microcentro de CDE.
ALTURA_TIPO = {
    "house": 6, "detached": 6, "residential": 7, "semidetached_house": 6, "terrace": 7, "bungalow": 4,
    "apartments": 18, "hotel": 24, "commercial": 12, "retail": 9, "office": 18, "industrial": 9,
    "warehouse": 9, "church": 14, "cathedral": 30, "mosque": 14, "school": 8, "university": 12,
    "hospital": 15, "garage": 3, "garages": 3, "shed": 3, "roof": 4, "kiosk": 3, "service": 4,
}
LARGURA_VIA = {
    "motorway": 18, "trunk": 16, "primary": 14, "secondary": 11, "tertiary": 9, "residential": 7,
    "unclassified": 7, "living_street": 6, "service": 4, "pedestrian": 6, "footway": 2, "path": 1.5,
    "cycleway": 2, "track": 3, "motorway_link": 8, "trunk_link": 8, "primary_link": 8,
}
VERDE = {"park": "green", "grass": "green", "garden": "green", "recreation_ground": "green",
         "pitch": "green", "golf_course": "green", "meadow": "green", "cemetery": "green",
         "forest": "forest", "wood": "forest", "scrub": "forest", "nature_reserve": "forest",
         "sand": "sand", "beach": "sand"}


def _num(v: str | None) -> float | None:
    if not v:
        return None
    try:
        return float(v.replace(",", ".").split()[0].rstrip("m"))
    except ValueError:
        return None


def altura_predio(tags: dict, rng_seed: int = 0) -> tuple[float, float]:
    """(altura, min_height) em metros a partir das tags OSM."""
    h = _num(tags.get("height"))
    if h is None and (lv := _num(tags.get("building:levels"))) is not None:
        h = lv * 3.2 + (_num(tags.get("roof:levels")) or 0) * 2
    if h is None:
        h = ALTURA_TIPO.get(tags.get("building", ""), 7.0)
        h *= 0.85 + (rng_seed % 31) / 100  # variação leve para a maquete não ficar uniforme
    mh = _num(tags.get("min_height"))
    if mh is None and (ml := _num(tags.get("building:min_level"))) is not None:
        mh = ml * 3.2
    return round(h, 1), round(mh or 0.0, 1)


def fetch_scene(run: Run, pilot: bool = False) -> dict:
    b = run.region["pilot_bbox"] if pilot and run.region.get("pilot_bbox") else run.region["bbox"]
    pad = 0.003
    bbox = f"{b['south'] - pad},{b['west'] - pad},{b['north'] + pad},{b['east'] + pad}"
    data = _overpass(f"""[out:json][timeout:600][maxsize:1073741824];
      (way["building"]({bbox}); relation["building"]["type"="multipolygon"]({bbox});
       way["building:part"]({bbox});
       way["highway"]({bbox}); way["railway"="rail"]({bbox});
       nwr["natural"~"water|wood|scrub|sand|beach"]({bbox}); nwr["waterway"="riverbank"]({bbox});
       nwr["leisure"~"park|garden|pitch|golf_course|nature_reserve|recreation_ground"]({bbox});
       nwr["landuse"~"grass|forest|meadow|cemetery|recreation_ground"]({bbox}););
      out geom;""")
    buildings, roads, green, water = [], [], [], []
    for el in data["elements"]:
        t = el.get("tags", {})
        if el["type"] == "node":
            continue
        if "building" in t or "building:part" in t:
            outer, inner = _rings_of(el)
            if not outer:
                continue
            h, mh = altura_predio(t, el["id"])
            buildings.append({"id": f"{el['type'][0]}{el['id']}", "rings": outer, "h": h, "min_h": mh,
                              "colour": t.get("building:colour"), "roof_colour": t.get("roof:colour")})
        elif "highway" in t and el["type"] == "way":
            hw = t["highway"]
            if hw in ("proposed", "construction", "steps", "elevator", "corridor"):
                continue
            w = _num(t.get("width")) or LARGURA_VIA.get(hw, 5)
            roads.append({"line": [[p["lat"], p["lon"]] for p in el["geometry"]], "w": w,
                          "major": hw in ("motorway", "trunk", "primary", "secondary"), "kind": "road"})
        elif t.get("railway") == "rail" and el["type"] == "way":
            roads.append({"line": [[p["lat"], p["lon"]] for p in el["geometry"]], "w": 3, "kind": "rail"})
        else:
            outer, inner = _rings_of(el)
            if not outer:
                continue
            kind = t.get("natural") or t.get("leisure") or t.get("landuse") or t.get("waterway")
            if kind in ("water", "riverbank") or t.get("waterway") == "riverbank":
                water.append({"rings": outer, "holes": inner})
            elif kind in VERDE:
                green.append({"rings": outer, "holes": inner, "kind": VERDE[kind]})
    scene = {"bbox": b, "buildings": buildings, "roads": roads, "green": green, "water": water}
    json.dump(scene, open(run.path("osm", "scene_ll.json"), "w"))
    com_altura = sum(1 for el in data["elements"] if el.get("tags", {}).get("height") or el.get("tags", {}).get("building:levels"))
    print(f"prédios {len(buildings)} (com altura/andares no OSM: {com_altura}), vias {len(roads)}, "
          f"verde {len(green)}, água {len(water)}")
    return scene
