"""OpenStreetMap layers projected with the render camera: water (mask + teacher hints + checks),
places (viewer search) and national borders (viewer overlay)."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from .region import Run, node

OVERPASS = "https://overpass-api.de/api/interpreter"


def _overpass(q: str) -> dict:
    data = urllib.parse.urlencode({"data": q}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(OVERPASS, data=data, headers={"User-Agent": "tinymap/0.1"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print(f"overpass failed ({e}); retrying")
            time.sleep(10 * (attempt + 1))
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
    json.dump({"h": run.region.get("water_height_m", 0), "features": feats}, open(run.path("osm", "water_ll.json"), "w"))
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
    json.dump({"h": run.region.get("water_height_m", 0), "features": lines}, open(run.path("osm", "borders_ll.json"), "w"))
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
