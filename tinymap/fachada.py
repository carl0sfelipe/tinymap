"""Fichas de fachada: janelas e andares de verdade a partir de uma foto da rua.

Onde não há 3D do Google, a maquete (OSM/Overture) tem forma e altura, mas a parede é lisa e a IA
inventa as janelas. A foto não vai para o desenho: ela vira DADO numa ficha por prédio (andares,
janelas por andar, cor da parede, loja no térreo), e a maquete desenha a grade de janelas por
regra. A LoRA só repinta no estilo, e o detalhe já está no render. Onde há 3D do Google, a ficha e a
foto entram como referência quando se pede à IA para refazer um prédio no estúdio.

Origens da foto, da mais limpa para a mais restrita:
  foto        sua, ou do lojista: enviada no estúdio ou com --file (padrão; sem custo e sem termos)
  mapillary   fotos de rua abertas (MAPILLARY_TOKEN); licença CC-BY-SA, atribuição no site
  streetview  Street View Static API (GOOGLE_MAPS_API_KEY). Metadados grátis; imagem paga por
              requisição, com teto mensal (`fachada.streetview_max_mes` no YAML, padrão 0 = desligado).
              Os termos da Google Maps Platform restringem extrair dados das imagens: confira antes.

Ângulo: a partir do ponto da foto (panorama ou GPS) e do prédio (centro e altura), calcula o rumo,
a inclinação e o campo de visão para enquadrar a fachada inteira.

Sugestão automática: periodicidade das bordas na foto (o mesmo truque da grade de pixel) estima a
altura do andar e o espaço entre janelas. É só sugestão: no estúdio, você confirma ou corrige.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

R_EARTH = 6371008.8
SV_META = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMG = "https://maps.googleapis.com/maps/api/streetview"
MAPILLARY = "https://graph.mapillary.com/images"


# ---------- geometria ----------
def bearing(lat1, lon1, lat2, lon2) -> float:
    """Rumo de 1 para 2, em graus a partir do norte (0-360)."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def distance(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def camera_for(cam_ll, bld_ll, height_m: float, width_m: float = 15.0, eye_m: float = 2.5) -> dict:
    """Rumo, inclinação e campo de visão que enquadram a fachada inteira vista do ponto da câmera."""
    d = max(distance(*cam_ll, *bld_ll), 3.0)
    heading = bearing(*cam_ll, *bld_ll)
    top = math.degrees(math.atan2(height_m - eye_m, d))
    bottom = math.degrees(math.atan2(-eye_m, d))
    pitch = (top + bottom) / 2
    v = (top - bottom) * 1.15
    h = 2 * math.degrees(math.atan2(width_m / 2, d)) * 1.15
    fov = float(np.clip(max(v, h), 30, 120))
    return {"heading": round(heading, 1), "pitch": round(float(np.clip(pitch, -30, 60)), 1), "fov": round(fov, 1),
            "dist_m": round(d, 1)}


# ---------- sugestão a partir da foto ----------
def _period(profile: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    """Período dominante (autocorrelação) entre lo e hi px, e a força dele (0-1)."""
    p = profile - profile.mean()
    if not p.any():
        return 0, 0.0
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac = ac / ac[0]
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return 0, 0.0
    seg = ac[lo:hi + 1]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0]
    if not peaks:
        return 0, 0.0
    best = max(seg[i] for i in peaks)
    # o primeiro pico forte: os múltiplos do período (2, 3 janelas) também correlacionam bem
    k = next(i for i in peaks if seg[i] >= 0.8 * best) + lo
    return k, float(ac[k])


def wall_colour(img: Image.Image) -> str:
    """Cor mais comum da metade central (fora céu e chão), em hex."""
    a = np.asarray(img.convert("RGB").resize((96, 96), Image.BOX), np.int16)[24:80, 16:80].reshape(-1, 3)
    q = (a // 24) * 24 + 12
    vals, n = np.unique(q, axis=0, return_counts=True)
    c = vals[int(np.argmax(n))]
    return "#%02x%02x%02x" % tuple(int(v) for v in np.clip(c, 0, 255))


def suggest(img: Image.Image, facade_box: tuple[int, int, int, int] | None = None) -> dict:
    """Estimativa de andares e janelas por andar numa foto (de preferência recortada na fachada).
    Bordas horizontais se repetem a cada andar; verticais, a cada janela."""
    im = img.convert("L")
    if facade_box:
        im = im.crop(facade_box)
    W, H = im.size
    s = 640 / max(W, H)
    im = im.resize((max(32, int(W * s)), max(32, int(H * s))), Image.BILINEAR)
    a = np.asarray(im, np.float32)
    gy = np.abs(np.diff(a, axis=0))[:, :]
    gx = np.abs(np.diff(a, axis=1))[:, :]
    rows = gy.mean(1)
    cols = gx.mean(0)
    h, w = a.shape
    fp, fs = _period(rows, max(6, h // 40), max(8, h // 2))
    wp, ws = _period(cols, max(5, w // 60), max(8, w // 2))
    levels = int(round(h / fp)) if fp else 0
    per = int(round(w / wp)) if wp else 0
    conf = round(float(min(fs, ws)), 2) if fp and wp else 0.0
    return {"levels": levels, "cols": per, "wall": wall_colour(img.crop(facade_box) if facade_box else img),
            "confianca": conf}


# ---------- fichas ----------
class Fichas:
    def __init__(self, run):
        self.run = run
        self.dir = run.dir / "fachadas"
        self.dir.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.dir.glob("*.json"))]

    def get(self, fid: str) -> dict | None:
        p = self.dir / f"{fid}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def save(self, f: dict) -> dict:
        f.setdefault("id", f"f{int(time.time() * 1000)}")
        f["atualizado"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (self.dir / f"{f['id']}.json").write_text(json.dumps(f, ensure_ascii=False, indent=1))
        return f

    def photo_path(self, fid: str) -> Path:
        return self.dir / f"{fid}.jpg"

    def add_photo(self, lat: float, lon: float, img: Image.Image, fonte: str, extra: dict | None = None) -> dict:
        f = {"lat": lat, "lon": lon, "fonte": fonte, "status": "sugerida", **(extra or {})}
        f = self.save(f)
        img.convert("RGB").save(self.photo_path(f["id"]), quality=90)
        f.update(suggest(img))
        f["foto"] = self.photo_path(f["id"]).name
        return self.save(f)


# ---------- cena da maquete ----------
def _inside(ring, lat, lon) -> bool:
    ok, j = False, len(ring) - 1
    for i in range(len(ring)):
        (yi, xi), (yj, xj) = ring[i][:2], ring[j][:2]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi:
            ok = not ok
        j = i
    return ok


def _centroid(ring) -> tuple[float, float]:
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def find_building(scene: dict, lat: float, lon: float, max_m: float = 25.0) -> dict | None:
    """O prédio da cena que contém o ponto; senão, o de centro mais perto (até max_m)."""
    best, bd = None, max_m
    for b in scene.get("buildings", []):
        if any(_inside(r, lat, lon) for r in b["rings"][:1]):
            return b
        d = distance(lat, lon, *_centroid(b["rings"][0]))
        if d < bd:
            best, bd = b, d
    return best


def facade_width_m(b: dict) -> float:
    """Lado mais comprido do prédio (aproxima a largura da fachada)."""
    r = b["rings"][0]
    return max(distance(*r[i][:2], *r[i + 1][:2]) for i in range(len(r) - 1))


def apply_to_scene(run) -> int:
    """Leva as fichas confirmadas para osm/scene_ll.json: cada prédio ganha `win` (altura do andar,
    espaço entre janelas, loja no térreo), cor da parede e altura pelos andares. Devolve quantos."""
    p = run.path("osm", "scene_ll.json")
    scene = json.loads(Path(p).read_text())
    n = 0
    for f in Fichas(run).all():
        if f.get("status") != "confirmada" or not f.get("levels"):
            continue
        b = find_building(scene, f["lat"], f["lon"])
        if b is None:
            continue
        floor_h = float(f.get("floor_h") or 3.0)
        b["h"] = round(f["levels"] * floor_h + (0.6 if f.get("shop") else 0), 1)
        if f.get("wall"):
            b["colour"] = f["wall"]
        w = facade_width_m(b)
        spacing = w / max(1, int(f.get("cols") or round(w / 3)))
        b["win"] = {"floor_h": floor_h, "spacing": round(spacing, 2), "shop": bool(f.get("shop")),
                    "levels": int(f["levels"])}
        b["fachada"] = f["id"]
        n += 1
    Path(p).write_text(json.dumps(scene))
    return n


def facts(f: dict) -> str:
    """Frase da ficha para o prompt da IA (estúdio, regiões com 3D do Google)."""
    bits = []
    if f.get("levels"):
        bits.append(f"{f['levels']} floors")
    if f.get("cols"):
        bits.append(f"{f['cols']} windows per floor on the street side")
    if f.get("wall"):
        bits.append(f"wall colour {f['wall']}")
    if f.get("shop"):
        bits.append("shops with awnings on the ground floor")
    return ("The selected building has " + ", ".join(bits) + ". Draw exactly that, no extra floors or windows.") if bits else ""


# ---------- fontes online ----------
class Budget:
    """Contador mensal de imagens pagas do Street View (runs/streetview_uso.json)."""

    def __init__(self, runs_dir: Path, max_mes: int):
        self.p, self.max = runs_dir / "streetview_uso.json", max_mes

    def _load(self) -> dict:
        try:
            return json.loads(self.p.read_text())
        except (OSError, ValueError):
            return {}

    def used(self) -> int:
        return self._load().get(time.strftime("%Y-%m"), 0)

    def take(self) -> None:
        if self.used() >= self.max:
            raise RuntimeError(f"teto mensal do Street View atingido ({self.max}); "
                               "aumente fachada.streetview_max_mes no YAML ou envie a foto à mão")
        d = self._load()
        d[time.strftime("%Y-%m")] = d.get(time.strftime("%Y-%m"), 0) + 1
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.p.write_text(json.dumps(d))


def _get(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def streetview(lat, lon, height_m, width_m, key: str, budget: Budget, radius: int = 60) -> tuple[Image.Image, dict]:
    """Panorama mais perto (metadados, grátis) e a imagem enquadrada na fachada (paga, com teto)."""
    import io
    q = urllib.parse.urlencode({"location": f"{lat},{lon}", "radius": radius, "source": "outdoor", "key": key})
    meta = json.loads(_get(f"{SV_META}?{q}"))
    if meta.get("status") != "OK":
        raise RuntimeError(f"sem Street View a {radius} m ({meta.get('status')})")
    cam = (meta["location"]["lat"], meta["location"]["lng"])
    ang = camera_for(cam, (lat, lon), height_m, width_m)
    budget.take()
    q = urllib.parse.urlencode({"pano": meta["pano_id"], "size": "640x640", "heading": ang["heading"],
                                "pitch": ang["pitch"], "fov": ang["fov"], "key": key})
    img = Image.open(io.BytesIO(_get(f"{SV_IMG}?{q}")))
    return img, {"pano": meta["pano_id"], "data_foto": meta.get("date"), "camera": list(cam), **ang,
                 "atribuicao": meta.get("copyright", "© Google")}


def mapillary(lat, lon, height_m, width_m, token: str, radius_m: int = 50) -> tuple[Image.Image, dict]:
    """Foto aberta mais perto que olha para o prédio (rumo da câmera a até 40° do rumo ao prédio)."""
    import io
    d = radius_m / 111_000
    q = urllib.parse.urlencode({"access_token": token, "fields": "id,computed_geometry,compass_angle,thumb_2048_url,captured_at,creator",
                                "bbox": f"{lon - d},{lat - d},{lon + d},{lat + d}", "limit": 100})
    data = json.loads(_get(f"{MAPILLARY}?{q}")).get("data", [])
    best, score = None, 1e9
    for it in data:
        lo, la = it["computed_geometry"]["coordinates"]
        need = bearing(la, lo, lat, lon)
        off = abs((it.get("compass_angle", need) - need + 180) % 360 - 180)
        dist = distance(la, lo, lat, lon)
        if off > 40 or dist < 4:
            continue
        sc = off / 40 + dist / radius_m
        if sc < score:
            best, score = (it, la, lo), sc
    if not best:
        raise RuntimeError(f"nenhuma foto do Mapillary olhando para o prédio a {radius_m} m")
    it, la, lo = best
    img = Image.open(io.BytesIO(_get(it["thumb_2048_url"])))
    return img, {"mapillary_id": it["id"], "camera": [la, lo], **camera_for((la, lo), (lat, lon), height_m, width_m),
                 "atribuicao": f"Mapillary ({(it.get('creator') or {}).get('username', 'contribuidor')}), CC-BY-SA"}


def fetch(run, lat: float, lon: float, fonte: str, env: dict | None = None) -> dict:
    """Busca a foto do prédio em (lat, lon) e cria a ficha sugerida."""
    from .doctor import _env
    env = env or _env()
    scene_p = run.path("osm", "scene_ll.json")
    b = find_building(json.loads(Path(scene_p).read_text()), lat, lon) if Path(scene_p).exists() else None
    h = float(b["h"]) if b else 15.0
    w = facade_width_m(b) if b else 15.0
    if b:
        lat, lon = _centroid(b["rings"][0])
    if fonte == "streetview":
        key = env.get("GOOGLE_MAPS_API_KEY") or env.get("GOOGLE_TILES_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY no .env (Street View Static API ativada no projeto)")
        mx = int(run.region.get("fachada", {}).get("streetview_max_mes", 0))
        if mx <= 0:
            raise RuntimeError("Street View desligado: defina fachada.streetview_max_mes no YAML (ex.: 300)")
        img, extra = streetview(lat, lon, h, w, key, Budget(run.dir.parent, mx))
    elif fonte == "mapillary":
        tok = env.get("MAPILLARY_TOKEN")
        if not tok:
            raise RuntimeError("MAPILLARY_TOKEN no .env (grátis em mapillary.com/dashboard/developers)")
        img, extra = mapillary(lat, lon, h, w, tok)
    else:
        raise ValueError(fonte)
    if b:
        extra["predio"] = b.get("id")
    return Fichas(run).add_photo(lat, lon, img, fonte, extra)
