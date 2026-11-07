"""Cidade de demonstração: um mapa pequeno, sintético e determinístico, com os defeitos típicos da
geração por IA já plantados (e anotados em truth.json).

Serve para três coisas:
  - abrir a oficina sem GPU, sem chave e sem dados reais (onboarding da plataforma);
  - testar de ponta a ponta todas as ferramentas determinísticas (polimento, caçador de lagos, autofix);
  - medir se uma ferramenta melhora o mapa, contra um gabarito conhecido.

    tinymap demo                      # cria runs/demo e regions/demo.yaml
    tinymap edit regions/demo.yaml    # abre a oficina nele

Defeitos plantados na "arte da IA":
  - pixel mole: a arte foi desenhada em pixels de 3 px e depois borrada, como sai do modelo;
  - emendas: cada bloco de 2048 px tem uma deriva de cor própria (mais quente, mais escuro...);
  - água inventada: lagoas azuis e turquesa em grama, terreno e rua;
  - terra no rio: um trecho do rio pintado de verde;
  - um shot sem arte (mostra a maquete).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

from .region import ROOT

NAME = "demo"
SHOT = 1024
COLS, ROWS = 4, 3
W, H = COLS * SHOT, ROWS * SHOT
MPP = 0.4
BLOCK = 2048
THETA = math.radians(15)
TILT = math.cos(math.radians(45))
LAT0, LON0 = -25.5097, -54.6110          # microcentro de Ciudad del Este, só para as coordenadas
GROUND = (207, 202, 185)
ROAD = (108, 116, 116)
RIVER = (64, 118, 196)
PARK = (112, 168, 84)
ROOFS = [(181, 96, 63), (154, 154, 150), (201, 195, 181), (125, 138, 145), (163, 74, 51), (216, 210, 196), (111, 127, 106)]


def _to_px(e: float, n: float, h: float = 0.0) -> tuple[float, float]:
    """metros (leste, norte, altura) -> px do canvas: girado 15°, inclinado 45°, centrado."""
    x = (e * math.cos(THETA) - n * math.sin(THETA)) / MPP + W / 2
    y = -(e * math.sin(THETA) + n * math.cos(THETA)) * TILT / MPP + H / 2 - h * TILT / MPP
    return x, y


def _poly(pts_m, h=0.0):
    return [_to_px(e, n, h) for e, n in pts_m]


def _river_center(e: float) -> float:
    return -250 + 90 * math.sin(e / 260)


def build_render(rng: np.random.Generator) -> tuple[Image.Image, dict]:
    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    span = 1200
    # rio: faixa sinuosa de 60 m
    top = [(e, _river_center(e) + 30) for e in range(-span, span + 1, 20)]
    bot = [(e, _river_center(e) - 30) for e in range(span, -span - 1, -20)]
    river_m = top + bot
    d.polygon(_poly(river_m), fill=RIVER)
    parks = []
    buildings = []
    step, road_w = 120, 12
    for gx in range(-span, span, step):
        for gy in range(-span, span, step):
            x0, y0, x1, y1 = gx + road_w / 2, gy + road_w / 2, gx + step - road_w / 2, gy + step - road_w / 2
            cy = (y0 + y1) / 2
            if abs(cy - _river_center((x0 + x1) / 2)) < 75:
                continue  # beira-rio sem quadra
            if rng.random() < 0.12:
                parks.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                continue
            for _ in range(int(rng.integers(3, 7))):
                w_, h_ = rng.uniform(12, 30), rng.uniform(12, 30)
                bx, by = rng.uniform(x0, x1 - w_), rng.uniform(y0, y1 - h_)
                buildings.append(((bx, by, bx + w_, by + h_), float(rng.choice([4, 6, 7, 9, 12, 18, 24, 36])),
                                  ROOFS[int(rng.integers(len(ROOFS)))]))
    for p in parks:
        d.polygon(_poly(p), fill=PARK)
    # ruas por cima do chão (pontes sobre o rio)
    for g in range(-span, span + 1, step):
        d.polygon(_poly([(g - road_w / 2, -span), (g + road_w / 2, -span), (g + road_w / 2, span), (g - road_w / 2, span)]), fill=ROAD)
        if abs(g - (-250)) > 160:
            d.polygon(_poly([(-span, g - road_w / 2), (span, g - road_w / 2), (span, g + road_w / 2), (-span, g + road_w / 2)]), fill=ROAD)
    # prédios do fundo para a frente (maior y de tela por último)
    buildings.sort(key=lambda b: _to_px((b[0][0] + b[0][2]) / 2, (b[0][1] + b[0][3]) / 2)[1])
    for (x0, y0, x1, y1), h, roof in buildings:
        base = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        wall_a = tuple(int(c * 0.78) for c in (236, 230, 216))
        wall_b = tuple(int(c * 0.62) for c in (236, 230, 216))
        # duas paredes visíveis: as voltadas para a câmera (sul e leste, aproximadamente)
        for (a, b), col in (((base[0], base[1]), wall_a), ((base[1], base[2]), wall_b)):
            d.polygon([_to_px(*a), _to_px(*b), _to_px(*b, h), _to_px(*a, h)], fill=col)
        d.polygon(_poly(base, h), fill=roof)
    ll = lambda pts: [list(_ll(e, n)) for e, n in pts]  # noqa: E731
    return im, {"river_m": river_m, "river_ll": ll(river_m), "parks": parks}


def _ll(e: float, n: float) -> tuple[float, float]:
    return LAT0 + n / 110540, LON0 + e / (111320 * math.cos(math.radians(LAT0)))


def stylize(render: Image.Image, rng: np.random.Generator, truth: dict) -> Image.Image:
    """Simula a saída do modelo: pixel art de 3 px, contorno escuro, borrada, com derivas e lagoas."""
    k = 3
    small = render.resize((W // k, H // k), Image.BOX)
    pal = small.quantize(40, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
    a = np.asarray(pal, np.int16)
    g = np.asarray(pal.convert("L"), np.int16)
    edge = (np.abs(np.diff(g, axis=1, prepend=g[:, :1])) + np.abs(np.diff(g, axis=0, prepend=g[:1]))) > 18
    a[edge] = (a[edge] * 0.55).astype(np.int16)
    # grade exata de k px desde (0, 0): amplia para o múltiplo de k e completa a sobra com a borda
    big = Image.fromarray(a.clip(0, 255).astype(np.uint8)).resize((W // k * k, H // k * k), Image.NEAREST)
    art = Image.fromarray(np.pad(np.asarray(big), ((0, H - big.height), (0, W - big.width), (0, 0)), mode="edge"))
    # água inventada e terra no rio (antes do borrado, como o modelo faria)
    d = ImageDraw.Draw(art)
    lakes = []
    ren = np.asarray(render, np.int16)
    tries = 0
    while len(lakes) < 7 and tries < 400:
        tries += 1
        cx, cy = int(rng.integers(150, W - 150)), int(rng.integers(150, H - 150))
        rx, ry = int(rng.integers(40, 110)), int(rng.integers(25, 70))
        patch = ren[max(0, cy - ry - 30):cy + ry + 30, max(0, cx - rx - 30):cx + rx + 30]
        if ((patch[..., 2] - patch[..., 0]) > 60).mean() > 0.01:
            continue  # longe do rio de verdade
        col = (70, 130, 205) if len(lakes) % 2 == 0 else (60, 165, 160)
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)
        lakes.append([cx - rx, cy - ry, cx + rx, cy + ry])
    rm = truth["river_m"]
    e0 = -100
    land = [_to_px(e, _river_center(e) + off) for e, off in ((e0, -28), (e0 + 110, -28), (e0 + 110, 28), (e0, 28))]
    d.polygon(land, fill=PARK)
    xs, ys = [p[0] for p in land], [p[1] for p in land]
    truth["lakes_px"] = lakes
    truth["land_in_river_px"] = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    del rm
    # deriva de cor por bloco (as emendas)
    a = np.asarray(art, np.float32)
    drifts = {}
    for by in range((H + BLOCK - 1) // BLOCK):
        for bx in range((W + BLOCK - 1) // BLOCK):
            gain = np.array([1.0, 1.0, 1.0])
            off = np.array([0.0, 0.0, 0.0])
            kind = (bx + by * 2) % 4
            if kind == 1:
                off = np.array([14, 4, -10])        # mais quente
            elif kind == 2:
                gain = np.array([0.88, 0.88, 0.9])  # mais escuro
            elif kind == 3:
                gain = np.array([0.95, 1.04, 1.0])  # mais verde
            sl = (slice(by * BLOCK, (by + 1) * BLOCK), slice(bx * BLOCK, (bx + 1) * BLOCK))
            a[sl] = a[sl] * gain + off
            drifts[f"{bx},{by}"] = {"gain": gain.tolist(), "off": off.tolist()}
    truth["drifts"] = drifts
    art = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    # pixel mole: borrado leve + ruído, como a saída do VAE
    art = art.filter(ImageFilter.GaussianBlur(0.9))
    noise = rng.normal(0, 2.5, (H, W, 1))
    art = Image.fromarray((np.asarray(art, np.float32) + noise).clip(0, 255).astype(np.uint8))
    truth["pixel_scale"] = k
    return art


def build(root: Path = ROOT, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    run_dir = root / "runs" / NAME
    for sub in ("render", "art", "osm"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    render, truth = build_render(rng)
    art = stylize(render, rng, truth)
    skip = (2, 3)  # um shot sem arte: a oficina mostra a maquete
    for r in range(ROWS):
        for c in range(COLS):
            box = (c * SHOT, r * SHOT, (c + 1) * SHOT, (r + 1) * SHOT)
            render.crop(box).save(run_dir / "render" / f"r{r}_c{c}.png")
            if (r, c) != skip:
                art.crop(box).save(run_dir / "art" / f"r{r}_c{c}.png")
    view = {"name": NAME, "lat": LAT0, "lon": LON0, "azimuth_deg": -15, "elevation_deg": -45,
            "target_height_m": 185, "meters_per_px": MPP, "shot_px": SHOT, "source": "osm",
            "origin_px": [0, 0], "canvas_px": [W, H], "cols": COLS, "rows": ROWS,
            "area_px": [[0, 0], [W, 0], [W, H], [0, H]],
            "shots": [[c, r] for r in range(ROWS) for c in range(COLS)]}
    (run_dir / "view.json").write_text(json.dumps(view))
    # projeção lat/lon -> px (linear, exata para a cidade sintética)
    de, dn = 1.0, 1.0
    x0, y0 = _to_px(0, 0)
    xe, ye = _to_px(de, 0)
    xn, yn = _to_px(0, dn)
    m_per_lat, m_per_lon = 110540, 111320 * math.cos(math.radians(LAT0))
    aff = {"origin": [LAT0, LON0], "max_error_px": 0.0,
           "a": [[x0, (xn - x0) * m_per_lat, (xe - x0) * m_per_lon, 0, 0, 0],
                 [y0, (yn - y0) * m_per_lat, (ye - y0) * m_per_lon, 0, 0, 0]]}
    (run_dir / "geo_affine.json").write_text(json.dumps(aff))
    water = {"h": 0, "features": [{"id": "w-demo-rio", "kind": "water", "rings": [truth["river_ll"]], "holes": []}]}
    (run_dir / "osm" / "water_ll.json").write_text(json.dumps(water))
    water_px = {"h": 0, "features": [{"id": "w-demo-rio", "kind": "water",
                                      "rings": [[list(_to_px(e, n)) for e, n in truth["river_m"]]], "holes": []}]}
    (run_dir / "osm" / "water_px.json").write_text(json.dumps(water_px))
    places = [("Ponte da Amizade (demo)", 0, -250), ("Microcentro (demo)", -300, 250), ("Parque (demo)", 400, 300)]
    (run_dir / "osm" / "places_px.json").write_text(json.dumps({"points": [
        {"name": n, "kind": "landmark", "lat": _ll(e, nn)[0], "lon": _ll(e, nn)[1], "px": list(_to_px(e, nn))}
        for n, e, nn in places]}))
    truth.pop("river_m")
    truth.pop("parks")
    truth.update({"block_px": BLOCK, "no_art": [list(skip)]})
    (run_dir / "truth.json").write_text(json.dumps(truth))
    region = {"name": NAME, "title": "Cidade demo", "source": "osm", "center": {"lat": LAT0, "lon": LON0},
              "camera": {"azimuth_deg": -15, "elevation_deg": -45, "target_height_m": 185,
                         "meters_per_px": MPP, "shot_px": SHOT},
              "style": {"prompt": "Convert this 3D city model render into clean isometric pixel art.",
                        "negative": "photo, blurry"},
              "generate": {"window_px": 1024, "stride_px": 768, "block_px": BLOCK, "margin_px": 512,
                           "steps": 16, "true_cfg": 2.5, "seed": 1}}
    rf = root / "regions" / "demo.yaml"
    rf.parent.mkdir(parents=True, exist_ok=True)
    rf.write_text(yaml.safe_dump(region, allow_unicode=True, sort_keys=False))
    print(f"cidade demo pronta em {run_dir} ({COLS}x{ROWS} shots de {SHOT} px). Abra: tinymap edit {rf}")
    return rf
