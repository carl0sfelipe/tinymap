"""Oficina do mapa: navegar, achar defeitos, exportar recortes para refinar fora (Gemini etc.) e
reimportar no lugar, com histórico.

    tinymap edit regions/triplice-fronteira.yaml [--port 8766]

A arte fica em runs/<nome>/art/r{R}_c{C}.png (2048 px). Onde não há arte (a geração não chegou),
a oficina e o site mostram o render (maquete + satélite); um recorte importado ali cria a arte com
transparência no resto do shot. Toda aplicação guarda o pedaço antigo em art_history/ para desfazer.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import subprocess
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .region import ROOT, Run, render_canvas

UI = Path(__file__).with_name("mapedit_ui.html")

BASE_PROMPT = (
    "You are editing a crop of an isometric pixel-art city map in the style of a 1990s city-builder (SimCity 2000, "
    "Transport Tycoon): orthographic camera tilted 45 degrees and rotated 15 degrees, hard 1-pixel edges, flat colour "
    "fills from a small palette, dark 1-pixel outlines on roofs and walls, simple dithered shading, small pixel trees. "
    "Return ONLY the edited crop, at exactly {w} x {h} pixels, with the same framing, the same camera angle and the same "
    "palette, so it can be pasted back into the map seamlessly. No blur, no photographic texture, no text, no labels. "
    "Keep every building, street, tree line and river exactly where it is unless told otherwise."
)
IMAGES = {
    "improve": " Image 1 is the current map crop to edit. Image 2 is the reference: a 3D model of the same area with the "
               "real satellite photo on the ground and roofs; use it to check what is really there.",
    "generate": " Image 1 is a 3D model of the area with the real satellite photo on the ground and roofs: redraw it as a "
                "map crop. Image 2 is a finished piece of the same map: copy its style, palette, outline and shading "
                "exactly, but draw the buildings, streets and rivers of image 1.",
}
PRESETS = {
    "agua": "Remove invented water: there is no lake, pond, river or canal in this crop except where the reference shows "
            "blue water. Replace invented water with the streets, yards, fields, trees and roofs that the reference shows.",
    "predios": "Add rich pixel-art detail to every building: windows, doors, roof tiles or metal roof ribs, rooftop AC "
               "units and water tanks, small signs, balconies, chimneys. Keep each building's footprint, height and "
               "position exactly.",
    "verde": "Make the vegetation richer: fuller tree canopies with 2-3 shades and dithering, hedges, garden patches, "
             "grass texture. Do not move any building or street.",
    "ruas": "Add street detail: lane markings, sidewalks, crosswalks, small parked and moving cars, traffic lights, "
            "street trees. Keep the street network exactly as it is.",
    "vida": "Add small signs of life: tiny people, cars, buses, boats on the river, market stalls, lit shop fronts. Keep "
            "everything else exactly as it is.",
    "fiel": "Redraw this crop so it matches the reference exactly: same buildings, same street layout, same green "
            "areas and water, in the map's pixel-art style.",
}


class MapEdit:
    def __init__(self, run: Run, region_file: str):
        self.run, self.region_file = run, region_file
        self.v = run.view
        self.s = self.v["shot_px"]
        self.art = run.dir / "art"
        self.art.mkdir(exist_ok=True)
        self.thumbs = run.dir / ".thumbs"
        self.thumbs.mkdir(exist_ok=True)
        self.hist = run.dir / "art_history"
        self.hist.mkdir(exist_ok=True)
        self.rc = render_canvas(run)
        self.lock = threading.Lock()
        self.problems = {"status": "parado", "done": 0, "total": 0, "items": []}
        self.pub: subprocess.Popen | None = None
        self._aff = None

    # ---------- shots ----------
    def _src(self, layer: str, r: int, c: int) -> Path:
        return (self.art if layer == "art" else self.run.dir / "render") / f"r{r}_c{c}.png"

    def tile(self, layer: str, r: int, c: int, size: int) -> bytes | None:
        src = self._src(layer, r, c)
        if not src.exists():
            return None
        if size >= self.s:
            return src.read_bytes()
        t = self.thumbs / f"{layer}_{size}_r{r}_c{c}.png"
        if not t.exists() or t.stat().st_mtime < src.stat().st_mtime:
            im = Image.open(src)
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
            im.resize((size, size), Image.LANCZOS).save(t)
        return t.read_bytes()

    def _art_shot(self, r: int, c: int) -> Image.Image:
        p = self.art / f"r{r}_c{c}.png"
        if p.exists():
            return Image.open(p).convert("RGBA")
        return Image.new("RGBA", (self.s, self.s), (0, 0, 0, 0))

    def crop(self, layer: str, box: tuple[int, int, int, int]) -> Image.Image:
        """Recorte em px do canvas. art: RGBA (transparente onde não há arte); render: RGB."""
        if layer == "render":
            return self.rc.crop(box)
        if layer == "mixed":  # o que o mapa mostra: arte por cima do render
            base = self.rc.crop(box).convert("RGBA")
            base.alpha_composite(self.crop("art", box))
            return base.convert("RGB")
        x0, y0, x1, y1 = box
        out = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
        s = self.s
        for r in range(max(0, y0 // s), (y1 - 1) // s + 1):
            for c in range(max(0, x0 // s), (x1 - 1) // s + 1):
                p = self.art / f"r{r}_c{c}.png"
                if p.exists():
                    out.paste(Image.open(p).convert("RGBA"), (c * s - x0, r * s - y0))
        return out

    def _write(self, box, img: Image.Image) -> list[tuple[int, int]]:
        """Substitui exatamente os pixels da caixa (inclusive a transparência) nos shots da arte."""
        x0, y0, x1, y1 = box
        s, touched = self.s, []
        img = img.convert("RGBA")
        for r in range(max(0, y0 // s), (y1 - 1) // s + 1):
            for c in range(max(0, x0 // s), (x1 - 1) // s + 1):
                shot = self._art_shot(r, c)
                sx0, sy0 = max(x0, c * s), max(y0, r * s)
                sx1, sy1 = min(x1, (c + 1) * s), min(y1, (r + 1) * s)
                part = img.crop((sx0 - x0, sy0 - y0, sx1 - x0, sy1 - y0))
                shot.paste(part, (sx0 - c * s, sy0 - r * s))  # sem máscara: copia o alfa também
                p = self.art / f"r{r}_c{c}.png"
                tmp = p.with_name(f".{p.name}")
                if shot.getextrema()[3][0] == 255:  # opaco: salva RGB, como a geração
                    shot.convert("RGB").save(tmp)
                else:
                    shot.save(tmp)
                tmp.replace(p)
                touched.append((r, c))
        return touched

    # ---------- aplicar / desfazer ----------
    def apply(self, box, png: bytes, note: str) -> dict:
        new = Image.open(io.BytesIO(png)).convert("RGBA")
        assert new.size == (box[2] - box[0], box[3] - box[1]), (new.size, box)
        with self.lock:
            hid = time.strftime("%Y%m%d-%H%M%S") + f"-{box[0]}-{box[1]}"
            d = self.hist / hid
            d.mkdir()
            self.crop("art", box).save(d / "antes.png")
            new.save(d / "depois.png")
            (d / "meta.json").write_text(json.dumps({"box": box, "note": note, "t": time.time()}, ensure_ascii=False))
            touched = self._write(box, new)
        return {"id": hid, "shots": touched}

    def history(self) -> list[dict]:
        out = []
        for d in sorted(self.hist.iterdir(), reverse=True):
            m = d / "meta.json"
            if m.exists():
                j = json.loads(m.read_text())
                out.append({"id": d.name, **j, "undone": (d / "desfeito").exists()})
        return out[:200]

    def undo(self, hid: str) -> dict:
        d = self.hist / hid
        j = json.loads((d / "meta.json").read_text())
        with self.lock:
            touched = self._write(tuple(j["box"]), Image.open(d / "antes.png"))
            (d / "desfeito").write_text(str(time.time()))
        return {"shots": touched}

    # ---------- exportar ----------
    @staticmethod
    def prompt(box, mode: str, preset: str, text: str, ref) -> str:
        w, h = box[2] - box[0], box[3] - box[1]
        p = BASE_PROMPT.format(w=w, h=h) + IMAGES[mode]
        if mode == "improve" and ref:
            p += " Image 3 is a finished piece of the same map: match its style and palette."
        if preset in PRESETS:
            p += " " + PRESETS[preset]
        if text.strip():
            p += " " + text.strip()
        return p

    def export(self, box, ref, mode: str, preset: str, text: str) -> bytes:
        w, h = box[2] - box[0], box[3] - box[1]
        prompt = self.prompt(box, mode, preset, text, ref)
        render = self.crop("render", box)
        art = self.crop("mixed", box)
        pad = max(w, h) // 2
        cb = (max(0, box[0] - pad), max(0, box[1] - pad), box[2] + pad, box[3] + pad)
        ctx = self.crop("mixed", cb)
        ImageDraw.Draw(ctx).rectangle((box[0] - cb[0], box[1] - cb[1], box[2] - cb[0] - 1, box[3] - cb[1] - 1),
                                      outline=(255, 0, 255), width=max(3, w // 200))
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
            def put(name, im):
                b = io.BytesIO()
                im.save(b, "PNG")
                f.writestr(name, b.getvalue())
            if mode == "improve":
                put("1_mapa_atual.png", art)
                put("2_referencia_maquete.png", render)
            else:
                put("1_maquete.png", render)
            if ref:
                put("2_referencia_estilo.png" if mode == "generate" else "3_referencia_estilo.png",
                    self.crop("mixed", tuple(ref)))
            put("contexto.png", ctx.resize((min(1600, ctx.width), int(ctx.height * min(1600, ctx.width) / ctx.width))))
            f.writestr("prompt.txt", prompt + "\n")
            f.writestr("meta.json", json.dumps({"box": box, "ref": ref, "mode": mode, "preset": preset,
                                                "w": w, "h": h, "lat_lon": self.ll((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)},
                                               ensure_ascii=False, indent=1))
            f.writestr("LEIA.txt", "1) Envie as imagens numeradas ao Gemini junto com o texto de prompt.txt.\n"
                                   "2) Copie a imagem que ele devolver e cole (Ctrl+V) na oficina com a mesma seleção.\n"
                                   f"3) A oficina ajusta o tamanho para {w} x {h} px, alinha, ajusta a paleta e aplica.\n")
        return z.getvalue()

    # ---------- geo ----------
    def aff(self):
        if self._aff is None:
            from . import geo
            self._aff = geo.load(self.run)
        return self._aff

    def ll(self, x, y):
        from . import geo
        try:
            return [round(v, 6) for v in geo.to_ll(self.aff(), x, y)]
        except Exception:  # noqa: BLE001
            return None

    def meta(self) -> dict:
        from . import geo
        v = self.v
        cx, cy = v["canvas_px"][0] / 2, v["canvas_px"][1] / 2
        axes = None
        try:
            lat, lon = geo.to_ll(self.aff(), cx, cy)
            e = 1 / (111320 * math.cos(math.radians(lat)))  # 1 m para leste em graus
            n = 1 / 110540                                   # 1 m para norte
            p0 = geo.to_px(self.aff(), lat, lon)
            pe, pn = geo.to_px(self.aff(), lat, lon + e), geo.to_px(self.aff(), lat + n, lon)
            axes = {"east": [pe[0] - p0[0], pe[1] - p0[1]], "north": [pn[0] - p0[0], pn[1] - p0[1]],
                    "up": [0, -math.cos(math.radians(abs(v["elevation_deg"]))) / v["meters_per_px"]]}
        except Exception:  # noqa: BLE001
            pass
        places = []
        pp = self.run.dir / "osm" / "places_px.json"
        if pp.exists():
            places = [{"n": p["name"], "k": p.get("kind"), "x": p["px"][0], "y": p["px"][1]}
                      for p in json.loads(pp.read_text())["points"]]
        return {"name": self.run.name, "title": self.run.region.get("title"), "canvas": v["canvas_px"], "shot": self.s,
                "cols": v["cols"], "rows": v["rows"], "mpp": v["meters_per_px"], "area": v["area_px"],
                "shots": v["shots"], "art": sorted(p.stem for p in self.art.glob("r*_c*.png")),
                "axes": axes, "places": places, "presets": list(PRESETS), "geo": self.aff() if axes else None}

    def palette(self, n: int = 48) -> list[str]:
        """Paleta global da arte (amostra das miniaturas), para ajustar recortes importados."""
        cache = self.run.dir / ".edit_palette.json"
        if cache.exists() and time.time() - cache.stat().st_mtime < 3600:
            return json.loads(cache.read_text())
        count: dict[int, int] = {}
        for p in sorted(self.art.glob("r*_c*.png"))[::3]:
            a = np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.NEAREST)).reshape(-1, 3) >> 3
            k = (a[:, 0].astype(np.int32) << 10) | (a[:, 1].astype(np.int32) << 5) | a[:, 2]
            u, cnt = np.unique(k, return_counts=True)
            for kk, cc in zip(u.tolist(), cnt.tolist()):
                count[kk] = count.get(kk, 0) + cc
        top = sorted(count, key=count.get, reverse=True)[:n]
        pal = ["#%02x%02x%02x" % (((k >> 10) & 31) * 8 + 4, ((k >> 5) & 31) * 8 + 4, (k & 31) * 8 + 4) for k in top]
        cache.write_text(json.dumps(pal))
        return pal

    # ---------- caçador de lagos ----------
    def scan(self) -> None:
        """Procura água inventada e terra no rio em todos os shots da arte (em segundo plano)."""
        from .checks import _wet
        shots = sorted(self.art.glob("r*_c*.png"))
        self.problems = {"status": "procurando", "done": 0, "total": len(shots), "items": []}
        from PIL import ImageFilter
        items, s, cell, h = [], self.s, 64, self.s // 8   # em 1/8 da resolução: acha manchas de 64 px ou mais
        for i, p in enumerate(shots):
            r, c = (int(x[1:]) for x in p.stem.split("_"))
            art = Image.open(p).convert("RGBA").resize((h, h), Image.BOX)
            ren = self.rc.crop((c * s, r * s, (c + 1) * s, (r + 1) * s)).resize((h, h), Image.BOX)
            mixed = ren.convert("RGBA")
            mixed.alpha_composite(art)
            a = np.asarray(mixed.convert("RGB"), np.int16)
            # na arte vale azul ou turquesa (o modelo pinta lagoas turquesa que a regra estrita não via)
            wet_art = (a[..., 2] - a[..., 0] > 40) & (a[..., 2] > a[..., 1] - 35)
            wr = Image.fromarray(_wet(ren).astype(np.uint8) * 255)
            river = np.asarray(wr.filter(ImageFilter.MaxFilter(3))) > 127   # ~12 px de folga na resolução cheia
            core = np.asarray(wr.filter(ImageFilter.MinFilter(3))) > 127
            inv, land = wet_art & ~river, ~wet_art & core
            k = cell // 8
            for kind, m in (("água inventada", inv), ("terra no rio", land)):
                g = m.reshape(h // k, k, h // k, k).mean(axis=(1, 3)) > 0.15
                seen = np.zeros_like(g)
                for gy, gx in zip(*np.nonzero(g)):
                    if seen[gy, gx]:
                        continue
                    stack, pts = [(gy, gx)], []
                    seen[gy, gx] = True
                    while stack:
                        y, x = stack.pop()
                        pts.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < g.shape[0] and 0 <= xx < g.shape[1] and g[yy, xx] and not seen[yy, xx]:
                                seen[yy, xx] = True
                                stack.append((yy, xx))
                    ys, xs = [q[0] for q in pts], [q[1] for q in pts]
                    items.append({"kind": kind, "area": int(len(pts) * cell * cell),
                                  "box": [int(c * s + min(xs) * cell), int(r * s + min(ys) * cell),
                                          int(c * s + (max(xs) + 1) * cell), int(r * s + (max(ys) + 1) * cell)]})
            self.problems["done"] = i + 1
        poly = self.v["area_px"]

        def inside(x, y):
            ok, j = False, len(poly) - 1
            for i in range(len(poly)):
                (xi, yi), (xj, yj) = poly[i], poly[j]
                if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
                    ok = not ok
                j = i
            return ok
        items = [it for it in items if inside((it["box"][0] + it["box"][2]) / 2, (it["box"][1] + it["box"][3]) / 2)]
        items.sort(key=lambda x: -x["area"])
        self.problems = {"status": "pronto", "done": len(shots), "total": len(shots), "items": items[:400]}

    # ---------- publicar ----------
    def publish(self) -> None:
        if self.pub is not None and self.pub.poll() is None:
            raise RuntimeError("já está publicando")
        log = open(self.run.dir / "publish.log", "w")
        self.pub = subprocess.Popen([sys.executable, "-m", "tinymap.cli", "site", str(self.region_file)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    def pub_status(self) -> dict:
        if self.pub is None:
            return {"running": False, "code": None}
        return {"running": self.pub.poll() is None, "code": self.pub.poll()}


def serve(ed: MapEdit, host: str, port: int) -> None:
    rx_tile = re.compile(r"^/tile/(art|render)/(\d+)/(\d+)/(\d+)$")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _q(self):
            from urllib.parse import parse_qs, urlparse
            u = urlparse(self.path)
            return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}

        def do_GET(self):
            path, q = self._q()
            try:
                if path == "/":
                    return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
                if path == "/api/meta":
                    return self._json(ed.meta())
                if path == "/api/palette":
                    return self._json(ed.palette())
                if path == "/api/history":
                    return self._json(ed.history())
                if path == "/api/problems":
                    return self._json(ed.problems)
                if path == "/api/publish":
                    return self._json(ed.pub_status())
                m = rx_tile.match(path)
                if m:
                    b = ed.tile(m[1], int(m[2]), int(m[3]), int(m[4]))
                    return self._send(200, b, "image/png") if b else self._send(204, b"", "image/png")  # fora da área
                if path == "/api/crop":
                    box = tuple(int(q[k]) for k in ("x0", "y0", "x1", "y1"))
                    im = ed.crop(q.get("layer", "art"), box)
                    b = io.BytesIO()
                    im.save(b, "PNG")
                    return self._send(200, b.getvalue(), "image/png")
                return self._send(404, b"not found", "text/plain")
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 500)

        def do_POST(self):
            path, _ = self._q()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if path == "/api/export":
                    z = ed.export(tuple(body["box"]), body.get("ref"), body.get("mode", "improve"),
                                  body.get("preset", ""), body.get("text", ""))
                    name = f"tinymap_{body['box'][0]}_{body['box'][1]}.zip"
                    return self._send(200, z, "application/zip", {"Content-Disposition": f'attachment; filename="{name}"'})
                if path == "/api/prompt":
                    return self._json({"prompt": ed.prompt(tuple(body["box"]), body.get("mode", "improve"),
                                                           body.get("preset", ""), body.get("text", ""), body.get("ref"))})
                if path == "/api/apply":
                    png = base64.b64decode(body["png"].split(",", 1)[1])
                    return self._json(ed.apply(tuple(body["box"]), png, body.get("note", "")))
                if path == "/api/undo":
                    return self._json(ed.undo(body["id"]))
                if path == "/api/problems":
                    if ed.problems["status"] != "procurando":
                        threading.Thread(target=ed.scan, daemon=True).start()
                    return self._json({"ok": True})
                if path == "/api/publish":
                    ed.publish()
                    return self._json({"ok": True})
                return self._json({"error": "rota desconhecida"}, 404)
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)

    srv = ThreadingHTTPServer((host, port), H)
    print(f"oficina do mapa em http://{host}:{port}  (arte em {ed.art}; Ctrl+C para parar)")
    srv.serve_forever()
