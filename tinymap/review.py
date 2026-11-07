"""Revisão humana dos pares de treino (human in the loop), numa página local.

    tinymap review regions/triplice-fronteira.yaml --pilot [--out pairs_v1] [--port 8765]

Para cada recorte: a maquete, os candidatos do professor (com as notas dos checks como sugestão) e
um editor de pixel art para corrigir à mão. Aprovar grava <id>_art.png (só o que você aprova vira par
de treino); "refazer" guarda uma dica que entra no prompt do próximo lote; "descartar" tira o recorte
da fila. O botão "gerar próximo lote" roda `modal run tinymap/modal_app.py::teacher` em segundo plano
e os candidatos novos aparecem na página conforme chegam.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import teacher as T
from .region import ROOT, Run

UI = Path(__file__).with_name("review_ui.html")
SAFE = re.compile(r"^[A-Za-z0-9_.-]+\.png$")
SECONDS_PER_IMAGE = 7.4      # Qwen-Image-2.1, 40 passos, 1024 px, H100 (medido no piloto)
H100_PER_HOUR = 3.95         # US$ na Modal
LOAD_SECONDS = 90            # carregar o modelo, por container


class Review:
    def __init__(self, run: Run, out: str, region_file: str, pilot: bool):
        self.run, self.d, self.region_file, self.pilot = run, run.dir / out, region_file, pilot
        self.out = out
        self.lock = threading.Lock()
        self.gen: subprocess.Popen | None = None
        self.water = None
        self._aff = None

    def ll(self, box) -> list[float] | None:
        """Centro do recorte em lat/lon (para o mapa de satélite ao lado)."""
        from . import geo
        try:
            if self._aff is None:
                self._aff = geo.load(self.run)
            lat, lon = geo.to_ll(self._aff, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            return [round(lat, 6), round(lon, 6)]
        except Exception:  # noqa: BLE001  (sem geo_affine.json e sem node: segue sem mapa)
            return None

    # ---------- estado ----------
    def _review(self) -> dict:
        return T.load_review(self.d)

    def _save_review(self, rv: dict) -> None:
        tmp = self.d / ".review.json"
        tmp.write_text(json.dumps(rv, indent=1, ensure_ascii=False))
        tmp.replace(self.d / "review.json")

    def state(self) -> dict:
        frames = json.loads((self.d / "frames.json").read_text()) if (self.d / "frames.json").exists() else []
        checks = json.loads((self.d / "checks.json").read_text()) if (self.d / "checks.json").exists() else {}
        rv = self._review()
        out = []
        for fr in frames:
            fid = fr["id"]
            r = rv.get(fid, {})
            cands = [{"name": c.name, "fails": checks.get(c.name, {}).get("fails"), "m": checks.get(c.name, {}).get("m"),
                      "t": c.stat().st_mtime} for c in T.candidates(self.d, fid)]
            art = self.d / f"{fid}_art.png"
            out.append({"id": fid, "box": fr["box"], "ll": self.ll(fr["box"]),
                        "m": round((fr["box"][2] - fr["box"][0]) * self.run.view["meters_per_px"]),
                        "round": fr.get("round", 0), "status": r.get("status", "todo"),
                        "chosen": r.get("chosen"), "hint": r.get("hint", ""), "edited": r.get("edited", False),
                        "art": art.exists() and r.get("status") == "approved",
                        "art_t": art.stat().st_mtime if art.exists() else 0, "cands": cands})
        cam = self.run.region["camera"]
        return {"pairs": self.out, "pilot": self.pilot, "frames": out, "gen": self.gen_status(),
                "camera": {"azimuth": cam["azimuth_deg"], "elevation": cam["elevation_deg"]},
                "cost": {"sec_per_image": SECONDS_PER_IMAGE, "h100_per_hour": H100_PER_HOUR, "load_s": LOAD_SECONDS}}

    def decide(self, fid: str, status: str, chosen: str | None = None, hint: str | None = None) -> None:
        assert status in ("todo", "approved", "redo", "discard"), status
        with self.lock:
            rv = self._review()
            r = rv.setdefault(fid, {})
            r["status"] = status
            if chosen is not None:
                r["chosen"] = chosen
            if hint is not None:
                r["hint"] = hint
            r["t"] = time.time()
            self._save_review(rv)
        if status != "approved":
            (self.d / f"{fid}_art.png").unlink(missing_ok=True)  # só aprovado vira par de treino

    def save_art(self, fid: str, data_url: str, chosen: str | None, edited: bool) -> None:
        png = base64.b64decode(data_url.split(",", 1)[1])
        img = Image.open(BytesIO(png)).convert("RGB")
        img.save(self.d / f"{fid}_art.png")  # 1024 px, como desenhado (o treino ajusta o render)
        with self.lock:
            rv = self._review()
            r = rv.setdefault(fid, {})
            r.update(status="approved", chosen=chosen, edited=edited, t=time.time())
            self._save_review(rv)

    def water_png(self, fid: str) -> bytes:
        from .osm import Water
        frames = {fr["id"]: fr for fr in json.loads((self.d / "frames.json").read_text())}
        if self.water is None:
            self.water = Water(self.run)
        m = self.water.mask(tuple(frames[fid]["box"])).resize((T.FRAME, T.FRAME), Image.NEAREST)
        b = BytesIO()
        m.save(b, "PNG")
        return b.getvalue()

    # ---------- próximo lote ----------
    def gen_status(self) -> dict:
        log = self.d / "gen.log"
        tail = log.read_text(errors="replace").splitlines()[-6:] if log.exists() else []
        running = self.gen is not None and self.gen.poll() is None
        return {"running": running, "code": None if running or self.gen is None else self.gen.returncode, "log": tail}

    def generate(self, cands: int, new: int, gpus: int) -> None:
        if self.gen is not None and self.gen.poll() is None:
            raise RuntimeError("já tem um lote rodando")
        cmd = [sys.executable, "-m", "modal", "run", "tinymap/modal_app.py::teacher", "--region", self.region_file,
               "--out", self.out, "--cands", str(cands), "--new", str(new), "--gpus", str(gpus)]
        if self.pilot:
            cmd.append("--pilot")
        log = open(self.d / "gen.log", "w")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        self.gen = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def serve(rev: Review, host: str, port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silencioso
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._json(rev.state())
                elif path.startswith("/img/") and SAFE.match(path[5:]) and (rev.d / path[5:]).exists():
                    self._send(200, (rev.d / path[5:]).read_bytes(), "image/png")
                elif path.startswith("/water/"):
                    self._send(200, rev.water_png(path[7:]), "image/png")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, 500)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/decide":
                    rev.decide(body["id"], body["status"], body.get("chosen"), body.get("hint"))
                elif self.path == "/api/art":
                    rev.save_art(body["id"], body["png"], body.get("chosen"), bool(body.get("edited")))
                elif self.path == "/api/generate":
                    rev.generate(int(body.get("cands", 4)), int(body.get("new", 0)), int(body.get("gpus", 4)))
                else:
                    return self._json({"error": "rota desconhecida"}, 404)
                self._json({"ok": True})
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, 400)

    srv = ThreadingHTTPServer((host, port), H)
    print(f"revisão em http://{host}:{port}  (pares em {rev.d}; Ctrl+C para parar)")
    srv.serve_forever()
