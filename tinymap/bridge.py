"""Ponte ao vivo: conta veículos e pedestres nas câmeras da ponte e envia ao now.services.

Roda no rig de GPU, junto com o resto do tinymap:

    tinymap bridge regions/triplice-fronteira.yaml --cameras cameras.yaml

Só use câmeras com autorização escrita de quem as opera (ADR 0006 do now.services, decisão 6).

Como mede (sem rastrear ninguém, sem guardar imagem):
  1. A cada `sample_seconds`, pega um quadro da câmera e roda um detector YOLO (classes COCO:
     car, motorcycle, bus, truck, person).
  2. Cada câmera tem polígonos de faixa por sentido (`lanes`), desenhados uma vez com
     `tinymap bridge-calibrate`. Um veículo conta para o sentido cuja faixa contém o ponto de
     contato dele com o chão (meio da borda inferior da caixa).
  3. Fila = fração ocupada da faixa x comprimento real da faixa em metros (`length_m`).
  4. Tempo de carro = fila em veículos / vazão da aduana (veículos por minuto, `discharge_per_min`),
     mais o tempo de rodar a ponte. A vazão é o número a calibrar: cronometre 3-4 carros reais da
     entrada da fila até a saída da aduana em horários diferentes e ajuste até bater.
  5. A cada minuto, envia a mediana do minuto para POST /api/mapa/ponte.

Nada de imagem sai da máquina: só contagens.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

COCO = {2: "cars", 3: "motorcycles", 5: "buses", 7: "trucks", 0: "pedestrians"}


@dataclass
class Lane:
    direction: str            # BR_PY | PY_BR
    polygon: list[list[float]]  # pixels da câmera
    length_m: float           # comprimento real coberto pela faixa
    discharge_per_min: float  # veículos que a aduana libera por minuto (calibrar)
    bridge_minutes: float = 3.0  # rodar a ponte sem fila
    walk_base_minutes: float = 10.0


@dataclass
class Camera:
    id: str
    url: str
    lanes: list[Lane]
    car_length_m: float = 6.0
    samples: list[dict] = field(default_factory=list)


def load_cameras(path: Path) -> list[Camera]:
    cfg = yaml.safe_load(path.read_text())
    cams = []
    for c in cfg["cameras"]:
        if not c.get("authorized_by"):
            raise SystemExit(f"câmera {c['id']}: preencha authorized_by (quem autorizou e quando) antes de usar")
        cams.append(Camera(id=c["id"], url=c["url"], car_length_m=c.get("car_length_m", 6.0),
                           lanes=[Lane(**ln) for ln in c["lanes"]]))
    return cams


def _inside(poly: np.ndarray, x: float, y: float) -> bool:
    n, c, j = len(poly), False, len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
            c = not c
        j = i
    return c


def grab(url: str):
    import cv2
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def measure(cam: Camera, frame, model) -> dict:
    res = model.predict(frame, verbose=False, conf=0.25, classes=list(COCO))[0]
    out = {ln.direction: {k: 0 for k in COCO.values()} for ln in cam.lanes}
    polys = [np.array(ln.polygon, dtype=np.float32) for ln in cam.lanes]
    for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
        x = (box[0] + box[2]) / 2
        y = box[3]  # ponto de contato com o chão
        for ln, poly in zip(cam.lanes, polys):
            if _inside(poly, x, y):
                out[ln.direction][COCO[cls]] += 1
                break
    return out


def estimate(lane: Lane, counts: dict, car_length_m: float) -> dict:
    vehicles = counts["cars"] + counts["trucks"] * 2 + counts["buses"] * 2
    # moto ocupa ~1/3 de carro na fila e quase sempre passa pelo corredor
    queue_m = min(lane.length_m, (vehicles + counts["motorcycles"] / 3) * car_length_m)
    queued = queue_m / car_length_m
    car_min = lane.bridge_minutes + queued / max(lane.discharge_per_min, 0.1)
    return {
        "queueMeters": round(queue_m),
        "carMinutes": round(car_min),
        "motoMinutes": round(lane.bridge_minutes + 0.25 * (car_min - lane.bridge_minutes)),
        "walkMinutes": round(lane.walk_base_minutes + counts["pedestrians"] / 20),
    }


def post(endpoint: str, token: str, readings: list[dict]) -> None:
    req = urllib.request.Request(endpoint, data=json.dumps(readings).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        print("enviado:", r.read().decode())


def run(cameras_file: Path, sample_seconds: int = 10, weights: str = "yolo11m.pt", device: str = "cuda:0",
        dry_run: bool = False) -> None:
    from ultralytics import YOLO

    endpoint = os.environ.get("NOW_PONTE_URL", "https://now.services/api/mapa/ponte")
    token = os.environ.get("PONTE_INGEST_TOKEN", "")
    if not token and not dry_run:
        raise SystemExit("defina PONTE_INGEST_TOKEN (o mesmo do now.services) ou use --dry-run")
    model = YOLO(weights).to(device)
    cams = load_cameras(cameras_file)
    pending: list[dict] = []
    minute = int(time.time() // 60)
    while True:
        for cam in cams:
            frame = grab(cam.url)
            if frame is None:
                print(f"{cam.id}: sem quadro")
                continue
            cam.samples.append(measure(cam, frame, model))
        now_min = int(time.time() // 60)
        if now_min != minute:
            taken = datetime.fromtimestamp(minute * 60, timezone.utc).isoformat()
            for cam in cams:
                for ln in cam.lanes:
                    rows = [s[ln.direction] for s in cam.samples if ln.direction in s]
                    if not rows:
                        continue
                    counts = {k: round(statistics.median(r[k] for r in rows)) for k in COCO.values()}
                    pending.append({"direction": ln.direction, "source": cam.id, "takenAt": taken,
                                    **counts, **estimate(ln, counts, cam.car_length_m)})
                cam.samples.clear()
            minute = now_min
            if dry_run:
                print(json.dumps(pending, indent=1))
                pending.clear()
            elif pending:
                try:
                    post(endpoint, token, pending[-120:])
                    pending.clear()
                except Exception as e:  # noqa: BLE001 — rede caiu: guarda e reenvia (a API ignora duplicatas)
                    print("falha ao enviar, tento no próximo minuto:", e)
                    pending = pending[-120:]
        time.sleep(sample_seconds)


def calibrate(cameras_file: Path, camera_id: str, out_png: Path) -> None:
    """Salva um quadro com as faixas desenhadas por cima, para conferir/ajustar os polígonos."""
    import cv2
    cam = next(c for c in load_cameras(cameras_file) if c.id == camera_id)
    frame = grab(cam.url)
    if frame is None:
        raise SystemExit("sem quadro da câmera")
    colors = {"BR_PY": (0, 200, 255), "PY_BR": (255, 120, 0)}
    for ln in cam.lanes:
        pts = np.array(ln.polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], True, colors[ln.direction], 3)
        cv2.putText(frame, ln.direction, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[ln.direction], 2)
    h, w = frame.shape[:2]
    for x in range(0, w, 100):
        cv2.putText(frame, str(x), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for y in range(0, h, 100):
        cv2.putText(frame, str(y), (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out_png), frame)
    print(out_png)
