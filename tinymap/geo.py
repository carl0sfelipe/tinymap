"""lat/lon -> pixel do mapa sem depender do three.js em tempo de execução.

Projeta uma grade de pontos com a câmera real (render/project.mjs) e ajusta um polinômio de grau 2
em (lat, lon) por mínimos quadrados. Uma afim pura erra ~11 px nos cantos de 15 km (curvatura da
Terra); o termo quadrático leva o erro a menos de 1 px. O erro medido fica gravado no arquivo.
O viewer (site/index.html) avalia o mesmo polinômio.
"""

from __future__ import annotations

import json

import numpy as np

from .region import Run, node


def fit(run: Run, n: int = 12) -> dict:
    b = run.region["bbox"]
    h = run.region["camera"]["target_height_m"]
    pts = [{"name": f"{i},{j}", "lat": b["south"] + (b["north"] - b["south"]) * i / (n - 1),
            "lon": b["west"] + (b["east"] - b["west"]) * j / (n - 1), "h": h}
           for i in range(n) for j in range(n)]
    src, dst = run.path("osm", "grid_ll.json"), run.path("osm", "grid_px.json")
    src.write_text(json.dumps({"points": pts}))
    node("project.mjs", run.dir, src, dst)
    proj = json.loads(dst.read_text())["points"]
    lat0, lon0 = run.region["center"]["lat"], run.region["center"]["lon"]
    A = np.array([_terms(p["lat"] - lat0, p["lon"] - lon0) for p in proj])
    X = np.array([p["px"] for p in proj])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)  # 6x2
    err = np.abs(A @ coef - X).max()
    out = {"origin": [lat0, lon0], "a": coef.T.tolist(), "max_error_px": round(float(err), 3)}
    run.path("geo_affine.json").write_text(json.dumps(out))
    print(f"projeção lat/lon -> px ajustada, erro máximo {err:.2f} px")
    return out


def load(run: Run) -> dict:
    p = run.dir / "geo_affine.json"
    return json.loads(p.read_text()) if p.exists() else fit(run)


def _terms(dl: float, dn: float) -> list[float]:
    return [1.0, dl, dn, dl * dl, dl * dn, dn * dn]


def to_px(aff: dict, lat: float, lon: float) -> tuple[float, float]:
    t = _terms(lat - aff["origin"][0], lon - aff["origin"][1])
    return tuple(sum(c * v for c, v in zip(row, t)) for row in aff["a"])  # type: ignore[return-value]
