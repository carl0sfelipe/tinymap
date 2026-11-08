"""Fichas de fachada: ângulo, sugestão de andares/janelas, maquete, Street View com teto, estúdio.

    python tests/test_fachada.py
"""
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tinymap.region as R  # noqa: E402
from tinymap import demo  # noqa: E402
from tinymap import fachada as F  # noqa: E402
from tinymap.mapedit import MapEdit  # noqa: E402


def _facade(levels=6, cols=8, W=800, H=600):
    im = Image.new("RGB", (W, H), (205, 190, 160))
    d = ImageDraw.Draw(im)
    fh, cw = H / levels, W / cols
    for i in range(levels):
        for j in range(cols):
            x, y = j * cw + cw * 0.25, i * fh + fh * 0.25
            d.rectangle([x, y, x + cw * 0.5, y + fh * 0.5], fill=(40, 50, 60))
    return im


def test_angulo_enquadra_o_predio():
    cam = (-19.9200, -43.9386)
    bld = (-19.9200 + 30 / 111_000, -43.9386)  # 30 m ao norte
    c = F.camera_for(cam, bld, height_m=20, width_m=15)
    assert abs(c["heading"]) < 1 or abs(c["heading"] - 360) < 1
    assert 25 < c["dist_m"] < 35 and c["pitch"] > 0 and 30 <= c["fov"] <= 120
    leste = F.camera_for(cam, (-19.92, -43.9386 + 30 / 104_000), 20)
    assert abs(leste["heading"] - 90) < 2


def test_sugere_andares_e_janelas():
    for lv, cols in ((6, 8), (4, 5), (10, 12)):
        s = F.suggest(_facade(lv, cols))
        assert s["levels"] == lv and s["cols"] == cols, (lv, cols, s)
        assert s["confianca"] > 0.3
    assert F.suggest(_facade())["wall"].startswith("#")


def _cidade(tmp):
    R.ROOT = tmp
    rf = demo.build(root=tmp)
    return R.load(rf), str(rf)


def test_ficha_vira_janelas_na_maquete_e_prompt():
    with tempfile.TemporaryDirectory() as t:
        run, rf = _cidade(Path(t))
        ed = MapEdit(run, rf)
        x, y = 1500, 1200
        lat, lon = ed.ll(x, y)
        d = 0.00012
        ring = [[lat - d, lon - d], [lat - d, lon + d], [lat + d, lon + d], [lat + d, lon - d], [lat - d, lon - d]]
        (run.dir / "osm").mkdir(exist_ok=True)
        run.path("osm", "scene_ll.json").write_text(json.dumps({"buildings": [{"id": 7, "rings": [ring], "h": 9}]}))
        b = io.BytesIO()
        _facade(6, 8).save(b, "PNG")
        f = ed.fachada_foto(x, y, b.getvalue())
        assert f["levels"] == 6 and f["cols"] == 8 and f["status"] == "sugerida"
        assert F.apply_to_scene(run) == 0  # só fichas confirmadas
        f = ed.fachada_salvar({"id": f["id"], "levels": 7, "cols": 8, "shop": True, "status": "confirmada"})
        run.region["source"] = "google"  # sem refazer render no teste
        r = ed.fachada_aplicar()
        assert r["predios"] == 1 and r["render"] is False  # com 3D do Google: só a cena e a referência
        bb = json.loads(run.path("osm", "scene_ll.json").read_text())["buildings"][0]
        assert bb["win"]["levels"] == 7 and bb["win"]["shop"] and bb["h"] == 21.6 and bb["fachada"] == f["id"]
        z = zipfile.ZipFile(io.BytesIO(ed.export((x - 200, y - 200, x + 200, y + 200), None, "improve", "", "")))
        assert any(n.startswith("foto_fachada_") for n in z.namelist())
        p = z.read("prompt.txt").decode()
        assert "7 floors" in p and "8 windows per floor" in p and "awnings" in p


def test_streetview_respeita_teto():
    with tempfile.TemporaryDirectory() as t:
        run, rf = _cidade(Path(t))
        lat, lon = MapEdit(run, rf).ll(1500, 1200)
        img = io.BytesIO()
        _facade(5, 6, 640, 640).save(img, "JPEG")
        chamadas = []

        def fake(url, timeout=30):
            chamadas.append(url)
            if "metadata" in url:
                return json.dumps({"status": "OK", "pano_id": "P1", "date": "2024-05", "copyright": "© Google",
                                   "location": {"lat": lat - 0.0003, "lng": lon}}).encode()
            return img.getvalue()
        F._get = fake
        run.region["fachada"] = {"streetview_max_mes": 1}
        env = {"GOOGLE_MAPS_API_KEY": "x"}
        f = F.fetch(run, lat, lon, "streetview", env)
        assert f["fonte"] == "streetview" and f["pano"] == "P1" and abs(f["heading"]) < 2 or abs(f["heading"] - 360) < 2
        assert f["levels"] == 5
        try:
            F.fetch(run, lat, lon, "streetview", env)
            raise AssertionError("passou do teto")
        except RuntimeError as e:
            assert "teto" in str(e)
        assert sum("metadata" not in u for u in chamadas) == 1  # só uma imagem paga
        run.region["fachada"] = {"streetview_max_mes": 0}
        try:
            F.fetch(run, lat, lon, "streetview", env)
            raise AssertionError("devia estar desligado")
        except RuntimeError as e:
            assert "desligado" in str(e)


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
