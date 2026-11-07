"""Testes do estúdio sem GPU, na cidade demo (defeitos plantados com gabarito).

    python tests/test_estudio.py        # ou: pytest tests/test_estudio.py
"""
import base64
import contextlib
import hashlib
import io
import json
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tinymap.region as R  # noqa: E402
from tinymap import autofix as A  # noqa: E402
from tinymap import demo  # noqa: E402
from tinymap import polish as P  # noqa: E402
from tinymap.mapedit import MapEdit, serve  # noqa: E402


def _cidade(tmp: Path):
    R.ROOT = tmp
    rf = demo.build(root=tmp)
    return R.load(rf), str(rf)


def _hashes(run):
    """Hash dos PIXELS de cada shot (o histórico regrava o PNG com outra compressão)."""
    return {p.name: hashlib.md5(Image.open(p).convert("RGBA").tobytes()).hexdigest()
            for p in sorted((run.dir / "art").glob("r*_c*.png"))}


def _px(p):
    return hashlib.md5(Image.open(p).convert("RGBA").tobytes()).hexdigest()


def _contagem(ed):
    ed.scan()
    kinds = ("água inventada", "terra no rio", "emenda", "borrado")
    return {k: sum(1 for i in ed.problems["items"] if i["kind"] == k) for k in kinds}


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def test_demo_deterministico():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ra, _ = _quiet(_cidade, Path(a))
        rb, _ = _quiet(_cidade, Path(b))
        assert _hashes(ra) == _hashes(rb)


def test_grade_e_polimento():
    with tempfile.TemporaryDirectory() as t:
        run, _ = _quiet(_cidade, Path(t))
        for p in sorted((run.dir / "art").glob("r*_c*.png")):
            r, c = (int(v[1:]) for v in p.stem.split("_"))
            g = P.detect_grid(Image.open(p))
            assert g["k"] == 3, (p.stem, g)
            assert (g["ox"], g["oy"]) == ((3 - c * 1024 % 3) % 3, (3 - r * 1024 % 3) % 3), (p.stem, g)
        for p in sorted((run.dir / "render").glob("r*_c*.png"))[:3]:
            assert P.detect_grid(Image.open(p))["k"] == 1, "render não tem grade"
        im = Image.open(run.dir / "art" / "r1_c1.png")
        pal = P.build_palette([im])
        out, info = P.polish_image(im, ["grid", "palette"], pal)
        assert info["depois"]["mole"] < 0.1 < info["antes"]["mole"]
        assert info["depois"]["cores"] * 4 < info["antes"]["cores"]


def test_fluxo_completo_e_desfazer():
    with tempfile.TemporaryDirectory() as t:
        run, rf = _quiet(_cidade, Path(t))
        original = _hashes(run)
        truth = json.loads((run.dir / "truth.json").read_text())
        antes = _contagem(MapEdit(run, rf))
        assert antes["água inventada"] >= len(truth["lakes_px"]) - 2, antes
        assert antes["terra no rio"] >= 1 and antes["emenda"] >= 10 and antes["borrado"] == 11, antes
        fx = _quiet(A.run_all, run, rf, lakes=True)
        assert fx["lagos_px"] > 0 and fx["rio_px"] > 0, fx
        po = _quiet(P.run_all, run, rf, True, True, True)
        depois = _contagem(MapEdit(run, rf))
        assert depois["água inventada"] == 0 and depois["terra no rio"] == 0 and depois["borrado"] == 0, depois
        assert depois["emenda"] <= 2, depois
        # desfazer tudo devolve exatamente os arquivos originais
        _quiet(P.undo_last, run, rf)
        _quiet(A.undo_last, run, rf)
        assert _hashes(run) == original
        assert po["shots"] == 11


def _porta():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _api(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def test_api_da_oficina():
    with tempfile.TemporaryDirectory() as t:
        run, rf = _quiet(_cidade, Path(t))
        ed = MapEdit(run, rf)
        port = _porta()
        threading.Thread(target=serve, args=(ed, "127.0.0.1", port), daemon=True).start()
        time.sleep(0.5)
        meta = _api(port, "/api/meta")
        assert meta["canvas"] == [4096, 3072] and len(meta["art"]) == 11
        box = [1024, 1024, 1536, 1536]
        j = _api(port, "/api/polish", {"box": box, "ops": ["grid", "palette"]})
        png = base64.b64decode(j["png"].split(",", 1)[1])
        assert Image.open(io.BytesIO(png)).size == (512, 512)
        assert j["info"]["grade"]["k"] == 3
        antes = _px(run.dir / "art" / "r1_c1.png")
        ap = _api(port, "/api/apply", {"box": box, "png": j["png"], "note": "teste"})
        assert _px(run.dir / "art" / "r1_c1.png") != antes
        _api(port, "/api/undo", {"id": ap["id"]})
        assert _px(run.dir / "art" / "r1_c1.png") == antes
        _api(port, "/api/autopolish", {"grid": True, "palette": False, "seams": False, "water": False})
        for _ in range(240):
            st = _api(port, "/api/autopolish")
            if st["status"] != "rodando":
                break
            time.sleep(0.5)
        assert st["status"] == "pronto" and st["n"] == 11, st


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            fn()
            print(f"ok  {name}  ({time.time() - t0:.0f}s)")
