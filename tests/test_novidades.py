"""Técnicas de nov/2026: tom global (Toronto), forma da cidade (Tirana), avisos do público (Warsaw),
teto de gasto (SF). Sem GPU, na cidade demo.

    python tests/test_novidades.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tinymap.region as R  # noqa: E402
from tinymap import demo  # noqa: E402
from tinymap import polish as P  # noqa: E402
from tinymap.mapedit import MapEdit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _cidade(tmp: Path):
    R.ROOT = tmp
    rf = demo.build(root=tmp)
    return R.load(rf), str(rf)


def test_global_tone_recupera_escurecimento():
    rng = np.random.default_rng(0)
    pares, mapa = [], []
    for i in range(12):
        ren = Image.fromarray(rng.integers(40, 220, (128, 128, 3), dtype=np.uint8))
        art = Image.fromarray(np.clip(np.asarray(ren, np.float32) * 0.9 + 15, 0, 255).astype(np.uint8))
        pares.append((art, ren))
        escuro = Image.fromarray(np.clip(np.asarray(art, np.float32) - 18, 0, 255).astype(np.uint8))
        mapa.append((escuro, ren))
    g, o = P.global_tone(pares, mapa)
    assert np.allclose(g, 1, atol=0.03), g
    assert np.allclose(o, 18, atol=3), o
    # nunca passa dos limites (ajuste global não pode virar outra paleta)
    g2, o2 = P.global_tone(pares, [(Image.new("RGB", (64, 64), (0, 0, 0)), r) for _, r in pares])
    assert (np.abs(g2 - 1) <= 0.15 + 1e-6).all() and (np.abs(o2) <= 20 + 1e-6).all()


def test_tom_e_forma_no_demo_e_desfaz():
    with tempfile.TemporaryDirectory() as t:
        run, rf = _cidade(Path(t))
        # pares "do professor": as próprias janelas do mapa, mais claras
        ed = MapEdit(run, rf)
        pd = run.dir / "pairs_v1"
        pd.mkdir()
        for i, (art, ren) in enumerate(P.tone_windows(ed, run, n=6, size=512)):
            Image.fromarray(np.clip(np.asarray(art, np.int16) + 12, 0, 255).astype(np.uint8)).save(pd / f"f{i}_art.png")
            ren.save(pd / f"f{i}_render.png")
        antes = {p.name: np.asarray(Image.open(p).convert("RGBA")).copy() for p in (run.dir / "art").glob("r*_c*.png")}
        tot = P.run_all(run, rf, False, False, False, tone=True)
        assert tot["tom"]["desloc"][0] > 5, tot
        P.undo_last(run, rf)
        for p in (run.dir / "art").glob("r*_c*.png"):
            assert (np.asarray(Image.open(p).convert("RGBA")) == antes[p.name]).all()

        # forma: um triângulo no canto; fora dele vira transparente
        from tinymap import geo
        aff = geo.load(run)
        W, H = run.view["canvas_px"]
        tri = [geo.to_ll(aff, 0, 0), geo.to_ll(aff, W * 0.6, 0), geo.to_ll(aff, 0, H * 0.6)]
        run.region["shape"] = [list(v) for v in tri]
        tot = P.run_all(run, rf, False, False, False, shape=True)
        assert tot["forma_shots"] > 0
        ed = MapEdit(run, rf)
        canto = ed.crop("art", (W - 200, H - 200, W - 100, H - 100))
        assert canto.mode != "RGBA" or np.asarray(canto.getchannel("A")).max() == 0
        dentro = ed.crop("art", (20, 20, 120, 120))
        assert np.asarray(dentro.convert("RGBA").getchannel("A")).min() == 255


def test_avisos_viram_itens_do_estudio():
    with tempfile.TemporaryDirectory() as t:
        run, rf = _cidade(Path(t))
        f = Path(t) / "avisos_export.json"
        f.write_text(json.dumps([{"x": 800, "y": 700, "texto": "prédio demolido"},
                                 {"x": 800, "y": 700, "texto": "prédio demolido"},
                                 {"x": 900, "y": 900, "texto": "<b>ok</b>", "resolvido": True}]))
        from tinymap.mapedit import importar_avisos
        assert importar_avisos(run, f) == 2
        assert len(json.loads((run.dir / "avisos.json").read_text())) == 2
        ed = MapEdit(run, rf)
        ed.scan()
        av = [it for it in ed.problems["items"] if it["kind"] == "aviso do público"]
        assert len(av) == 1 and av[0]["texto"] == "prédio demolido"
        assert ed.problems["items"][0]["kind"] == "aviso do público"


def test_budget_guard_para_no_teto():
    # um job falso do tinymap: o guarda tem que matar este e só este
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "tinymap.cli"])
    with tempfile.TemporaryDirectory() as t:
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "budget_guard.py"), "--max-usd", "1",
                              "--usd-hour", "0.5", "--spent", "1.2", "--every", "1"],
                             capture_output=True, text=True, timeout=90, env={"PATH": "/usr/bin:/bin", "HOME": t})
        assert out.returncode == 0, out.stderr
        assert "TETO" in out.stdout
    assert job.wait(timeout=5) != 0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
