"""Aumentos do treino (Isometric Mumbai): cor alterada e preenchimento com moldura vermelha."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinymap import augment as G  # noqa: E402
from tinymap.region import Run  # noqa: E402


def _par(seed=0, n=512):
    rng = np.random.default_rng(seed)
    r = Image.fromarray(rng.integers(40, 220, (n // 8, n // 8, 3), dtype=np.uint8)).resize((n, n), Image.NEAREST)
    a = r.quantize(12).convert("RGB")
    return r, a


def test_cor_muda_cor_mas_nao_o_desenho():
    r, _ = _par()
    c1, c2 = G.color_shift(r, 1), G.color_shift(r, 1)
    assert np.array_equal(np.asarray(c1), np.asarray(c2)), "determinístico"
    assert not np.array_equal(np.asarray(c1), np.asarray(G.color_shift(r, 2)))
    # o desenho (onde as bordas estão) não muda: bordas da luminância batem
    def edges(im):
        g = np.asarray(im.convert("L"), np.int16)
        return (np.abs(np.diff(g, axis=1)) > 0)
    e0, e1 = edges(r), edges(c1)
    assert (e0 == e1).mean() > 0.98
    assert np.abs(np.asarray(r, np.int16) - np.asarray(c1, np.int16)).mean() > 4, "a cor mudou de fato"


def test_preenchimento():
    r, a = _par(3)
    for k, kind in enumerate(G.KINDS):
        ctl = np.asarray(G.infill_example(r, a, kind, 10 + k), np.int16)
        x0, y0, x1, y1 = G.infill_box(a.size, kind, 10 + k)
        A, R = np.asarray(a, np.int16), np.asarray(r, np.int16)
        fora = np.ones(A.shape[:2], bool)
        fora[y0:y1, x0:x1] = False
        assert np.array_equal(ctl[fora], A[fora]), f"{kind}: fora da caixa é a arte"
        assert np.array_equal(ctl[y0 + 3:y1 - 3, x0 + 3:x1 - 3], R[y0 + 3:y1 - 3, x0 + 3:x1 - 3]), f"{kind}: dentro é a maquete"
        borda = ctl[y0 + 1, x0 + 2:x1 - 2] if kind != "bottom" else ctl[y1 - 2, x0 + 2:x1 - 2]
        assert (borda == [255, 0, 0]).all(), f"{kind}: moldura vermelha"
        area = (x1 - x0) * (y1 - y0) / (a.width * a.height)
        assert 0.15 < area < 0.55, (kind, area)


def test_build_com_aumentos():
    from tinymap import train
    import tinymap.region as R
    with tempfile.TemporaryDirectory() as t:
        R.ROOT = Path(t)
        run = Run({"name": "t", "style": {"prompt": "p"}}, Path(t) / "runs" / "t")
        pd = run.dir / "pairs_v1"
        pd.mkdir(parents=True)
        for i in range(4):
            r, a = _par(i, 1024)
            r.save(pd / f"f{i}_render.png")
            a.save(pd / f"f{i}_art.png")
        cfg = train.build(run, [pd], "r1", "Qwen/Qwen-Image-Edit-2511", 100, 16, color_copies=2, infill=1)
        info = json.loads((cfg.parent / "lab.json").read_text())
        assert info["pairs"] == 4 and info["examples"] == 4 * (1 + 2 + 1), info
        ds = cfg.parent / "dataset"
        assert len(list((ds / "control").glob("*.png"))) == 16 == len(list((ds / "target").glob("*.png")))
        assert "red box" in next((ds / "target").glob("*__fill0.txt")).read_text()


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok ", n)
