"""Testes do laboratório sem GPU: LPIPS trocado por uma distância simples, pintores falsos."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinymap import lab  # noqa: E402
from tinymap.region import Run  # noqa: E402


def fake_lpips(a, b):
    x = np.asarray(a.convert("RGB").resize((128, 128)), np.float32) / 255
    y = np.asarray(b.convert("RGB").resize((128, 128)), np.float32) / 255
    return float(np.abs(x - y).mean() * 10)


def cidade(seed, size=1024):
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (size, size), (190, 185, 170))
    d = ImageDraw.Draw(im)
    for y in range(0, size, 160):
        d.rectangle([0, y, size, y + 18], fill=(90, 90, 90))
    for _ in range(40):
        x, y = rng.integers(0, size - 120, 2)
        d.rectangle([x, y, x + 90, y + 70], fill=tuple(int(c) for c in rng.integers(60, 220, 3)), outline="black")
    return im


def pixel_art(im):
    return im.resize((128, 128), Image.NEAREST).quantize(16).convert("RGB").resize(im.size, Image.NEAREST)


def montar(tmp: Path, n=40):
    region = {"name": "t", "style": {"prompt": "p"}, "generate": {"steps": 1}}
    run = Run(region, tmp / "runs" / "t")
    pd = run.dir / "pairs_v1"
    pd.mkdir(parents=True)
    frames = []
    for i in range(n):
        size = 512 if i % 2 else 1024
        r = cidade(i)
        r.save(pd / f"f{i:03d}_render.png")
        pixel_art(r).save(pd / f"f{i:03d}_art.png")
        frames.append({"id": f"f{i:03d}", "box": [0, 0, size, size]})
    (pd / "frames.json").write_text(json.dumps(frames))
    return run, pd


def test_tudo():
    with tempfile.TemporaryDirectory() as t:
        run, pd = montar(Path(t))
        teste = lab.freeze_testset(run, [pd], n=32)
        assert len(teste) == 32 and sum(x["kind"] == "detalhe" for x in teste) == 16
        assert lab.freeze_testset(run, [pd], n=8) == teste, "teste congelado não muda"
        assert len(lab.test_ids(run)) == 32
        info = lab.calibrate(run, lp=fake_lpips)
        run.region["lab"] = {"copy_min": round(info["copy_p10"] * 0.8, 2)}  # o que o aviso recomenda

        # bordas: a pintura da mesma cidade fica perto; outra cidade fica longe
        assert lab.edge_distance(pixel_art(cidade(1)), cidade(1)) < lab.edge_distance(pixel_art(cidade(1)), cidade(2))

        bom = lambda img, s: pixel_art(img)  # noqa: E731
        copia = lambda img, s: img.filter(ImageFilter.GaussianBlur(0.3))  # noqa: E731
        ruim = lambda img, s: pixel_art(cidade(999))  # noqa: E731  # estilo certo, cidade errada
        bom_ruido = lambda img, s: pixel_art(img.rotate(0.4 * (s % 3)))  # noqa: E731

        e = lab.evaluate(run, "copia", copia, lp=fake_lpips)
        assert e["status"] == "descartado" and "copiou" in e["why"]
        e = lab.evaluate(run, "base", bom_ruido, lp=fake_lpips)
        assert e["status"] == "melhor"
        e2 = lab.evaluate(run, "base", bom_ruido, lp=fake_lpips, seed=1)
        assert e2["status"] == "repetição", e2["why"]
        noise, origem = lab.noise_floor(lab.load_runs(run))
        assert origem == "medido" and noise >= 0
        e = lab.evaluate(run, "geometria-errada", ruim, lp=fake_lpips)
        assert e["status"] == "descartado" and "bordas" in e["why"], e["why"]
        e = lab.evaluate(run, "r1", bom, lp=fake_lpips)
        assert e["status"] in ("melhor", "empate"), e
        html = lab.report(run).read_text()
        assert "geometria-errada" in html and "<svg" in html
        assert (lab.lab_dir(run) / "sheets" / "r1_s0.jpg").exists()

        # o treino exclui os 32 do teste
        import tinymap.region as R
        from tinymap import train
        R.ROOT = Path(t)
        cfg = train.build(run, [pd], "r1", "Qwen/Qwen-Image-Edit-2511", 100, 16, lr=2e-4)
        info = json.loads((cfg.parent / "lab.json").read_text())
        assert info["pairs"] == 8 and info["held_out"] == 32, info
        print("noise", noise, e2["summary"]["score"])


if __name__ == "__main__":
    test_tudo()
    print("ok")
