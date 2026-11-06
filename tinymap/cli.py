"""tinymap: isometric pixel-art map of any place, generated on your own GPUs.

  tinymap plan     <region.yaml> [--pilot]        canvas + shot list (runs/<name>/view.json)
  tinymap probe    <region.yaml> [--pilot]        1 km renders at each landmark: is Google 3D good here?
  tinymap render   <region.yaml> [--workers 4]    Google Photorealistic 3D Tiles -> render shots
  tinymap osm      <region.yaml>                  water / places / borders from OpenStreetMap
  tinymap stitch   <region.yaml> [--scale 0.125]  small preview of render (or --art) for checking
  tinymap teacher  <region.yaml> --backend qwen|gemini|folder --out pairs_v1 [--n 120] [--lora ...]
  tinymap sheets   <region.yaml> --out pairs_v1   contact sheets for the human pass
  tinymap train    <region.yaml> --pairs pairs_v1 [pairs_v2 ...] --name v1 [--steps 4000]
  tinymap try      <region.yaml> --lora X.safetensors [--at lat,lon]   one block, to judge a LoRA
  tinymap generate <region.yaml> --lora X.safetensors [--gpu 0]       paint the map (one worker)
  tinymap unlock   <region.yaml>                  clear locks left by a crashed worker
  tinymap site     <region.yaml> [--version v1]   Deep Zoom pyramid + viewer in site/
  tinymap bridge   <region.yaml> --cameras cameras.yaml [--dry-run]   ponte ao vivo -> now.services
  tinymap bridge-calibrate <region.yaml> --cameras cameras.yaml --camera ID   quadro com as faixas
  tinymap postcard <region.yaml> --at lat,lon --name "Nome"        cartão 1200x630 (og:image)
  tinymap postcards <region.yaml> [--pinos URL]                    um cartão por loja/imóvel do now.services
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import region as R


def _model(a):
    from .models import ModelCfg, QwenEdit
    return QwenEdit(ModelCfg(model_id=a.model, lora=a.lora, lora_scale=a.lora_scale, quant=a.quant, device=f"cuda:{a.gpu}"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="tinymap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd")
    ap.add_argument("region")
    ap.add_argument("--pilot", action="store_true", help="use pilot_bbox (separate run folder)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.125)
    ap.add_argument("--art", action="store_true")
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--out", default="pairs_v1")
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--model", default="Qwen/Qwen-Image-Edit-2511")
    ap.add_argument("--lora")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--at", help="lat,lon for `try`")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--cameras", default="cameras.yaml")
    ap.add_argument("--camera")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--pinos")
    ap.add_argument("--precise", default="yes", choices=["yes", "no"])
    a = ap.parse_args()
    run = R.load(a.region, pilot=a.pilot)

    if a.cmd == "plan":
        json.dump(run.region, open(run.path("region.json"), "w"), ensure_ascii=False, default=str)
        R.node("plan.mjs", run.dir, *(["--pilot"] if a.pilot else []))
    elif a.cmd == "probe":
        pts = [f"{lm['lat']},{lm['lon']},{lm['name']}" for lm in run.region.get("landmarks", [])]
        R.node("capture.mjs", run.dir, "--probe", *pts)
        print(f"look at {run.dir / 'probe'}: melted/flat buildings = poor 3D coverage there")
    elif a.cmd == "render":
        R.node("capture.mjs", run.dir, "--workers", a.workers)
    elif a.cmd == "osm":
        from .osm import fetch
        fetch(run)
    elif a.cmd == "stitch":
        from PIL import Image
        v = run.view
        cv = R.art_canvas(run) if a.art else R.render_canvas(run)
        w, h = v["canvas_px"]
        t = max(1, int(v["shot_px"] * a.scale))
        out = Image.new("RGB", (v["cols"] * t, v["rows"] * t), (20, 20, 20))
        for r in range(v["rows"]):
            for c in range(v["cols"]):
                if cv.has(r, c):
                    out.paste(cv._shot(r, c).resize((t, t), Image.LANCZOS), (c * t, r * t))
        p = run.path(("art" if a.art else "render") + "_preview.jpg")
        out.save(p, quality=88)
        print(p)
    elif a.cmd == "teacher":
        from .teacher import run_teacher
        model = _model(a) if a.backend == "qwen" else None
        run_teacher(run, run.path(a.out, ".keep").parent, a.n, a.backend, model)
    elif a.cmd == "sheets":
        from .teacher import contact_sheet
        print(contact_sheet(run.dir / a.out))
    elif a.cmd == "train":
        from .train import build
        build(run, [run.dir / p for p in a.pairs], a.name, a.model, a.steps, a.rank)
    elif a.cmd == "try":
        from .region import render_canvas
        v = run.view
        if a.at:
            Path(run.path("osm", "try_ll.json")).write_text(json.dumps({"points": [{"name": "try", "lat": float(a.at.split(",")[0]), "lon": float(a.at.split(",")[1])}]}))
            R.node("project.mjs", run.dir, run.path("osm", "try_ll.json"), run.path("osm", "try_px.json"))
            x, y = json.loads(run.path("osm", "try_px.json").read_text())["points"][0]["px"]
        else:
            x, y = v["canvas_px"][0] / 2, v["canvas_px"][1] / 2
        size = 3072
        x0, y0 = int(x - size / 2) // 16 * 16, int(y - size / 2) // 16 * 16
        render = render_canvas(run).crop((x0, y0, x0 + size, y0 + size))
        g = run.region["generate"]
        img = _model(a).joint(render, run.region["style"]["prompt"], steps=g["steps"], true_cfg=g.get("true_cfg", 1.0),
                              negative=run.region["style"].get("negative"), window=g["window_px"], stride=g["stride_px"])
        tag = Path(a.lora).stem if a.lora else "base"
        render.save(run.path("try", f"{x0}_{y0}_render.png"))
        img.save(run.path("try", f"{x0}_{y0}_{tag}.png"))
        print(run.path("try", f"{x0}_{y0}_{tag}.png"))
    elif a.cmd == "generate":
        from .generate import worker
        worker(run, _model(a))
    elif a.cmd == "unlock":
        for p in (run.dir / "blocks").glob("*.lock"):
            p.unlink()
            print("removed", p.name)
    elif a.cmd == "site":
        from .pyramid import build
        build(run, a.version)
    elif a.cmd == "bridge":
        from .bridge import run as bridge_run
        bridge_run(Path(a.cameras), weights=a.weights, device=f"cuda:{a.gpu}", dry_run=a.dry_run)
    elif a.cmd == "bridge-calibrate":
        from .bridge import calibrate
        calibrate(Path(a.cameras), a.camera, run.path("bridge", f"{a.camera}_faixas.png"))
    elif a.cmd == "postcard":
        from .postcard import make
        lat, lon = map(float, a.at.split(","))
        print(make(run, lat, lon, a.name if a.name != "v1" else a.at, precise=a.precise == "yes"))
    elif a.cmd == "postcards":
        from .postcard import batch
        batch(run, a.pinos or run.region["now_services"]["pinos"])
    else:
        ap.error(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
