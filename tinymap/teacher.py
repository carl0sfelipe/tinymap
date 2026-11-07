"""Teacher pairs: pick 1024 px frames of the render, redraw them, keep only redraws that pass the checks.

Backends (choose with --backend):
  qwen    local, on your GPUs: Qwen-Image-Edit (+ optional LoRA, e.g. the public isometric-nyc LoRA
          or your own previous round). Free; quality depends on the LoRA.
  gemini  Gemini 3 Pro Image / Nano Banana Pro through the API (what tiny.london, Toronto, Utrecht,
          isometric.nyc used). Best teacher; costs money (~US$0.13 per image at launch prices).
  folder  you drop your own drawings as <id>_art.png (hand-fixed, from another tool...).

The usual loop is: ~100 frames with the best teacher you can afford -> train LoRA v1 locally ->
redraw ~1000 frames locally with v1 -> filter with checks + your eyes -> train v2 -> generate the map.
"""

from __future__ import annotations

import base64
import json
import os
import random
import unicodedata
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .checks import check
from .osm import Water, water_sentence
from .region import Run, render_canvas

FRAME = 1024
ATTEMPTS = 3


def _usable(canvas, box, W: int, H: int) -> bool:
    """Inside the canvas, not sky/empty (missing shots read as sky) and not one flat colour."""
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return False
    crop = np.asarray(canvas.crop(box), np.int16)
    sky = (np.abs(crop - np.array([135, 206, 235])).sum(2) < 12).mean()
    return sky <= 0.05 and crop.std() >= 12


def sample_frames(run: Run, n: int, seed: int = 0) -> list[dict]:
    """Landmarks + stratified random frames inside the area, skipping sky/empty renders.
    Half of the random frames are 512 px quarters (redrawn zoomed 2x), which teachers copy less."""
    v = run.view
    canvas = render_canvas(run)
    out = []
    W, H = v["canvas_px"]
    # landmarks from places_px.json
    places = run.dir / "osm" / "places_px.json"
    if places.exists():
        for p in json.loads(places.read_text())["points"]:
            if p.get("kind") == "landmark":
                x, y = map(int, p["px"])
                if not _usable(canvas, (x - FRAME // 2, y - FRAME // 2, x + FRAME // 2, y + FRAME // 2), W, H):
                    continue  # landmark fora deste recorte (piloto) ou sem render
                slug = unicodedata.normalize("NFKD", p["name"].lower()).encode("ascii", "ignore").decode()
                out.append({"id": "lm_" + "".join(ch for ch in slug if ch.isalnum())[:24],
                            "box": [x - FRAME // 2, y - FRAME // 2, x + FRAME // 2, y + FRAME // 2]})
    rng = random.Random(seed)
    shots = v["shots"]
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        c, r = rng.choice(shots)
        size = FRAME if rng.random() < 0.5 else FRAME // 2
        x = c * v["shot_px"] + rng.randrange(0, v["shot_px"] - size)
        y = r * v["shot_px"] + rng.randrange(0, v["shot_px"] - size)
        if not canvas.has(r, c) or not _usable(canvas, (x, y, x + size, y + size), W, H):
            continue
        out.append({"id": f"f{len(out):04d}_{size}", "box": [x, y, x + size, y + size]})
    return out


def gemini_redraw(img: Image.Image, prompt: str) -> Image.Image | None:
    key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")
    buf = BytesIO()
    img.save(buf, "PNG")
    body = {"contents": [{"parts": [{"text": prompt},
                                    {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}]}],
            "generationConfig": {"responseModalities": ["IMAGE"]}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.load(r)
    for part in res.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        data = part.get("inline_data") or part.get("inlineData")
        if data:
            return Image.open(BytesIO(base64.b64decode(data["data"]))).convert("RGB")
    return None


def run_teacher(run: Run, out_dir: Path, n: int, backend: str, model=None, seed: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_file = out_dir / "frames.json"
    if frames_file.exists():
        frames = json.loads(frames_file.read_text())
    else:
        frames = sample_frames(run, n, seed)
        frames_file.write_text(json.dumps(frames, indent=1))
    canvas, water = render_canvas(run), Water(run)
    style = run.region["style"]
    report = open(out_dir / "report.txt", "a")
    kept = 0
    for i, fr in enumerate(frames):
        fid, box = fr["id"], tuple(fr["box"])
        if (out_dir / f"{fid}.done").exists():
            kept += (out_dir / f"{fid}_art.png").exists()
            continue
        size = box[2] - box[0]
        render = canvas.crop(box)
        mask = water.mask(box)
        render.save(out_dir / f"{fid}_render.png")
        big = render.resize((FRAME, FRAME), Image.LANCZOS)
        prompt = f"{style['prompt']} {water_sentence(mask)}"
        line = f"{fid}: "
        for attempt in range(1, ATTEMPTS + 1):
            if backend == "gemini":
                drawn = gemini_redraw(big, prompt)
            elif backend == "qwen":
                drawn = model.edit(big, prompt, steps=run.region["generate"].get("teacher_steps", 30),
                                   true_cfg=run.region["generate"].get("teacher_cfg", 4.0),
                                   negative=style.get("negative"), seed=seed + attempt)
            elif backend == "folder":
                p = out_dir / f"{fid}_art.png"
                drawn = Image.open(p).convert("RGB") if p.exists() else None
            else:
                raise ValueError(backend)
            if drawn is None:
                line += f"#{attempt} no image; "
                continue
            drawn_1k = drawn.resize((FRAME, FRAME), Image.LANCZOS)
            small = drawn_1k.resize((size, size), Image.LANCZOS)
            fails, m = check(small, render, mask)
            line += f"#{attempt} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()) + (" FAIL " + ",".join(fails) if fails else " PASS") + "; "
            if not fails:
                small.save(out_dir / f"{fid}_art.png")
                kept += 1
                break
            small.save(out_dir / f"{fid}_try{attempt}.png")
            if backend == "folder":
                break
        (out_dir / f"{fid}.done").write_text(line)
        report.write(line + "\n")
        report.flush()
        print(f"[{i + 1}/{len(frames)}] {line[-80:]}")
    print(f"{kept}/{len(frames)} pairs kept in {out_dir}")


def load_review(out_dir: Path) -> dict:
    """Decisões da revisão humana (tinymap review): {id: {status, chosen, hint, edited}}.
    status: todo (a revisar) | approved | redo (gerar de novo, com a dica) | discard."""
    p = out_dir / "review.json"
    return json.loads(p.read_text()) if p.exists() else {}


def candidates(out_dir: Path, fid: str) -> list[Path]:
    return sorted(out_dir.glob(f"{fid}_cand*.png"), key=lambda p: int(p.stem.rsplit("cand", 1)[1]))


def extend(run: Run, out_dir: Path, m: int, seed: int) -> list[dict]:
    """Acrescenta m recortes novos a frames.json (rodada nova: ids r<k>_..., sem repetir caixa)."""
    frames = json.loads((out_dir / "frames.json").read_text())
    rnd = 1 + max(fr.get("round", 0) for fr in frames)
    seen = {tuple(fr["box"]) for fr in frames}
    new = [fr for fr in sample_frames(run, m + 50, seed + 1000 * rnd) if not fr["id"].startswith("lm_")
           and tuple(fr["box"]) not in seen][:m]
    for fr in new:
        fr["id"], fr["round"] = f"r{rnd}_{fr['id']}", rnd
    frames += new
    (out_dir / "frames.json").write_text(json.dumps(frames, indent=1))
    return new


def prepare(run: Run, out_dir: Path, n: int, seed: int = 0) -> list[dict]:
    """Inputs for a batch teacher (e.g. Qwen-Image-2.1 on Modal): frames.json, <id>_render.png and
    one prompt per frame (style.teacher_prompt or style.prompt, plus where the river is, plus the
    reviewer's hint from review.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_file = out_dir / "frames.json"
    if frames_file.exists():
        frames = json.loads(frames_file.read_text())
    else:
        frames = sample_frames(run, n, seed)
        frames_file.write_text(json.dumps(frames, indent=1))
    canvas, water = render_canvas(run), Water(run)
    style = run.region["style"]
    base = style.get("teacher_prompt") or style["prompt"]
    review = load_review(out_dir)
    jobs = []
    for fr in frames:
        fid, box = fr["id"], tuple(fr["box"])
        rp = out_dir / f"{fid}_render.png"
        if rp.exists():
            render = Image.open(rp).convert("RGB")
        else:
            render = canvas.crop(box)
            render.save(rp)
        hint = review.get(fid, {}).get("hint", "").strip()
        prompt = f"{base} {water_sentence(water.mask(box))}" + (f" Note for this view: {hint}" if hint else "")
        jobs.append({"id": fid, "prompt": prompt, "image": render.resize((FRAME, FRAME), Image.LANCZOS),
                     "status": review.get(fid, {}).get("status", "todo")})
    (out_dir / "prompts.json").write_text(json.dumps({j["id"]: j["prompt"] for j in jobs}, indent=1, ensure_ascii=False))
    return jobs


def _score_one(args) -> tuple[str, dict]:
    run, out_dir, fr, done = args
    if run.dir not in _WATER:
        _WATER[run.dir] = Water(run)
    fid, box = fr["id"], tuple(fr["box"])
    size = box[2] - box[0]
    render = Image.open(out_dir / f"{fid}_render.png").convert("RGB")
    mask = _WATER[run.dir].mask(box)
    res = {}
    for cand in candidates(out_dir, fid):
        if cand.name in done:
            continue
        small = Image.open(cand).convert("RGB").resize((FRAME, FRAME), Image.LANCZOS).resize((size, size), Image.LANCZOS)
        fails, m = check(small, render, mask)
        res[cand.name] = {"fails": fails, "m": {k: round(v, 3) for k, v in m.items()}}
    return fid, res


def score(run: Run, out_dir: Path, workers: int | None = None) -> dict:
    """Checks de todos os candidatos ainda sem nota -> checks.json. Só sugerem: quem aprova é a revisão."""
    from concurrent.futures import ProcessPoolExecutor

    frames = json.loads((out_dir / "frames.json").read_text())
    p = out_dir / "checks.json"
    checks = json.loads(p.read_text()) if p.exists() else {}
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        for fid, res in ex.map(_score_one, [(run, out_dir, fr, set(checks)) for fr in frames]):
            checks.update(res)
    p.write_text(json.dumps(checks))
    ok = sum(1 for v in checks.values() if not v["fails"])
    print(f"checks: {len(checks)} candidatos, {ok} passam (sugestões para a revisão)")
    return checks


_WATER = {}


def _accept_one(args) -> tuple[str, str, bool]:
    run, out_dir, fr = args
    if run.dir not in _WATER:  # um Water por processo
        _WATER[run.dir] = Water(run)
    fid, box = fr["id"], tuple(fr["box"])
    size = box[2] - box[0]
    render = Image.open(out_dir / f"{fid}_render.png").convert("RGB")
    mask = _WATER[run.dir].mask(box)
    line = f"{fid}: "
    for cand in sorted(out_dir.glob(f"{fid}_cand*.png")):
        small = Image.open(cand).convert("RGB").resize((FRAME, FRAME), Image.LANCZOS).resize((size, size), Image.LANCZOS)
        fails, m = check(small, render, mask)
        line += f"{cand.stem[len(fid) + 1:]} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()) + (" FAIL " + ",".join(fails) if fails else " PASS") + "; "
        if not fails:
            small.save(out_dir / f"{fid}_art.png")
            (out_dir / f"{fid}.done").write_text(line)
            return fid, line, True
    (out_dir / f"{fid}.done").write_text(line)
    return fid, line, False


def accept(run: Run, out_dir: Path, workers: int | None = None) -> int:
    """Check <id>_cand<k>.png from a batch teacher; the first that passes becomes <id>_art.png.
    Runs on every CPU core (the checks are ~10 s per frame of 4 candidates on one core)."""
    from concurrent.futures import ProcessPoolExecutor

    frames = json.loads((out_dir / "frames.json").read_text())
    kept = 0
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex, open(out_dir / "report.txt", "w") as report:
        for i, (fid, line, ok) in enumerate(ex.map(_accept_one, [(run, out_dir, fr) for fr in frames]), 1):
            kept += ok
            report.write(line + "\n")
            report.flush()
            if i % 20 == 0:
                print(f"  checados {i}/{len(frames)}, {kept} aprovados", flush=True)
    print(f"{kept}/{len(frames)} pairs kept in {out_dir} (details in report.txt)")
    return kept


def contact_sheet(pairs_dir: Path, cols: int = 4, thumb: int = 384) -> Path:
    """render | art side by side, one sheet per 16 pairs, for the human pass (delete bad *_art.png)."""
    arts = sorted(pairs_dir.glob("*_art.png"))
    sheets = pairs_dir / "sheets"
    sheets.mkdir(exist_ok=True)
    for s in range(0, len(arts), cols * 4):
        chunk = arts[s:s + cols * 4]
        sheet = Image.new("RGB", (cols * thumb * 2, ((len(chunk) + cols - 1) // cols) * thumb), "white")
        for k, a in enumerate(chunk):
            r = Image.open(str(a).replace("_art", "_render")).resize((thumb, thumb))
            d = Image.open(a).resize((thumb, thumb))
            x, y = (k % cols) * thumb * 2, (k // cols) * thumb
            sheet.paste(r, (x, y))
            sheet.paste(d, (x + thumb, y))
        sheet.save(sheets / f"sheet_{s // (cols * 4):03d}.jpg", quality=85)
        (sheets / f"sheet_{s // (cols * 4):03d}.txt").write_text("\n".join(a.name for a in chunk))
    return sheets
