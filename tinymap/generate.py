"""Paint the whole map, block by block, on every GPU you have.

The canvas is cut into blocks (default 6144 px). Each block reads a margin of finished neighbours as
pinned context and is sampled jointly (models.QwenEdit.joint), so blocks join without seams.
Blocks run in four parity phases (bx%2, by%2): blocks in one phase never touch, so each GPU takes a
different block of the same phase; a phase starts when the previous one is finished (tiny.london's
schedule). Workers coordinate through lock files, so you can start, stop and resume freely:

    scripts/gpus.sh regions/triplice-fronteira.yaml     # one worker per GPU
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from .region import Run, art_canvas, render_canvas


def blocks(run: Run) -> list[dict]:
    v, g = run.view, run.region["generate"]
    B, s = g["block_px"], v["shot_px"]
    assert B % s == 0, "block_px must be a multiple of shot_px"
    W, H = v["canvas_px"]
    shots = {tuple(x) for x in v["shots"]}
    out = []
    for by in range((H + B - 1) // B):
        for bx in range((W + B - 1) // B):
            x0, y0 = bx * B, by * B
            x1, y1 = min(W, x0 + B), min(H, y0 + B)
            # skip blocks with no rendered shot in them
            cells = [(c, r) for r in range(y0 // s, (y1 - 1) // s + 1) for c in range(x0 // s, (x1 - 1) // s + 1)]
            if not any(cl in shots for cl in cells):
                continue
            out.append({"id": f"b{by:02d}_{bx:02d}", "bx": bx, "by": by, "box": [x0, y0, x1, y1],
                        "phase": (by % 2) * 2 + (bx % 2)})
    return out


def _claim(lock: Path) -> bool:
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def worker(run: Run, model, only: list[str] | None = None) -> None:
    g = run.region["generate"]
    style = run.region["style"]
    W, H = run.view["canvas_px"]
    M = g["margin_px"]
    rc, ac = render_canvas(run), art_canvas(run)
    state = run.path("blocks", ".keep").parent
    all_blocks = [b for b in blocks(run) if not only or b["id"] in only]
    for phase in range(4):
        mine = [b for b in all_blocks if b["phase"] == phase]
        for b in mine:
            done, lock = state / f"{b['id']}.done", state / f"{b['id']}.lock"
            if done.exists() or not _claim(lock):
                continue
            t0 = time.time()
            x0, y0, x1, y1 = b["box"]
            # block plus margin, clamped to the canvas, rounded to whole windows
            cx0, cy0 = max(0, x0 - M), max(0, y0 - M)
            cx1, cy1 = min(W, x1 + M), min(H, y1 + M)
            win = g["window_px"]
            cx1 = max(cx1, cx0 + win)
            cy1 = max(cy1, cy0 + win)
            box = (cx0, cy0, cx1, cy1)
            render = rc.crop(box)
            pinned = ac.crop(box)
            # pinned = margin pixels whose art exists (neighbours finished in earlier phases)
            have = np.zeros((cy1 - cy0, cx1 - cx0), bool)
            s = run.view["shot_px"]
            for r in range(cy0 // s, (cy1 - 1) // s + 1):
                for c in range(cx0 // s, (cx1 - 1) // s + 1):
                    if ac.has(r, c):
                        ys, xs = max(0, r * s - cy0), max(0, c * s - cx0)
                        have[ys:min(cy1 - cy0, (r + 1) * s - cy0), xs:min(cx1 - cx0, (c + 1) * s - cx0)] = True
            core = np.zeros_like(have)
            core[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = True
            pin = have & ~core
            print(f"[{b['id']} phase {phase}] {box}, pinned {pin.mean():.0%}", flush=True)
            img = model.joint(render, f"{style['prompt']}", pinned=pinned, pin_mask=pin,
                              steps=g["steps"], true_cfg=g.get("true_cfg", 1.0), negative=style.get("negative"),
                              window=win, stride=g["stride_px"], seed=g["seed"] + b["bx"] * 1000 + b["by"])
            core_img = img.crop((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0))
            # pad partial edge blocks to whole shots before writing
            if core_img.width % s or core_img.height % s:
                from PIL import Image
                pad = Image.new("RGB", (-(-core_img.width // s) * s, -(-core_img.height // s) * s), (0, 0, 0))
                pad.paste(core_img, (0, 0))
                core_img = pad
            ac.write((x0, y0), core_img)
            done.write_text(json.dumps({"seconds": round(time.time() - t0), "box": b["box"]}))
            lock.unlink(missing_ok=True)
            print(f"[{b['id']}] done in {time.time() - t0:.0f}s", flush=True)
        # wait for the other GPUs to finish this phase before starting the next one
        while not all((state / f"{b['id']}.done").exists() for b in mine):
            time.sleep(15)
    print("all phases finished")
