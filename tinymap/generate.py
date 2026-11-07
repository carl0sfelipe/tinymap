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


def _small(mask: np.ndarray, f: int) -> np.ndarray:
    """Reduz uma máscara booleana por f (máximo em cada bloco f x f)."""
    h, w = mask.shape
    H, W = -(-h // f) * f, -(-w // f) * f
    m = np.zeros((H, W), bool)
    m[:h, :w] = mask
    return m.reshape(H // f, f, W // f, f).any(axis=(1, 3))


def _big(small: np.ndarray, f: int, shape) -> np.ndarray:
    return np.repeat(np.repeat(small, f, 0), f, 1)[:shape[0], :shape[1]]


def _filt(mask: np.ndarray, kind: str, k: int) -> np.ndarray:
    from PIL import Image, ImageFilter
    F = ImageFilter.MaxFilter if kind == "max" else ImageFilter.MinFilter
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(F(k))) > 127


def water_errors(img, render, f: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """(água inventada, terra no rio), comparando a arte com o render.

    No render o rio do OSM está pintado de azul e as pontes de cinza por cima, então "onde deve
    haver água" sai do próprio render (e uma ponte nunca vira erro). Feito em 1/f da resolução:
    os filtros do PIL ficam lentos demais em blocos de 8 mil px."""
    from .checks import _wet
    art = _wet(img)
    a16 = np.asarray(img.convert("RGB"), dtype=np.int16)
    # para "terra no rio" vale qualquer azul, até acinzentado/esverdeado: o modelo nem sempre usa o azul
    # saturado do render, e a regra estrita marcava 61% de um bloco de rio como terra
    loose = (a16[..., 2] - a16[..., 0] > 25) & (a16[..., 2] >= a16[..., 1] - 5)
    ref = _small(_wet(render), f)
    a = _filt(_filt(_small(art, f), "min", 3), "max", 3)      # sem pontinhos (~9 px na resolução cheia)
    near = _filt(ref, "max", 2 * (12 // f) + 1)                # 12 px de folga na margem
    core = _filt(ref, "min", 2 * (12 // f) + 1)                # miolo do rio
    invented = _big(a & ~near, f, art.shape) & art
    land = _big(_filt(_filt(_small(~loose, f), "min", 3), "max", 3) & core, f, art.shape) & ~loose
    return invented, land


def grow_mask(bad: np.ndarray, r: int = 32, f: int = 8) -> np.ndarray:
    return _big(_filt(_small(bad, f), "max", 2 * (r // f) + 1), f, bad.shape)


STEAL_AFTER = 25 * 60

NO_WATER = (" There is no river, lake, pond or canal in this part of the view: draw the streets, yards, trees and"
            " roofs that are there.")
ONLY_WATER = " This part of the view is open river water: draw only water with small ripples, no land, no buildings."


def repair(model, render, img, prompt: str, g: dict, style: dict, seed: int, tries: int = 1, log=print,
           max_share: float = 0.30, area: np.ndarray | None = None):
    """Redesenha só os pedaços com água inventada ou terra no rio (o resto fica fixo por pinning)."""
    W, H = img.size
    for t in range(tries):
        invented, land = water_errors(img, render)
        if area is not None:  # fora da área do mapa o site recorta: não vale gastar GPU consertando
            invented, land = invented & area, land & area
        if invented.mean() + land.mean() < 0.001:
            break
        log(f"  conserto {t + 1}: água inventada {invented.mean():.1%}, terra no rio {land.mean():.1%}")
        for bad, hint in ((invented, NO_WATER), (land, ONLY_WATER)):
            if bad.mean() < 0.0005:
                continue
            if bad.mean() > max_share:  # mais de 30% do bloco: provavelmente a regra de cor, não a arte
                log(f"  conserto pulado ({bad.mean():.0%} marcado)")
                continue
            grow = grow_mask(bad)
            T = 2048
            for ty in range(0, H, T):
                for tx in range(0, W, T):
                    sub = grow[ty:ty + T, tx:tx + T]
                    if sub.mean() < 0.002:
                        continue
                    ys, xs = np.nonzero(sub)
                    x0, x1 = tx + xs.min() - 512, tx + xs.max() + 512
                    y0, y1 = ty + ys.min() - 512, ty + ys.max() + 512
                    # caixa entre 1 janela e o bloco inteiro; antes, caixa maior que o bloco dava x0 negativo
                    w = min(W // 16 * 16, max(g["window_px"], (x1 - x0 + 15) // 16 * 16))
                    h = min(H // 16 * 16, max(g["window_px"], (y1 - y0 + 15) // 16 * 16))
                    x0 = int(max(0, min(x0, W - w))) // 16 * 16
                    y0 = int(max(0, min(y0, H - h))) // 16 * 16
                    box = (x0, y0, x0 + w, y0 + h)
                    fixed = model.joint(render.crop(box), prompt + hint, pinned=img.crop(box),
                                        pin_mask=~grow[y0:y0 + h, x0:x0 + w], steps=g["steps"],
                                        true_cfg=g.get("true_cfg", 1.0), negative=style.get("negative"),
                                        window=g["window_px"], stride=g["stride_px"],
                                        seed=seed + 7919 * (t + 1) + x0 + y0, log=lambda *_: None)
                    img.paste(fixed, (x0, y0))
    return img


def _claim(lock: Path) -> bool:
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def checkpoint() -> None:
    """Gancho para o Volume da Modal publicar o checkpoint (modal_app troca por runs.commit)."""


def worker(run: Run, model, only: list[str] | None = None, shard: tuple[int, int] | None = None) -> None:
    """shard=(i, n): o worker i pega só os blocos i, i+n, ... de cada fase. Na Modal o lock por arquivo
    não é atômico entre containers (dois workers pegaram o mesmo bloco), então lá a divisão é fixa."""
    g = run.region["generate"]
    style = run.region["style"]
    W, H = run.view["canvas_px"]
    M = g["margin_px"]
    rc, ac = render_canvas(run), art_canvas(run)
    state = run.path("blocks", ".keep").parent
    all_blocks = [b for b in blocks(run) if not only or b["id"] in only]
    def paint(b, phase) -> None:
        done, lock = state / f"{b['id']}.done", state / f"{b['id']}.lock"
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
        painted = state / f"{b['id']}.painted.png"
        if painted.exists():  # a GPU foi tomada (preempção) depois da pintura: retoma do conserto
            from PIL import Image
            img = Image.open(painted).convert("RGB")
            print(f"[{b['id']}] retomando do checkpoint", flush=True)
        else:
            img = model.joint(render, f"{style['prompt']}", pinned=pinned, pin_mask=pin,
                              steps=g["steps"], true_cfg=g.get("true_cfg", 1.0), negative=style.get("negative"),
                              window=win, stride=g["stride_px"], seed=g["seed"] + b["bx"] * 1000 + b["by"])
            tmp = state / f".{b['id']}.painted.png"
            img.save(tmp)
            tmp.rename(painted)
            checkpoint()
        if g.get("repair", True):
            from PIL import Image, ImageDraw
            am = Image.new("L", (cx1 - cx0, cy1 - cy0), 0)
            ImageDraw.Draw(am).polygon([(x - cx0, y - cy0) for x, y in run.view["area_px"]], fill=255)
            img = repair(model, render, img, style["prompt"], g, style, g["seed"] + b["bx"] * 1000 + b["by"],
                         area=np.asarray(am) > 127)
        core_img = img.crop((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0))
        # pad partial edge blocks to whole shots before writing
        if core_img.width % s or core_img.height % s:
            from PIL import Image
            pad = Image.new("RGB", (-(-core_img.width // s) * s, -(-core_img.height // s) * s), (0, 0, 0))
            pad.paste(core_img, (0, 0))
            core_img = pad
        ac.write((x0, y0), core_img)
        done.write_text(json.dumps({"seconds": round(time.time() - t0), "box": b["box"]}))
        painted.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
        print(f"[{b['id']}] done in {time.time() - t0:.0f}s", flush=True)

    for phase in range(4):
        mine = [b for b in all_blocks if b["phase"] == phase]
        for b in (mine[shard[0]::shard[1]] if shard else mine):
            done, lock = state / f"{b['id']}.done", state / f"{b['id']}.lock"
            if done.exists() or (shard is None and not _claim(lock)):
                continue
            paint(b, phase)
        # espera as outras GPUs terminarem a fase; se um worker caiu, depois de STEAL_AFTER segundos este
        # assume os blocos que faltam (senão as GPUs ficam paradas, cobrando, esperando um bloco que nunca vem)
        waited = 0
        while not all((state / f"{b['id']}.done").exists() for b in mine):
            time.sleep(15)
            waited += 15
            if waited >= STEAL_AFTER:
                for i, b in enumerate(mine):
                    if (state / f"{b['id']}.done").exists():
                        continue
                    ck = state / f"{b['id']}.painted.png"
                    if ck.exists() and time.time() - ck.stat().st_mtime < 3600:
                        continue  # já pintado há menos de 1 h: o dono está vivo, consertando
                    if shard and (i % shard[1] + 1) % shard[1] != shard[0]:
                        continue  # só o worker seguinte ao dono assume (senão todos pegam o mesmo bloco)
                    print(f"[{b['id']}] pendente há {waited // 60} min: assumindo", flush=True)
                    paint(b, phase)
                waited = 0
    print("all phases finished")
