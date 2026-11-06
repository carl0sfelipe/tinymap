"""Automatic checks on a redraw: did the model really draw pixel art, and did it keep the geography?

Thresholds come from tiny.london's write-up (careful redraws 0.42-0.55 flat share, half-copies
0.18-0.35) and tiny-utrecht's checks; tune them on your own first 50 pairs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

FLAT_MIN = 0.15
SOFT_MAX = 0.31
INVENTED_WATER_MAX = 0.003
BLANK_MAX = 0.01
EDGE_MIN = 0.12


def flat_share(img: Image.Image, tol: int = 6) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    h, w, _ = a.shape
    c = a[1:-1, 1:-1]
    flat = np.ones((h - 2, w - 2), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            flat &= (np.abs(a[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] - c) <= tol).all(2)
    return float(flat.mean())


def soft_share(img: Image.Image) -> float:
    a = np.asarray(img.convert("L"), dtype=np.int16)
    d = np.concatenate([np.abs(np.diff(a, axis=0)).ravel(), np.abs(np.diff(a, axis=1)).ravel()])
    return float(((d >= 3) & (d <= 12)).mean())


def blank_share(img: Image.Image) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    white = (a.min(2) > 225) & (a.max(2) - a.min(2) < 14)
    big = Image.fromarray(white.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(21))
    return float((np.asarray(big) > 127).mean())


def invented_water(drawn: Image.Image, mask: Image.Image) -> float:
    a = np.asarray(drawn.convert("RGB"), dtype=np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    wet = (b - r > 60) & (b - g > 15)
    blobs = Image.fromarray(wet.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(9)).filter(ImageFilter.MaxFilter(9))
    near = np.asarray(mask.filter(ImageFilter.MaxFilter(25))) > 127
    return float(((np.asarray(blobs) > 127) & ~near).mean())


def edge_agreement(drawn: Image.Image, render: Image.Image) -> float:
    """Correlation of coarse edge maps: low means buildings moved or were invented."""
    def edges(im):
        g = im.convert("L").resize((256, 256), Image.BILINEAR).filter(ImageFilter.FIND_EDGES)
        return np.asarray(g.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32).ravel()
    a, b = edges(drawn), edges(render)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def check(drawn: Image.Image, render: Image.Image, water_mask: Image.Image) -> tuple[list[str], dict]:
    m = {
        "flat": flat_share(drawn), "soft": soft_share(drawn),
        "water": invented_water(drawn, water_mask),
        "blank": blank_share(drawn) - blank_share(render),
        "edges": edge_agreement(drawn, render),
    }
    fails = [name for name, bad in (
        ("photo-copy", m["flat"] < FLAT_MIN or m["soft"] > SOFT_MAX),
        ("invented-water", m["water"] > INVENTED_WATER_MAX),
        ("blank-area", m["blank"] > BLANK_MAX),
        ("geometry-drift", m["edges"] < EDGE_MIN),
    ) if bad]
    return fails, m
