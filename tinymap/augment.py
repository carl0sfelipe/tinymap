"""Aumentos do conjunto de treino da LoRA (vindos do Isometric Mumbai, nov/2026).

1. Cor da maquete alterada. A mesma pintura-alvo é pareada com cópias do render em que matiz,
   saturação, brilho e ganho de canal variam. A LoRA aprende que a cor da pintura vem do estilo e
   não da foto de satélite ou da textura do Google: copia menos e fica igual entre bairros com
   satélite de datas e luzes diferentes.

2. Preenchimento com moldura vermelha ("infill"). A entrada é a pintura já pronta com uma caixa
   vazia: dentro da caixa, o render; fora, arte terminada; uma moldura vermelha de 2 px marca o que
   pintar. O alvo é a pintura inteira. A LoRA aprende a pintar só dentro da caixa, continuando o que
   está em volta: é o que permite gerar um tile novo encostado em tiles prontos sem emenda (e
   consertar uma área no estúdio com a IA respeitando os vizinhos). Faixas nas 4 bordas e miolo.

Tudo determinístico pela semente.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

RED = (255, 0, 0)
INFILL_PROMPT = (" Only paint inside the red box: everything outside it is finished map; continue it "
                 "seamlessly and remove the red box.")
KINDS = ("left", "right", "top", "bottom", "center")


def color_shift(img: Image.Image, seed: int) -> Image.Image:
    """Render com matiz ±18°, saturação ×0,6-1,4, brilho ×0,85-1,15, contraste ×0,85-1,15 e ganho de
    canal ±6%. Forte o bastante para tirar a muleta da cor, fraco para não virar outra estação."""
    rng = np.random.default_rng(seed)
    a = np.asarray(img.convert("RGB"), np.float32) / 255
    # HSV em numpy puro (rápido)
    mx, mn = a.max(-1), a.min(-1)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.select([mx == r, mx == g], [((g - b) / d) % 6, (b - r) / d + 2], (r - g) / d + 4) / 6
    h = np.where(mx == mn, 0, h)
    h = (h + rng.uniform(-0.05, 0.05)) % 1
    s = np.clip(s * rng.uniform(0.6, 1.4), 0, 1)
    v = np.clip(v * rng.uniform(0.85, 1.15), 0, 1)
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    out = np.stack([
        np.choose(i, [v, q, p, p, t, v]), np.choose(i, [t, v, v, q, p, p]), np.choose(i, [p, p, t, v, v, q])], -1)
    c = rng.uniform(0.85, 1.15)
    out = (out - out.mean()) * c + out.mean()
    out = out * rng.uniform(0.94, 1.06, 3)
    return Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8))


def infill_box(size: tuple[int, int], kind: str, seed: int) -> tuple[int, int, int, int]:
    """Caixa a pintar: faixa de 25-50% numa borda, ou miolo de 40-60%."""
    rng = np.random.default_rng(seed)
    W, H = size
    if kind == "center":
        w, h = (int(W * rng.uniform(0.4, 0.6)), int(H * rng.uniform(0.4, 0.6)))
        x, y = int(rng.uniform(0.15, 0.85 - w / W) * W), int(rng.uniform(0.15, 0.85 - h / H) * H)
        return x, y, x + w, y + h
    f = rng.uniform(0.25, 0.5)
    return {"left": (0, 0, int(W * f), H), "right": (W - int(W * f), 0, W, H),
            "top": (0, 0, W, int(H * f)), "bottom": (0, H - int(H * f), W, H)}[kind]


def infill_example(render: Image.Image, art: Image.Image, kind: str, seed: int) -> Image.Image:
    """Entrada do exemplo de preenchimento: arte fora, render dentro, moldura vermelha de 2 px."""
    art = art.convert("RGB")
    render = render.convert("RGB").resize(art.size, Image.LANCZOS)
    box = infill_box(art.size, kind, seed)
    ctl = art.copy()
    ctl.paste(render.crop(box), box[:2])
    d = ImageDraw.Draw(ctl)
    x0, y0, x1, y1 = box
    d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=RED, width=2)
    return ctl
