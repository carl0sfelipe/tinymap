"""Region config, run folders, and reading any box of the canvas from its shot files."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
RENDER_DIR = ROOT / "render"


@dataclass
class Run:
    region: dict
    dir: Path

    @property
    def view(self) -> dict:
        return json.loads((self.dir / "view.json").read_text())

    @property
    def name(self) -> str:
        return self.region["name"]

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load(region_file: str | Path, pilot: bool = False) -> Run:
    region = yaml.safe_load(Path(region_file).read_text())
    name = region["name"] + ("-pilot" if pilot else "")
    run = Run(region, ROOT / "runs" / name)
    run.dir.mkdir(parents=True, exist_ok=True)
    return run


def node(script: str, *args: str) -> None:
    subprocess.run(["node", str(RENDER_DIR / script), *map(str, args)], check=True, cwd=RENDER_DIR)


class Canvas:
    """Read-only view over a folder of shot images (r{row}_c{col}.png) as one big image.

    Missing shots read as `fill`. Used for renders (out/render) and finished art (out/art).
    """

    def __init__(self, folder: Path, shot: int, size: tuple[int, int], fill=(135, 206, 235)):
        self.folder, self.shot, self.size, self.fill = Path(folder), shot, size, fill

    @lru_cache(maxsize=64)
    def _shot(self, r: int, c: int) -> Image.Image | None:
        p = self.folder / f"r{r}_c{c}.png"
        return Image.open(p).convert("RGB") if p.exists() else None

    def has(self, r: int, c: int) -> bool:
        return (self.folder / f"r{r}_c{c}.png").exists()

    def crop(self, box: tuple[int, int, int, int]) -> Image.Image:
        x0, y0, x1, y1 = box
        out = Image.new("RGB", (x1 - x0, y1 - y0), self.fill)
        s = self.shot
        for r in range(max(0, y0 // s), (y1 - 1) // s + 1):
            for c in range(max(0, x0 // s), (x1 - 1) // s + 1):
                im = self._shot(r, c)
                if im is not None:
                    out.paste(im, (c * s - x0, r * s - y0))
        return out

    def write(self, box_origin: tuple[int, int], img: Image.Image) -> None:
        """Paste img into the shot files it covers (creating them as needed). Shot-aligned writes only."""
        x0, y0 = box_origin
        s = self.shot
        assert x0 % s == 0 and y0 % s == 0 and img.width % s == 0 and img.height % s == 0
        self.folder.mkdir(parents=True, exist_ok=True)
        for r in range(img.height // s):
            for c in range(img.width // s):
                tile = img.crop((c * s, r * s, (c + 1) * s, (r + 1) * s))
                R, C = y0 // s + r, x0 // s + c
                tmp = self.folder / f".r{R}_c{C}.png"
                tile.save(tmp)
                tmp.rename(self.folder / f"r{R}_c{C}.png")
                self._shot.cache_clear()


def render_canvas(run: Run) -> Canvas:
    v = run.view
    # shot não planejado lia azul-céu, que o modelo pintava como lago e o conserto contava como rio;
    # na maquete OSM o preenchimento é a cor do chão
    fill = (207, 202, 185) if v.get("source") == "osm" else (135, 206, 235)
    return Canvas(run.dir / "render", v["shot_px"], tuple(v["canvas_px"]), fill=fill)


def art_canvas(run: Run) -> Canvas:
    v = run.view
    return Canvas(run.dir / "art", v["shot_px"], tuple(v["canvas_px"]), fill=(0, 0, 0))
