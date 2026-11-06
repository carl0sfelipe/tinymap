"""Rodar o tinymap nas GPUs da Modal em vez (ou além) das 3090 locais.

Do seu computador, com o token da Modal configurado (`modal token set ...`):

    modal run tinymap/modal_app.py::upload   --region regions/triplice-fronteira.yaml
    modal run tinymap/modal_app.py::try_lora --region regions/triplice-fronteira.yaml --lora loras/v2.safetensors
    modal run tinymap/modal_app.py::generate --region regions/triplice-fronteira.yaml --lora loras/v2.safetensors --gpus 4
    modal run tinymap/modal_app.py::download --region regions/triplice-fronteira.yaml

O render (Google 3D Tiles) e os pares de treino continuam sendo feitos antes, localmente:
`upload` envia a pasta runs/<região> (render, osm, view.json) para um Volume da Modal; cada worker
de `generate` é um container com GPU que roda exatamente o mesmo `generate.worker` das 3090 —
os lock files no Volume coordenam os blocos, então dá para misturar Modal e 3090 se a pasta for
sincronizada. `download` traz a arte pintada de volta para runs/<região>/art.

Na H100 (80 GB) o modelo roda em bf16 sem quantização: mais rápido e um pouco melhor que NF4.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("tinymap")
runs = modal.Volume.from_name("tinymap-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("tinymap-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libvips", "nodejs", "npm")
    .pip_install(
        "torch>=2.4", "diffusers>=0.36", "transformers>=4.51", "accelerate", "peft>=0.17",
        "bitsandbytes>=0.45", "safetensors", "sentencepiece", "pyyaml", "numpy", "pillow", "pyvips",
    )
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "tinymap", "/app/tinymap")
    .add_local_dir(ROOT / "regions", "/app/regions")
    .add_local_dir(ROOT / "render", "/app/render", ignore=["node_modules"])
)
GPU = "H100"
MODEL = "Qwen/Qwen-Image-Edit-2511"


def _setup(region_file: str):
    """Aponta o tinymap para o Volume (runs/) e carrega a região."""
    import os
    import sys
    sys.path.insert(0, "/app")
    os.makedirs("/runs", exist_ok=True)
    if not os.path.exists("/app/runs"):
        os.symlink("/runs", "/app/runs")
    from tinymap import region as R
    return R.load(f"/app/{region_file}")


def _model(lora_path: str | None):
    from tinymap.models import ModelCfg, QwenEdit
    return QwenEdit(ModelCfg(model_id=MODEL, lora=lora_path, quant="none", device="cuda:0"))


@app.function(image=image, volumes={"/runs": runs}, timeout=3600)
def _receive(name: str, files: list[tuple[str, bytes]]):
    for rel, data in files:
        p = Path("/runs") / name / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    runs.commit()
    return len(files)


@app.local_entrypoint()
def upload(region: str = "regions/triplice-fronteira.yaml", loras: str = "loras"):
    import yaml
    name = yaml.safe_load(Path(region).read_text())["name"]
    src = ROOT / "runs" / name
    pares = [(str(p.relative_to(src)), p) for p in src.rglob("*")
             if p.is_file() and p.relative_to(src).parts[0] in ("render", "osm")]
    pares += [(n, src / n) for n in ("view.json", "region.json", "geo_affine.json") if (src / n).exists()]
    # LoRAs vão junto, em runs/<nome>/loras/
    if Path(loras).exists():
        pares += [(f"loras/{lp.name}", lp) for lp in Path(loras).glob("*.safetensors")]
    batch, sent = [], 0
    for rel, p in pares:
        batch.append((rel, p.read_bytes()))
        if sum(len(b) for _, b in batch) > 200e6:
            sent += _receive.remote(name, batch)
            batch = []
    if batch:
        sent += _receive.remote(name, batch)
    print(f"{sent} arquivos enviados para o Volume tinymap-runs/{name}")


@app.function(image=image, gpu=GPU, volumes={"/runs": runs, "/hf": hf_cache}, timeout=24 * 3600)
def worker(region: str, lora_name: str | None, worker_id: int):
    run = _setup(region)
    lora = str(run.dir / "loras" / lora_name) if lora_name else None
    print(f"worker {worker_id} começando")
    model = _model(lora)
    hf_cache.commit()
    from tinymap.generate import worker as gen_worker

    # O Volume só mostra a outro container o que foi commitado e só vê o que foi recarregado:
    # recarrega antes de pegar bloco e enquanto espera fase; commita lock, arte e .done.
    import time as _time

    import tinymap.generate as G
    from tinymap import region as R
    orig_claim, orig_write, orig_sleep = G._claim, R.Canvas.write, _time.sleep

    def claim(lock):
        runs.reload()
        ok = orig_claim(lock)
        runs.commit()
        return ok

    def write(self, origin, img):
        orig_write(self, origin, img)
        runs.commit()

    def sleep(sec):
        runs.commit()
        orig_sleep(sec)
        runs.reload()

    G._claim, R.Canvas.write = claim, write
    G.time.sleep = sleep
    gen_worker(run, model)
    runs.commit()


@app.local_entrypoint()
def generate(region: str = "regions/triplice-fronteira.yaml", lora: str = "", gpus: int = 2):
    name = Path(lora).name if lora else None
    calls = [worker.spawn(region, name, i) for i in range(gpus)]
    print(f"{gpus} workers em {GPU} disparados; acompanhe no painel da Modal")
    for c in calls:
        c.get()
    print("geração terminada; rode o entrypoint download")


@app.function(image=image, gpu=GPU, volumes={"/runs": runs, "/hf": hf_cache}, timeout=3600)
def _try(region: str, lora_name: str | None, lat: float | None, lon: float | None) -> tuple[bytes, bytes]:
    from io import BytesIO
    run = _setup(region)
    from tinymap import geo
    from tinymap.region import render_canvas
    v = run.view
    if lat is not None:
        x, y = geo.to_px(geo.load(run), lat, lon)
    else:
        x, y = v["canvas_px"][0] / 2, v["canvas_px"][1] / 2
    size = 3072
    x0, y0 = int(x - size / 2) // 16 * 16, int(y - size / 2) // 16 * 16
    render = render_canvas(run).crop((x0, y0, x0 + size, y0 + size))
    g = run.region["generate"]
    lora = str(run.dir / "loras" / lora_name) if lora_name else None
    img = _model(lora).joint(render, run.region["style"]["prompt"], steps=g["steps"],
                             window=g["window_px"], stride=g["stride_px"])
    hf_cache.commit()
    out = []
    for im in (render, img):
        b = BytesIO()
        im.save(b, "PNG")
        out.append(b.getvalue())
    return out[0], out[1]


@app.local_entrypoint()
def try_lora(region: str = "regions/triplice-fronteira.yaml", lora: str = "", at: str = "-25.5925,-54.5913"):
    lat, lon = map(float, at.split(",")) if at else (None, None)
    r, a = _try.remote(region, Path(lora).name if lora else None, lat, lon)
    out = ROOT / "runs" / "modal_try"
    out.mkdir(parents=True, exist_ok=True)
    (out / "render.png").write_bytes(r)
    (out / f"{Path(lora).stem if lora else 'base'}.png").write_bytes(a)
    print(f"resultado em {out}")


@app.function(image=image, volumes={"/runs": runs}, timeout=3600)
def _art(name: str) -> list[tuple[str, bytes]]:
    runs.reload()
    d = Path("/runs") / name / "art"
    return [(p.name, p.read_bytes()) for p in d.glob("r*_c*.png")] if d.exists() else []


@app.local_entrypoint()
def download(region: str = "regions/triplice-fronteira.yaml"):
    import yaml
    name = yaml.safe_load(Path(region).read_text())["name"]
    dest = ROOT / "runs" / name / "art"
    dest.mkdir(parents=True, exist_ok=True)
    files = _art.remote(name)
    for fn, data in files:
        (dest / fn).write_bytes(data)
    print(f"{len(files)} shots de arte em {dest}; agora: tinymap site {region}")
