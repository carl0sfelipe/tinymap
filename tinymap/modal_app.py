"""Rodar o tinymap nas GPUs da Modal em vez (ou além) das 3090 locais.

Do seu computador, com o token da Modal configurado (`modal token set ...`):

    modal run tinymap/modal_app.py::upload   --region regions/triplice-fronteira.yaml
    modal run tinymap/modal_app.py::try_lora --region regions/triplice-fronteira.yaml --lora loras/v2.safetensors
    modal run tinymap/modal_app.py::generate --region regions/triplice-fronteira.yaml --lora loras/v2.safetensors --gpus 4
    modal run tinymap/modal_app.py::download --region regions/triplice-fronteira.yaml

`--pilot` em qualquer entrypoint usa o recorte piloto (runs/<nome>-pilot, região com pilot=True).

O render (Google 3D Tiles) e os pares de treino continuam sendo feitos antes, localmente:
`upload` envia a pasta runs/<região> (render, osm, view.json) para um Volume da Modal; cada worker
de `generate` é um container com GPU que roda exatamente o mesmo `generate.worker` das 3090 —
os lock files no Volume coordenam os blocos, então dá para misturar Modal e 3090 se a pasta for
sincronizada. `download` traz a arte pintada de volta para runs/<região>/art.

Na H100 (80 GB) o modelo roda em bf16 sem quantização: mais rápido e um pouco melhor que NF4.
"""


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
        "torch>=2.4", "torchvision", "diffusers>=0.36", "transformers>=4.51", "accelerate", "peft>=0.17",
        "bitsandbytes>=0.45", "safetensors", "sentencepiece", "pyyaml", "numpy", "pillow", "pyvips",
    )
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "tinymap", "/app/tinymap")
    .add_local_dir(ROOT / "regions", "/app/regions")
    .add_local_dir(ROOT / "render", "/app/render", ignore=["node_modules"])
)
GPU = "H100"
MODEL = "Qwen/Qwen-Image-Edit-2511"

# Professor em lote: Qwen-Image-2.1 (edição + geração, 7B; licença Qwen RESEARCH, só uso não comercial).
# Imagem separada para não mexer nas versões que rodam o 2511.
image21 = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", "torchvision", "diffusers>=0.42.1", "transformers>=4.57", "accelerate",
                 "safetensors", "pillow")
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1"})
)
TEACHER = "Qwen/Qwen-Image-2.1"


def _run_name(region: str, pilot: bool) -> str:
    """Nome da pasta em runs/ (e no Volume), igual ao de region.load: <name> ou <name>-pilot."""
    import yaml
    return yaml.safe_load(Path(region).read_text())["name"] + ("-pilot" if pilot else "")


def _setup(region_file: str, pilot: bool = False):
    """Aponta o tinymap para o Volume (runs/) e carrega a região."""
    import os
    import sys
    sys.path.insert(0, "/app")
    os.makedirs("/runs", exist_ok=True)
    if not os.path.exists("/app/runs"):
        os.symlink("/runs", "/app/runs")
    from tinymap import region as R
    return R.load(f"/app/{region_file}", pilot=pilot)


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
def upload(region: str = "regions/triplice-fronteira.yaml", loras: str = "loras", pilot: bool = False):
    name = _run_name(region, pilot)
    src = ROOT / "runs" / name
    # o container não tem o node_modules do render: a projeção lat/lon -> px vai pronta
    from tinymap import geo
    from tinymap import region as R
    geo.load(R.load(region, pilot=pilot))
    pares = [(str(p.relative_to(src)), p) for p in src.rglob("*")
             if p.is_file() and p.relative_to(src).parts[0] in ("render", "osm")
             and not p.relative_to(src).parts[-1].startswith("sat")  # satélite já está no render
             and "sat_tiles" not in p.relative_to(src).parts]
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
def worker(region: str, lora_name: str | None, worker_id: int, pilot: bool = False):
    run = _setup(region, pilot)
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
def generate(region: str = "regions/triplice-fronteira.yaml", lora: str = "", gpus: int = 2, pilot: bool = False):
    name = Path(lora).name if lora else None
    calls = [worker.spawn(region, name, i, pilot) for i in range(gpus)]
    print(f"{gpus} workers em {GPU} disparados; acompanhe no painel da Modal")
    for c in calls:
        c.get()
    print("geração terminada; rode o entrypoint download")


@app.function(image=image, gpu=GPU, volumes={"/runs": runs, "/hf": hf_cache}, timeout=3600)
def _try(region: str, lora_name: str | None, lat: float | None, lon: float | None,
         pilot: bool = False, steps: int = 0, true_cfg: float = 0, prompt: str = "",
         single: str = "", strength: float = 1.0) -> tuple[bytes, bytes]:
    import time
    from io import BytesIO
    run = _setup(region, pilot)
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
    g, style = run.region["generate"], run.region["style"]
    lora = str(run.dir / "loras" / lora_name) if lora_name else None
    # 0 / "" = o que está no YAML da região
    steps, true_cfg = steps or g["steps"], true_cfg or g.get("true_cfg", 1.0)
    model = _model(lora)
    t0 = time.time()
    if single:
        # diagnóstico: o mesmo recorte de 1024 px pelo pipeline oficial (esquerda) e pelo joint() de
        # uma janela só (direita). Se só o joint() errar, o bug é dele; se os dois errarem, é modelo/prompt.
        from PIL import Image
        sx, sy = map(int, single.split(","))
        render = render.crop((sx, sy, sx + g["window_px"], sy + g["window_px"]))
        neg = style.get("negative") if true_cfg > 1 else None
        a = model.edit(render, prompt or style["prompt"], steps=steps, true_cfg=true_cfg, negative=neg)
        b = model.joint(render, prompt or style["prompt"], steps=steps, true_cfg=true_cfg,
                        negative=style.get("negative"), window=g["window_px"], stride=g["stride_px"],
                        strength=strength)
        img = Image.new("RGB", (2 * render.width, render.height))
        img.paste(a, (0, 0))
        img.paste(b, (render.width, 0))
    else:
        img = model.joint(render, prompt or style["prompt"], steps=steps, true_cfg=true_cfg,
                          negative=style.get("negative"), window=g["window_px"], stride=g["stride_px"],
                          strength=strength)
    print(f"{'edit+joint' if single else 'joint'}: {steps} passos, true_cfg {true_cfg}, {time.time() - t0:.0f} s")
    hf_cache.commit()
    out = []
    for im in (render, img):
        b = BytesIO()
        im.save(b, "PNG")
        out.append(b.getvalue())
    return out[0], out[1]


@app.local_entrypoint()
def try_lora(region: str = "regions/triplice-fronteira.yaml", lora: str = "", at: str = "-25.5925,-54.5913",
             pilot: bool = False, steps: int = 0, true_cfg: float = 0, prompt: str = "", tag: str = "",
             single: str = "", strength: float = 1.0):
    """steps/true_cfg/prompt sobrescrevem o YAML para comparar; tag entra no nome do arquivo.
    --strength 0.5 parte do render com ruído parcial (troca o estilo sem redesenhar a planta)."""
    lat, lon = map(float, at.split(",")) if at else (None, None)
    r, a = _try.remote(region, Path(lora).name if lora else None, lat, lon, pilot, steps, true_cfg, prompt, single,
                       strength)
    out = ROOT / "runs" / "modal_try"
    out.mkdir(parents=True, exist_ok=True)
    # no diagnóstico --single o render é só o recorte de 1024 px
    (out / (f"render_{tag or 'single'}.png" if single else "render.png")).write_bytes(r)
    (out / f"{Path(lora).stem if lora else 'base'}{'_' + tag if tag else ''}.png").write_bytes(a)
    print(f"resultado em {out}")


@app.cls(image=image21, gpu=GPU, volumes={"/hf": hf_cache}, timeout=4 * 3600, max_containers=4,
         scaledown_window=120)
class Teacher:
    """Carrega o modelo professor uma vez por container; cada recorte volta assim que fica pronto.
    Qwen-Image-Edit-* usa o QwenImageEditPlusPipeline (Apache-2.0); o resto, o QwenImage21Pipeline."""
    model_id: str = modal.parameter(default=TEACHER)

    @modal.enter()
    def load(self):
        import torch
        self.edit = "Edit" in self.model_id
        if self.edit:
            from diffusers import QwenImageEditPlusPipeline as P
        else:
            from diffusers import QwenImage21Pipeline as P
        self.pipe = P.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        hf_cache.commit()

    @modal.method()
    def redraw(self, fid: str, png: bytes, prompt: str, steps: int, seeds: list[int]) -> list[tuple[str, int, bytes]]:
        import time
        from io import BytesIO

        import torch
        from PIL import Image
        img = Image.open(BytesIO(png)).convert("RGB")
        out = []
        for s in seeds:
            t0 = time.time()
            g = torch.Generator("cuda").manual_seed(s)
            if self.edit:
                res = self.pipe(image=[img], prompt=prompt, width=img.width, height=img.height,
                                num_inference_steps=steps, true_cfg_scale=1.0, generator=g).images[0]
            else:
                res = self.pipe(prompt=prompt, image=img, width=img.width, height=img.height,
                                num_inference_steps=steps, generator=g).images[0]
            b = BytesIO()
            res.save(b, "PNG")
            out.append((fid, s, b.getvalue()))
            print(f"{fid} seed {s} {time.time() - t0:.1f}s", flush=True)
        return out


@app.local_entrypoint()
def teacher(region: str = "regions/triplice-fronteira.yaml", pilot: bool = False, out: str = "pairs_v1",
            n: int = 120, cands: int = 2, gpus: int = 4, steps: int = 0, smoke: int = 0,
            model: str = "", seed: int = 0, new: int = 0):
    """Lote de pares de treino com o Qwen-Image-2.1, no ciclo com revisão humana (tinymap review).

    Redesenha: recortes ainda sem candidato + os marcados "refazer" na revisão (com a dica no prompt).
    --new 20 acrescenta 20 recortes novos. --smoke 4 faz só 4 para medir tempo/custo.
    Cada candidato é gravado assim que chega (<id>_cand<k>.png); no fim, os checks dão notas
    (checks.json), que a página de revisão mostra como sugestão."""
    import json
    from io import BytesIO

    from tinymap import region as R
    from tinymap import teacher as T
    run = R.load(region, pilot=pilot)
    d = run.dir / out
    # modelo e passos: argumento > YAML (style.teacher_model / generate.teacher_steps) > padrão
    model = model or run.region["style"].get("teacher_model") or TEACHER
    steps = steps or run.region["generate"].get("teacher_steps") or (20 if "Edit" in model else 40)
    print(f"professor: {model}, {steps} passos", flush=True)
    if new and (d / "frames.json").exists():
        print(f"{len(T.extend(run, d, new, seed))} recortes novos")
    jobs = T.prepare(run, d, n, seed)
    if smoke:
        jobs = jobs[:smoke]
    todo = [j for j in jobs if j["status"] == "redo" or (j["status"] != "discard" and not T.candidates(d, j["id"]))]
    print(f"{len(todo)} recortes para redesenhar ({cands} candidatos cada) em até {gpus} {GPU}", flush=True)

    def png(im):
        b = BytesIO()
        im.save(b, "PNG")
        return b.getvalue()

    if todo:
        args = []
        for j in todo:
            k0 = len(T.candidates(d, j["id"]))
            args.append((j["id"], png(j["image"]), j["prompt"], steps, [seed + k0 + k for k in range(cands)]))
        tm = Teacher.with_options(max_containers=gpus)(model_id=model)
        done = 0
        for res in tm.redraw.starmap(args, order_outputs=False):
            for fid, s, data in res:
                k = len(T.candidates(d, fid))
                (d / f"{fid}_cand{k}.png").write_bytes(data)
            done += 1
            print(f"recebido {done}/{len(todo)}: {res[0][0] if res else '?'}", flush=True)
        # quem foi refeito volta para a fila de revisão
        rv = T.load_review(d)
        for j in todo:
            if rv.get(j["id"], {}).get("status") == "redo":
                rv[j["id"]]["status"] = "todo"
        (d / "review.json").write_text(json.dumps(rv, indent=1, ensure_ascii=False))
    T.score(run, d)
    print("lote pronto; revise em: tinymap review " + region + (" --pilot" if pilot else ""))


@app.function(image=image, volumes={"/runs": runs}, timeout=3600)
def _art(name: str) -> list[tuple[str, bytes]]:
    runs.reload()
    d = Path("/runs") / name / "art"
    return [(p.name, p.read_bytes()) for p in d.glob("r*_c*.png")] if d.exists() else []


@app.local_entrypoint()
def download(region: str = "regions/triplice-fronteira.yaml", pilot: bool = False):
    name = _run_name(region, pilot)
    dest = ROOT / "runs" / name / "art"
    dest.mkdir(parents=True, exist_ok=True)
    files = _art.remote(name)
    for fn, data in files:
        (dest / fn).write_bytes(data)
    print(f"{len(files)} shots de arte em {dest}; agora: tinymap site {region}")
