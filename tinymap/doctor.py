"""tinymap doctor: confere a máquina antes de gastar hora de GPU alugada.

    tinymap doctor regions/belo-horizonte.yaml

Cada linha diz OK, AVISO ou FALTA, com o que fazer. Sai com código 1 se faltar algo obrigatório
para o caminho da região (render do Google, geração e treino na GPU).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

from .region import ROOT


def _env() -> dict:
    vals = dict(os.environ)
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                vals.setdefault(k.strip(), v.split("#")[0].strip().strip('"').strip("'"))
    return vals


def _gpu() -> list[tuple[str, int]]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return [(n.strip(), int(m)) for n, m in (l.split(",") for l in out.strip().splitlines() if "," in l)]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def check(region: dict | None = None) -> int:
    rows: list[tuple[str, str, str]] = []
    env = _env()
    src = (region or {}).get("source", "google")

    gpus = _gpu()
    if not gpus:
        rows.append(("FALTA", "GPU NVIDIA", "nvidia-smi não respondeu: sem GPU não há geração nem treino"))
    for name, mem in gpus:
        if mem >= 23000:
            rows.append(("OK", f"GPU {name}", f"{mem // 1024} GB"))
        else:
            rows.append(("FALTA", f"GPU {name}", f"{mem // 1024} GB: precisa de 24 GB (3090/4090/A5000+)"))

    try:
        import torch
        ok = torch.cuda.is_available()
        rows.append(("OK" if ok else "FALTA", f"torch {torch.__version__}", "CUDA ativo" if ok else "torch sem CUDA"))
    except ImportError:
        rows.append(("FALTA", "torch", "pip install -e '.[lab]'"))

    mem_gb = 0
    try:
        mem_gb = int(Path("/proc/meminfo").read_text().split()[1]) // 1024 // 1024
    except (OSError, IndexError, ValueError):
        pass
    rows.append(("OK" if mem_gb >= 60 else "AVISO", f"RAM {mem_gb} GB",
                 "ok" if mem_gb >= 60 else "o Qwen 20B passa pela RAM ao quantizar em NF4: prefira 64 GB+"))

    free = shutil.disk_usage(ROOT).free // 2**30
    rows.append(("OK" if free >= 150 else "FALTA", f"disco livre {free} GB",
                 "ok" if free >= 150 else "modelo (~60 GB) + cache do treino + render: peça 200 GB"))

    for mod, why in (("diffusers", "geração"), ("transformers", "geração"), ("bitsandbytes", "NF4 na 3090"),
                     ("peft", "LoRA"), ("pyvips", "site (pirâmide)"), ("scipy", "estúdio e autofix"),
                     ("shapely", "estúdio"), ("lpips", "laboratório (pip install -e '.[lab]')")):
        rows.append(("OK" if importlib.util.find_spec(mod) else "FALTA", mod, why))

    node = shutil.which("node")
    if node:
        v = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        rows.append(("OK" if int(v.lstrip("v").split(".")[0]) >= 20 else "FALTA", f"node {v}", "render"))
    else:
        rows.append(("FALTA", "node", "Node 20+ para o render"))
    rows.append(("OK" if (ROOT / "render" / "node_modules").exists() else "FALTA", "render/node_modules",
                 "cd render && npm install"))
    pw = Path.home() / ".cache" / "ms-playwright"
    chrome = env.get("CHROME_PATH")
    rows.append(("OK" if (pw.exists() and any(pw.glob("chromium*"))) or (chrome and Path(chrome).exists()) else "FALTA",
                 "Chromium do render", "cd render && npx playwright install --with-deps chromium"))

    if src == "google":
        has = env.get("GOOGLE_TILES_API_KEY") or env.get("CESIUM_ION_TOKEN")
        rows.append(("OK" if has else "FALTA", "chave do Google 3D Tiles",
                     "GOOGLE_TILES_API_KEY (Map Tiles API) ou CESIUM_ION_TOKEN no .env"))
    ai = ROOT.parent / "ai-toolkit"
    rows.append(("OK" if (ai / "run.py").exists() else "AVISO", "ai-toolkit", f"treino da LoRA ({ai})"))
    fac = (region or {}).get("fachada", {})
    rows.append(("OK" if env.get("MAPILLARY_TOKEN") else "AVISO", "MAPILLARY_TOKEN", "fachadas: fotos de rua abertas (opcional)"))
    if int(fac.get("streetview_max_mes", 0)) > 0:
        rows.append(("OK" if env.get("GOOGLE_MAPS_API_KEY") or env.get("GOOGLE_TILES_API_KEY") else "FALTA",
                     "chave do Street View", "GOOGLE_MAPS_API_KEY com a Street View Static API ativada"))
    hf = env.get("HF_TOKEN")
    rows.append(("OK" if hf else "AVISO", "HF_TOKEN", "downloads do Hugging Face mais rápidos e sem limite"))

    w = max(len(r[1]) for r in rows)
    for st, name, msg in rows:
        print(f"{st:6s} {name:{w}s}  {msg}")
    falta = [r for r in rows if r[0] == "FALTA"]
    print(f"\n{len(falta)} faltando" if falta else "\npronto")
    return 1 if falta else 0
