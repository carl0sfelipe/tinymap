"""Turn accepted teacher pairs into an ai-toolkit dataset + LoRA config for Qwen-Image-Edit.

tiny.london: rank 32, lr 1e-4, batch 1, 8,000 steps on 6,326 pairs (one H100, 3h50).
On one 3090 the same recipe runs roughly 3-4x slower, so start with ~300-800 pairs and 3,000-4,000 steps.

ai-toolkit (https://github.com/ostris/ai-toolkit) changes its model options between versions.
If `arch` or `qtype` below is refused, copy the `model:` block from its
config/examples/train_lora_qwen_image_edit*_24gb.yaml and keep the rest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from PIL import Image

from .region import Run


def build(run: Run, pairs_dirs: list[Path], name: str, base: str, steps: int, rank: int,
          lr: float = 1e-4, save_every: int = 1000) -> Path:
    ds = run.path("lora", name, "dataset")
    tgt, ctl = ds / "target", ds / "control"
    for d in (tgt, ctl):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    prompt = run.region["style"]["prompt"]
    from .lab import test_ids
    teste = test_ids(run)  # recortes do laboratório nunca entram no treino
    n = fora = 0
    for pd in pairs_dirs:
        for art in sorted(Path(pd).glob("*_art.png")):
            fid = art.name[:-8]
            if f"{Path(pd).name}/{fid}" in teste:
                fora += 1
                continue
            render = art.with_name(f"{fid}_render.png")
            if not render.exists():
                continue
            a, r = Image.open(art).convert("RGB"), Image.open(render).convert("RGB")
            if a.width < 1024:  # zoomed quarters: train at 1024 too, the model sees 2x detail
                a = a.resize((1024, 1024), Image.NEAREST)
            if r.size != a.size:  # arte da revisão (tinymap review) já vem em 1024 px
                r = r.resize(a.size, Image.LANCZOS)
            key = f"{Path(pd).name}_{fid}"
            a.save(tgt / f"{key}.png")
            r.save(ctl / f"{key}.png")
            (tgt / f"{key}.txt").write_text(prompt)
            n += 1
    cfg = {
        "job": "extension",
        "config": {
            "name": f"{run.name}_{name}",
            "process": [{
                "type": "diffusion_trainer",
                "training_folder": str(run.path("lora", name, "out")),
                "device": "cuda:0",
                "network": {"type": "lora", "linear": rank, "linear_alpha": rank},
                "save": {"dtype": "bf16", "save_every": save_every, "max_step_saves_to_keep": 20},
                "datasets": [{
                    "folder_path": str(tgt), "control_path": str(ctl), "caption_ext": "txt",
                    "resolution": [512, 768, 1024], "cache_latents_to_disk": True,
                }],
                "train": {
                    "batch_size": 1, "steps": steps, "gradient_accumulation": 1,
                    "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch", "optimizer": "adamw8bit", "lr": lr,
                    "dtype": "bf16", "cache_text_embeddings": True, "timestep_type": "weighted",
                },
                "model": {
                    "name_or_path": base, "arch": "qwen_image_edit_plus",
                    "quantize": True, "qtype": "qfloat8", "quantize_te": True, "qtype_te": "qfloat8",
                    "low_vram": True,
                },
                "sample": {"sampler": "flowmatch", "sample_every": 500, "width": 1024, "height": 1024,
                           "samples": [], "neg": "", "seed": 42, "guidance_scale": 1, "sample_steps": 20},
            }],
        },
    }
    cfg_path = run.path("lora", name, "train.yaml")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    (cfg_path.parent / "lab.json").write_text(json.dumps(
        {"base": base, "rank": rank, "lr": lr, "steps": steps, "pairs": n, "held_out": fora}))
    print(f"{n} pairs ({fora} do teste ficaram de fora) -> {ds}\nconfig -> {cfg_path}\n"
          f"run:  cd ai-toolkit && CUDA_VISIBLE_DEVICES=0 python run.py {cfg_path}")
    return cfg_path
