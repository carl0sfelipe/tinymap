"""Local image model on one RTX 3090 (24 GB): Qwen-Image-Edit in 4-bit, with an optional style LoRA.

Two ways to use it:
  edit()   one 1024 px render -> one redraw (teacher pairs, quick tests)
  joint()  MultiDiffusion-style joint sampling over a big block, as tiny.london did: one latent for
           the whole block, each step averages the model's prediction over overlapping windows with a
           tent weight, alternating a grid shifted by half a stride, so there is never a seam. Pixels
           that are already finished (neighbour blocks) are pinned RePaint-style, so new map grows out
           of old map.

The model id is configurable. Default is Qwen/Qwen-Image-Edit-2511 (edit model, diffusers
QwenImageEditPlusPipeline). Any checkpoint with the same pipeline class works the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

DTYPE = torch.bfloat16


def latent_pin(pin_mask: np.ndarray, f: int, free_band_px: int = 64) -> np.ndarray:
    """Máscara de pinning em células latentes (Pixel Singapore): uma célula só fica fixa se TODOS os
    pixels dela são conhecidos, e uma faixa de `free_band_px` ao longo da borda do conhecido fica
    livre, para o modelo costurar o novo com o velho em vez de bater numa parede."""
    h, w = pin_mask.shape
    cells = pin_mask[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f).all(axis=(1, 3))
    band = max(0, free_band_px // f)
    if band:
        # erosão: célula fixa só se todas as vizinhas num raio `band` também são conhecidas
        # fora do recorte conta como conhecido: a borda da imagem não é divisa com pintura nova
        pad = np.pad(cells, band, constant_values=True)
        out = np.ones_like(cells)
        for dy in range(-band, band + 1):
            for dx in range(-band, band + 1):
                out &= pad[band + dy: band + dy + cells.shape[0], band + dx: band + dx + cells.shape[1]]
        cells = out
    return cells


def held_windows(cells: np.ndarray, wins: list[tuple[int, int]], lwin: int, f: int) -> set[tuple[int, int]]:
    """Janelas (x, y em px) inteiramente dentro do que já está pintado: não chamam o transformer
    (no Singapore, 112 de 581 janelas num bloco). O valor delas vem só do pinning."""
    return {(x, y) for (x, y) in wins if cells[y // f: y // f + lwin, x // f: x // f + lwin].all()}


@dataclass
class ModelCfg:
    model_id: str = "Qwen/Qwen-Image-Edit-2511"
    lora: str | None = None
    lora_scale: float = 1.0
    quant: str = "nf4"          # nf4 (fits 24 GB) | none (needs ~60 GB) | offload (bf16 with CPU offload, slow)
    device: str = "cuda:0"
    # Adaptador Lightning (lightx2v, Apache-2.0): o Qwen-Image-Edit-2511 em 4 passos sem CFG. Usado
    # no professor (Isometric SF): ~4x mais pares por hora de GPU. Arquivo dentro do repositório HF.
    lightning: bool = False
    lightning_repo: str = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    lightning_file: str = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


LIGHTNING_STEPS = 4
# Scheduler do Qwen-Image-Lightning (README do lightx2v): shift dinâmico fixo em ln 3.
LIGHTNING_SCHED = dict(base_image_seq_len=256, base_shift=math.log(3), invert_sigmas=False,
                       max_image_seq_len=8192, max_shift=math.log(3), num_train_timesteps=1000, shift=1.0,
                       shift_terminal=None, stochastic_sampling=False, time_shift_type="exponential",
                       use_beta_sigmas=False, use_dynamic_shifting=True, use_exponential_sigmas=False,
                       use_karras_sigmas=False)


class QwenEdit:
    def __init__(self, cfg: ModelCfg):
        from diffusers import QwenImageEditPlusPipeline

        self.cfg = cfg
        kw = {}
        if cfg.quant == "nf4":
            from diffusers import BitsAndBytesConfig as DBnb
            from diffusers import QwenImageTransformer2DModel
            from transformers import BitsAndBytesConfig as TBnb
            from transformers import Qwen2_5_VLForConditionalGeneration

            kw["transformer"] = QwenImageTransformer2DModel.from_pretrained(
                cfg.model_id, subfolder="transformer", torch_dtype=DTYPE, device_map={"": cfg.device},
                quantization_config=DBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE),
            )
            kw["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_id, subfolder="text_encoder", torch_dtype=DTYPE, device_map={"": cfg.device},
                quantization_config=TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE),
            )
        p = QwenImageEditPlusPipeline.from_pretrained(cfg.model_id, torch_dtype=DTYPE, **kw)
        names, weights = [], []
        if cfg.lightning:
            from diffusers import FlowMatchEulerDiscreteScheduler
            p.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHED)
            p.load_lora_weights(cfg.lightning_repo, weight_name=cfg.lightning_file, adapter_name="lightning")
            names.append("lightning"); weights.append(1.0)
        if cfg.lora:
            p.load_lora_weights(cfg.lora, adapter_name="style")
            names.append("style"); weights.append(cfg.lora_scale)
        if names:
            p.set_adapters(names, adapter_weights=weights)
        if cfg.quant == "offload":
            p.enable_model_cpu_offload(gpu_id=torch.device(cfg.device).index or 0)
        else:
            p.to(cfg.device)
        p.vae.enable_tiling() if hasattr(p.vae, "enable_tiling") else None
        p.set_progress_bar_config(disable=True)
        self.p = p
        self.dev = torch.device(cfg.device)
        self.C = p.transformer.config.in_channels // 4
        self.vsf = p.vae_scale_factor

    # ---------- single image ----------
    @torch.no_grad()
    def edit(self, img: Image.Image, prompt: str, steps: int = 20, true_cfg: float = 1.0,
             negative: str | None = None, seed: int = 0) -> Image.Image:
        g = torch.Generator(self.dev).manual_seed(seed)
        if self.cfg.lightning:
            steps, true_cfg = LIGHTNING_STEPS, 1.0
        kw = dict(negative_prompt=negative or " ") if true_cfg > 1 else {}
        out = self.p(image=[img], prompt=prompt, num_inference_steps=steps, true_cfg_scale=true_cfg,
                     height=img.height, width=img.width, generator=g, **kw)
        return out.images[0]

    # ---------- helpers ----------
    def _encode(self, img: Image.Image) -> torch.Tensor:
        """Image -> normalized latent (1, C, h/8, w/8)."""
        x = self.p.image_processor.preprocess(img, img.height, img.width).unsqueeze(2).to(self.dev, DTYPE)
        return self.p._encode_vae_image(x, None)[:, :, 0]

    def _decode(self, z: torch.Tensor) -> np.ndarray:
        """Normalized latent (1, C, h, w) -> float RGB HxWx3 in 0..255."""
        vae = self.p.vae
        mean = torch.tensor(vae.config.latents_mean).view(1, self.C, 1, 1, 1).to(z.device, torch.float32)
        std = torch.tensor(vae.config.latents_std).view(1, self.C, 1, 1, 1).to(z.device, torch.float32)
        # z chega em float32 do loop do joint(); a conta em float e o VAE em bf16
        x = vae.decode((z.unsqueeze(2).float() * std + mean).to(DTYPE), return_dict=False)[0][:, :, 0]
        return ((x[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy()

    def _cond(self, crop: Image.Image, prompt: str, negative: str | None):
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import CONDITION_IMAGE_SIZE, calculate_dimensions

        cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, crop.width / crop.height)
        small = self.p.image_processor.resize(crop, ch, cw)
        pe, pm = self.p.encode_prompt(prompt=prompt, image=[small], device=self.dev)
        npe = npm = None
        if negative is not None:
            npe, npm = self.p.encode_prompt(prompt=negative, image=[small], device=self.dev)
        lat = self._encode(crop)
        il = self.p._pack_latents(lat.unsqueeze(2), 1, self.C, lat.shape[2], lat.shape[3])
        cpu = lambda t: None if t is None else t.to("cpu")  # noqa: E731
        return tuple(map(cpu, (pe, pm, npe, npm, il)))

    # ---------- joint sampling ----------
    @torch.no_grad()
    def joint(self, render: Image.Image, prompt: str, *, pinned: Image.Image | None = None,
              pin_mask: np.ndarray | None = None, steps: int = 20, true_cfg: float = 1.0,
              negative: str | None = None, window: int = 1024, stride: int = 768, seed: int = 0,
              strength: float = 1.0, log=print, ckpt: "Path | str | None" = None,
              free_band_px: int = 64) -> Image.Image:
        """strength < 1: parte do próprio render com ruído parcial (SDEdit) em vez de ruído puro.
        Com 0,4-0,6 o modelo troca a aparência mas não consegue mudar a planta (prédios, ruas, rio).

        ckpt: arquivo onde o latente é salvo a cada passo; se existir, retoma do último passo salvo
        (queda de luz, travamento, Ctrl-C na 3090 não jogam fora uma hora de bloco).
        free_band_px: faixa ao longo da borda do já pintado que fica livre (não fixada)."""
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift, retrieve_timesteps

        p, dev, f = self.p, self.dev, self.vsf
        W, H = render.size
        assert W % window == 0 or W >= window, "block smaller than a window"
        assert W % 16 == 0 and H % 16 == 0
        lw, lh, lwin = W // f, H // f, window // f

        def grid(off: int):
            def starts(n):
                s = list(range(off, max(1, n - window + 1), stride))
                if not s or s[0] != 0:
                    s = [0] + s
                if s[-1] != n - window:
                    s.append(n - window)
                return sorted(set(x for x in s if 0 <= x <= n - window))
            return [(x, y) for y in starts(H) for x in starts(W)]

        grids = [grid(0), grid(stride // 2)]
        wins = sorted(set(grids[0]) | set(grids[1]))
        neg = negative if true_cfg > 1 else None
        held: set = set()
        cells = None
        if pinned is not None and pin_mask is not None and pin_mask.any():
            cells = latent_pin(pin_mask, f, free_band_px)
            held = held_windows(cells, wins, lwin, f)
            wins = [w for w in wins if w not in held]
        log(f"  encoding {len(wins)} windows ({len(held)} já pintadas, puladas)")
        cache = {xy: self._cond(render.crop((xy[0], xy[1], xy[0] + window, xy[1] + window)), prompt, neg) for xy in wins}

        # tent weight: 0 at the window edge, flat across the middle half
        i = np.arange(lwin) + 0.5
        ramp = np.clip(np.minimum(i, lwin - i) / (lwin / 4), 1e-3, 1)
        tent = torch.tensor(np.outer(ramp, ramp), device=dev, dtype=torch.float32)

        # pinned context, encoded in window-sized pieces
        x0 = mask = None
        if pinned is not None and pin_mask is not None and pin_mask.any():
            x0 = torch.zeros((1, self.C, lh, lw), device=dev)
            for (x, y) in grid(0):
                piece = pinned.crop((x, y, x + window, y + window))
                x0[:, :, y // f:y // f + lwin, x // f:x // f + lwin] = self._encode(piece).float()
            mask = torch.from_numpy(cells)[None, None].to(dev)

        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift((lwin // 2) ** 2, p.scheduler.config.get("base_image_seq_len", 256),
                             p.scheduler.config.get("max_image_seq_len", 4096),
                             p.scheduler.config.get("base_shift", 0.5), p.scheduler.config.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(p.scheduler, steps, dev, sigmas=sigmas, mu=mu)
        g = torch.Generator(dev).manual_seed(seed)
        noise = torch.randn((1, self.C, lh, lw), generator=g, device=dev, dtype=torch.float32)
        z = noise.clone()
        k0 = 0
        if strength < 1:
            sig = p.scheduler.sigmas
            k0 = next((k for k in range(len(timesteps)) if float(sig[k]) <= strength), len(timesteps) - 1)
            xr = torch.zeros((1, self.C, lh, lw), device=dev)
            for (x, y) in grid(0):
                xr[:, :, y // f:y // f + lwin, x // f:x // f + lwin] = self._encode(render.crop((x, y, x + window, y + window))).float()
            s0 = float(sig[k0])
            z = s0 * noise + (1 - s0) * xr  # ponto da trajetória flow-matching no nível de ruído s0
            log(f"  strength {strength}: começando no passo {k0 + 1}/{steps} (sigma {s0:.2f})")
        if ckpt is not None:
            ckpt = __import__("pathlib").Path(ckpt)
            if ckpt.exists():
                st = torch.load(ckpt, map_location=dev)
                if tuple(st["z"].shape) == tuple(z.shape) and st.get("steps") == steps:
                    z, noise, k0 = st["z"], st["noise"], st["k"] + 1
                    log(f"  retomando do passo {k0 + 1}/{steps} ({ckpt.name})")
        p.scheduler.set_begin_index(k0)
        shapes = [[(1, lwin // 2, lwin // 2), (1, lwin // 2, lwin // 2)]]
        tr = p.transformer

        for k, t in enumerate(timesteps):
            if k < k0:
                continue
            acc = torch.zeros_like(z)
            wsum = torch.zeros((1, 1, lh, lw), device=dev)
            for (x, y) in grids[k % 2]:
                if (x, y) in held:
                    continue
                lx, ly = x // f, y // f
                pe, pm, npe, npm, il = (None if v is None else v.to(dev) for v in cache[(x, y)])
                zl = z[:, :, ly:ly + lwin, lx:lx + lwin].to(DTYPE).contiguous()
                packed = p._pack_latents(zl.unsqueeze(2), 1, self.C, lwin, lwin)
                inp = torch.cat([packed, il], 1)
                ts = t.expand(1).to(DTYPE) / 1000
                v = tr(hidden_states=inp, timestep=ts, guidance=None, encoder_hidden_states=pe,
                       encoder_hidden_states_mask=pm, img_shapes=shapes, return_dict=False)[0][:, :packed.size(1)]
                if npe is not None:
                    nv = tr(hidden_states=inp, timestep=ts, guidance=None, encoder_hidden_states=npe,
                            encoder_hidden_states_mask=npm, img_shapes=shapes, return_dict=False)[0][:, :packed.size(1)]
                    comb = nv + true_cfg * (v - nv)
                    v = comb * (v.norm(dim=-1, keepdim=True) / comb.norm(dim=-1, keepdim=True))
                v = p._unpack_latents(v, window, window, f)[:, :, 0].float()
                acc[:, :, ly:ly + lwin, lx:lx + lwin] += v * tent
                wsum[:, :, ly:ly + lwin, lx:lx + lwin] += tent
            # células cobertas só por janelas puladas ficam sem velocidade; o pinning abaixo as preenche
            z = p.scheduler.step(acc / wsum.clamp_min(1e-6), t, z, return_dict=False)[0]
            if x0 is not None:
                s_next = float(p.scheduler.sigmas[k + 1])
                z = torch.where(mask, s_next * noise + (1 - s_next) * x0, z)
            if ckpt is not None:
                tmp = ckpt.with_suffix(".tmp")
                torch.save({"z": z, "noise": noise, "k": k, "steps": steps}, tmp)
                tmp.replace(ckpt)
            log(f"  step {k + 1}/{steps}")

        # decode in overlapping windows and blend, so the VAE adds no seams either
        out = np.zeros((H, W, 3), np.float32)
        wpx = np.zeros((H, W, 1), np.float32)
        tent_px = np.kron(np.outer(ramp, ramp), np.ones((f, f)))[..., None].astype(np.float32)
        for (x, y) in grids[0]:
            rgb = self._decode(z[:, :, y // f:y // f + lwin, x // f:x // f + lwin])
            out[y:y + window, x:x + window] += rgb * tent_px
            wpx[y:y + window, x:x + window] += tent_px
        img = Image.fromarray(np.clip(out / np.maximum(wpx, 1e-6), 0, 255).astype(np.uint8))
        if pinned is not None and pin_mask is not None:
            from PIL import ImageFilter
            # keep finished pixels exactly, with a 32 px feather inside the pinned edge
            m = Image.fromarray((pin_mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(33))
            img.paste(pinned, (0, 0), m.filter(ImageFilter.GaussianBlur(8)))
        return img
