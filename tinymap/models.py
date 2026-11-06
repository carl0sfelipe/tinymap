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


@dataclass
class ModelCfg:
    model_id: str = "Qwen/Qwen-Image-Edit-2511"
    lora: str | None = None
    lora_scale: float = 1.0
    quant: str = "nf4"          # nf4 (fits 24 GB) | none (needs ~60 GB) | offload (bf16 with CPU offload, slow)
    device: str = "cuda:0"


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
        if cfg.lora:
            p.load_lora_weights(cfg.lora, adapter_name="style")
            p.set_adapters(["style"], adapter_weights=[cfg.lora_scale])
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
              log=print) -> Image.Image:
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
        log(f"  encoding {len(wins)} windows")
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
            m = torch.tensor(pin_mask, dtype=torch.float32)[None, None]
            m = -torch.nn.functional.max_pool2d(-m, f)  # a latent cell is pinned only if all its pixels are
            mask = (m > 0.5).to(dev)

        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift((lwin // 2) ** 2, p.scheduler.config.get("base_image_seq_len", 256),
                             p.scheduler.config.get("max_image_seq_len", 4096),
                             p.scheduler.config.get("base_shift", 0.5), p.scheduler.config.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(p.scheduler, steps, dev, sigmas=sigmas, mu=mu)
        p.scheduler.set_begin_index(0)
        g = torch.Generator(dev).manual_seed(seed)
        noise = torch.randn((1, self.C, lh, lw), generator=g, device=dev, dtype=torch.float32)
        z = noise.clone()
        shapes = [[(1, lwin // 2, lwin // 2), (1, lwin // 2, lwin // 2)]]
        tr = p.transformer

        for k, t in enumerate(timesteps):
            acc = torch.zeros_like(z)
            wsum = torch.zeros((1, 1, lh, lw), device=dev)
            for (x, y) in grids[k % 2]:
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
            z = p.scheduler.step(acc / wsum, t, z, return_dict=False)[0]
            if x0 is not None:
                s_next = float(p.scheduler.sigmas[k + 1])
                z = torch.where(mask, s_next * noise + (1 - s_next) * x0, z)
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
