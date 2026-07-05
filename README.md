
# HyperScale AI

free topaz photo ai alternative that runs on your gpu. no account. no $200/year. just actual good upscaling.

> **[→ try it at hyperscale-ai-amsachs-projects.vercel.app](https://hyperscale-ai-amsachs-projects.vercel.app)**

![before and after comparison — climber photo upscaled 4x with HAT-GAN, skin pores visible at 100% zoom](climber_out_hat_only.png)

---

## quick start

```powershell
python -m venv venv && venv\Scripts\activate
python check_and_install_deps.py && python download_all_models.py
run_server.bat
```

open `localhost:8080`. drop a photo. hit upscale. that's it.

(first run downloads a few GB of model weights. after that it's instant.)

---

## what it does

- **4x to 8x upscaling** on your GPU, 100% offline, no upload
- **HAT-GAN** (Hybrid Attention Transformer) as the default engine — looks at big patches of the image and reasons about texture instead of averaging from a lookup table. portraits look like portraits, not porcelain
- **Stable Diffusion ControlNet Tile** pass for extreme upscales — synthesizes realistic micro-texture (skin pores, fabric weave, leaf veins) while keeping your original colors and shapes locked
- **GFPGAN v1.4 face pass** runs automatically — preserves eye symmetry, lip shape, skin tone so faces don't wax out
- **dynamic VRAM tiling** — adjusts tile size based on how much free VRAM you actually have right now, not just what your card has in total. works on 6GB cards
- **before/after comparison slider** at full native resolution (not a cropped thumbnail)
- **CLI version** if you'd rather drag a file into a terminal

---

## how it works (the part i'm actually proud of)

standard upscalers learn average textures from training data and paste them onto your image. looks sharp from a distance, falls apart under zoom — skin turns to porcelain, fabric loses weave, hair clumps into plastic strands. that's the "topaz look" people talk about.

HAT-GAN fixes the averaging problem. it uses self-attention to look at wide patches of the image at once, same core idea as GPT but for pixels, so it figures out what *should* be in the fine detail based on context instead of blending a lookup table.

for the SD pass, the naive approach is to run Stable Diffusion on the output after upscaling — but SD is generative, it wants to change things, so you end up with slightly wrong eyebrows and color drift. the fix is frequency separation: throw away SD's low frequencies entirely (keep the original's colors and shapes) and only inject SD's high-frequency micro-texture layer. result: structure 100% preserved, but now there are actual pores visible when you zoom in.

i also re-ordered the pipeline so SD refinement runs *before* the final HAT upscale instead of after it. same output quality. 3 minutes instead of 30. figured that out at 1am and it genuinely made my day.

---

## models

all weights go into `/models/` on first run:

| model | good for |
|---|---|
| `Real_HAT_GAN_sharper.pth` | default. best for real-world photos |
| `HAT_SRx4_ImageNet-pretrain.pth` | more faithful, less aggressive |
| `GFPGANv1.4.pth` | face restoration (runs automatically) |
| `RealESRGAN_x4plus.pth` | fallback, general purpose |
| `RealESRGAN_x4plus_anime_6B.pth` | anime / line art |
| `realesr-general-x4v3.pth` | fast, compact, good for batch |

---

## requirements

- python 3.10+
- NVIDIA GPU with 6GB+ VRAM recommended (works on CPU but slow)
- CUDA 11.8+

do whatever you want with this i dont care lol.
