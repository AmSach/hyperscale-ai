

```
  ▲    █   █ █   █ █▀▀▀█ █▀▀▀ █▀▀▀▄ █▀▀▀▀ █▀▀▀█ █▀▀▀▄ █     █▀▀▀▀   ▲
  ★   █▀▀▀█ ▀▄▄▄▀ █▄▄▄█ █▀▀▀ █▄▄▄▀ ▀▀▀▀█ █     █▄▄▄█ █     █▀▀▀    ★
  ●   █   █   █   █     █▄▄▄ █   █ ▄▄▄▄█ █▄▄▄█ █   █ █▄▄▄▄ █▄▄▄▄   ●
```

okay so i got really fed up with topaz and all those $200/year cloud upscalers turning photos into this weird smooth plastic garbage. every portrait looked like it was rendered in a video game cutscene from 2009. so i spent a few weekends wiring up actual state-of-the-art vision transformers to run directly on your GPU, offline, for free.

this runs 100% on your machine. no account, no subscription, no "processing queue", no uploading your photos to some server in oregon.

---

## what it actually does

standard upscalers (RealESRGAN, Topaz Photo AI at default settings, etc.) work by learning average textures from training data and pasting them onto your image. the result looks sharp from a distance but falls apart under a zoom — skin turns to porcelain, fabric loses weave detail, hair clumps into plastic strands.

**HyperScale uses HAT-GAN (Hybrid Attention Transformer) as the default engine.** HAT looks at wide patches of the image at once using self-attention, same core idea as GPT but for pixels. it figures out what *should* be in the fine details based on structure and context instead of averaging it out. the difference on portraits and landscapes is night and day.

for extreme upscaling (8x to 16x), there is a second pass using Stable Diffusion ControlNet Tile. it synthesizes realistic high-frequency texture while keeping the geometry and color locked to the original — so it is not hallucinating a completely different image, it is filling in what *plausibly belongs there*.

faces get special treatment. a GFPGAN v1.4 pass runs automatically on detected faces to preserve eye symmetry, lip shape, and skin tone so they do not end up looking like wax figures.

the server also handles VRAM tiling dynamically so it does not crash on 6GB or 8GB cards when you throw a large image at it.

---

## setup

the installer handles everything — Python, venv, PyTorch, CUDA, all the model weights. you just run one command:

```powershell
python -c "import urllib.request, subprocess; open('install.ps1', 'wb').write(urllib.request.urlopen('https://hyperscale-ai-amsachs-projects.vercel.app/install.ps1').read()); subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'install.ps1'])"
```

or download [`install.bat`](https://hyperscale-ai-amsachs-projects.vercel.app/install.bat) and double-click it. same result.

if you want to do it yourself:

```bash
git clone https://github.com/AmSach/hyperscale-ai.git
cd hyperscale-ai
python -m venv venv
venv\Scripts\activate
python check_and_install_deps.py
python download_all_models.py
run_server.bat
```

---

## the interface

opens at `http://localhost:8080`. there is a full-frame before/after comparison slider — not a cropped thumbnail, the actual full image — so you can see what is happening at every part of the photo. live inference speed and VRAM usage shown in real time.

---

## models downloaded during setup

all weights go into `/models/` on first run:

- `Real_HAT_GAN_sharper.pth` — the main engine, runs by default
- `HAT_SRx4_ImageNet-pretrain.pth` — more faithful to original content, less aggressive
- `GFPGANv1.4.pth` — face restoration
- `RealESRGAN_x4plus.pth` and `RealESRGAN_x4plus_anime_6B.pth` — fallback options
- `realesr-general-x4v3.pth` — faster, lighter, good for batch jobs

---

MIT License. do whatever you want with it.
