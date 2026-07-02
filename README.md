

```
  ▲    █   █ █   █ █▀▀▀█ █▀▀▀ █▀▀▀▄ █▀▀▀▀ █▀▀▀█ █▀▀▀▄ █     █▀▀▀▀   ▲
  ★   █▀▀▀█ ▀▄▄▄▀ █▄▄▄█ █▀▀▀ █▄▄▄▀ ▀▀▀▀█ █     █▄▄▄█ █     █▀▀▀    ★
  ●   █   █   █   █     █▄▄▄ █   █ ▄▄▄▄█ █▄▄▄█ █   █ █▄▄▄▄ █▄▄▄▄   ●
```

> **"Okay so basically I got sick of standard upscalers turning every human portrait into a weird melted plastic Barbie doll, so I locked myself in my bedroom with 4 cans of soda and wired up a state-of-the-art Hybrid Attention Transformer directly to my NVIDIA RTX card."**  
> *-- Lab Notes, Entry #404*

Welcome to **HyperScale AI Studio**, a 100% locally-hosted, absurdly powerful neural upscaling and texture reconstruction rig engineered for Windows rigs. It runs offline on your GPU, takes tiny crusty JPEGs, and injects up to **127+ Megapixels** of crispy DSLR detail without phoning home to some sketchy cloud server.

---

## [!] THE SECRET SAUCE // HOW IT WORKS

### [1] The Flagship Beast: HAT-GAN (hat-sharper)
Forget RealESRGAN -- that stuff is like 2021 dinosaur tech. We set our flagship engine to **HAT-GAN (Hybrid Attention Transformer)**. Instead of guessing pixel values and smoothing over skin pores, HAT uses wide attention spans to synthesize legitimate micro-detail. We're talking real skin pores, individual stray eyelashes, film grain, and fabric fuzz that looks legit enough to pass under a microscope.

### [2] Generative ControlNet Tile (Extreme Synthesis Mode)
Need to blow up a tiny 256x256 avatar into an 8K poster? When you switch into **Extreme Synthesis (8x-16x)** mode, our custom pipeline fires off two passes:
* **Pass 1:** Cleans up edge boundaries and removes JPEG mosquito noise.
* **Pass 2:** Uses Stable Diffusion ControlNet Tile to hallucinate crisp, hyper-realistic high-frequency textures while locking the original colors so your sky doesn't randomly turn neon green.

### [3] GFPGAN v1.4 Face Shield (DECRYPT MEDS)
Sometimes AI upscalers get confused by blurry teeth and turn your grandma into an alien. Not here. Whenever our scanner detects a face, it deploys a **GFPGAN v1.4 Face Shield** that protects eye symmetry, pupils, and lip outlines like a protective forcefield.

### [4] Dynamic VRAM Slicing (WALLET DAMAGE CONTROL)
Got an 8GB or 6GB GPU? No sweat. Our custom server script dynamically calculates tile overlap grids and auto-slices the tensor math so your PC doesn't blue-screen into oblivion when rendering a 10,000x10,000 canvas. Zero subscription costs. Zero cloud queues. 100% yours.

---

## [>>>] HOW TO FIRE IT UP (ZERO FRICTION SETUP)

Pick your poison. We made this thing so easy to install my cat could run it on a potato computer:

### Option A: The "Copy-Paste Hacker" One-Liner (Recommended)
Open Windows Command Prompt or PowerShell, paste this command, and hit enter. It checks your GPU, grabs Python 3.12 automatically if you don't have it, downloads all the neural weights into `/models/`, drops a shortcut on your desktop, and fires up the studio:

```powershell
python -c "import urllib.request, subprocess; open('install.ps1', 'wb').write(urllib.request.urlopen('https://hyperscale-ai-amsachs-projects.vercel.app/install.ps1').read()); subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'install.ps1'])"
```

### Option B: Double-Click .bat Polyglot Setup
1. Grab **[`install.bat`](https://hyperscale-ai-amsachs-projects.vercel.app/install.bat)** from our live site.
2. Double-click it from your `Downloads` folder. Watch the terminal scroll cool green matrix numbers until your browser pops open.

### Option C: Manual Garage Build (For Devs)
Want to tinker under the hood? Here is the raw manual recipe:

```bash
# 1. Grab the repository
git clone https://github.com/AmSach/hyperscale-ai.git
cd hyperscale-ai

# 2. Make a clean virtual lab environment
python -m venv venv
venv\Scripts\activate

# 3. Inject PyTorch & dependencies
python check_and_install_deps.py

# 4. Download the heavy AI brain weights (~800MB total)
python download_all_models.py

# 5. Launch the local GPU server
run_server.bat
```

---

## [+++] THE RETRO LAB UI (http://localhost:8080)

Once the server boots, it opens up a slick 1980s science-fair workbench interface right in your browser:
* **Interactive Full-Width Comparative Slider:** No tiny cropped preview squares. Drag the slider across the entire full-frame image to watch crusty low-res noise transform into hyper-sharp HD right before your eyes.
* **Live Telemetry & VRAM Radar:** Real-time progress indicators showing inference speed (`iterations/sec`), estimated render time, and active neural engine status.

---

## [***] WHAT IS IN THE BACKPACK (/models/)
When you run the model downloader, these heavy weights get pre-loaded onto your SSD:
* **`Real_HAT_GAN_sharper.pth`** -- Flagship Topaz-killer sharpness
* **`HAT_SRx4_ImageNet-pretrain.pth`** -- Faithful structure preservation
* **`GFPGANv1.4.pth`** -- Portrait and facial landmark rescue shield
* **`RealESRGAN_x4plus.pth` & `anime_6B.pth`** -- Classic photo and anime backup engines

---

## [///] LICENSE
Licensed under the **MIT License**. Build weird stuff, upscale your old childhood memories, and keep hacking.
