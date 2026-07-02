# ⚡ HyperScale AI Studio (V2.5)

[![NVIDIA CUDA Supported](https://img.shields.io/badge/NVIDIA-CUDA_12.1+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://hyperscale-ai-amsachs-projects.vercel.app)

**HyperScale AI Studio** is a professional, 100% locally-hosted neural image reconstruction and extreme super-resolution engine for Windows NVIDIA GPUs. Designed with an engaging **Retro 80s Science Fair / Garage Hacker** aesthetic, it combines state-of-the-art vision transformers with generative diffusion models to reconstruct up to **127+ Megapixels** of authentic photographic detail.

---

## ✨ Architectural Highlights

### 🧠 1. State-of-the-Art Flagship Engine: HAT-GAN (Hybrid Attention Transformer)
Unlike older interpolative neural networks that over-smooth images into plastic textures, HyperScale uses **`hat-sharper` (HAT-GAN)** as its flagship default engine. By leveraging self-attention across wide receptive fields, HAT synthesizes authentic micro-details—reproducing realistic skin pores, fine facial hair, DSLR film grain, and intricate fabric weaves.

### 🎨 2. Generative ControlNet Tile & SD Hybrid Mode
For extreme upscaling (8x to 16x), our engine executes a **Multi-Pass Context-Aware Synthesis pipeline**:
1. **First Pass:** Scales to 4x using HAT-GAN to establish pristine edge boundaries.
2. **Second Pass:** Injects high-frequency generative textures using Stable Diffusion ControlNet Tile while locking original geometry and color grading via custom frequency-separation blending.

### 🛡️ 3. GFPGAN v1.4 Automatic Face Restoration Shield
When processing group photos or portraits, generative models can sometimes distort facial landmarks. HyperScale automatically runs facial detection and applies dedicated **GFPGAN v1.4** eye/mouth recovery to keep faces crystalline and true to life.

### ⚡ 4. Dynamic VRAM Tiling & Zero Cloud Latency
Engineered specifically for consumer NVIDIA hardware (6GB / 8GB / 12GB+ VRAM). The backend dynamically calculates optimal tile sizes, VAE slicing, and CPU memory offloading to prevent Out-Of-Memory (OOM) crashes on massive 8K canvases. **No cloud queues, zero subscription costs, and 100% data privacy.**

---

## 🚀 Quick Setup & Installation

You can install and launch HyperScale AI Studio in seconds using any of our automated setup methods:

### Option A: Windows PowerShell One-Liner (Recommended)
Open your Windows Command Prompt or PowerShell and paste this one-liner. It will automatically check your GPU, install Python 3.12 (if needed), set up a virtual environment, pre-download all AI models, create a Desktop shortcut, and open the studio:

```powershell
python -c "import urllib.request, subprocess; open('install.ps1', 'wb').write(urllib.request.urlopen('https://hyperscale-ai-amsachs-projects.vercel.app/install.ps1').read()); subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'install.ps1'])"
```

### Option B: Self-Contained Polyglot Installer (.bat)
1. Download **[`install.bat`](https://hyperscale-ai-amsachs-projects.vercel.app/install.bat)** directly from our live website.
2. Double-click `install.bat` from your `Downloads` folder. The embedded setup engine runs 100% locally and launches the studio server on completion.

### Option C: Manual Setup for Developers
If you prefer manual control over your Python environment:

```bash
# 1. Clone the repository
git clone https://github.com/AmSach/hyperscale-ai.git
cd hyperscale-ai

# 2. Create and activate a clean virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install PyTorch (CUDA 12.1) & dependencies
python check_and_install_deps.py

# 4. Pre-download neural network weights (HAT-GAN, GFPGAN, RealESRGAN)
python download_all_models.py

# 5. Launch the local GPU Studio Server & UI
run_server.bat
```

---

## 🖥️ Local Studio Interface

Once launched, the engine runs a multi-threaded HTTP API server locally and automatically opens your browser to:
👉 **`http://localhost:8080`**

* **Interactive Full-Frame Comparison Slider:** Drag to compare high-frequency neural reconstruction against original low-res inputs in real time without cropping.
* **Batch Processing & Tile Monitor:** Real-time progress bars indicating tile processing speeds (it/s) and estimated completion times.

---

## 📁 Included Model Weights (Pre-Loaded during Setup)
All neural weights are downloaded directly into the local `/models/` directory:
* **`Real_HAT_GAN_sharper.pth`** *(Flagship Topaz-Level Sharpness)*
* **`HAT_SRx4_ImageNet-pretrain.pth`** *(Faithful Structure Restoration)*
* **`GFPGANv1.4.pth`** *(Portrait Face Shield)*
* **`RealESRGAN_x4plus.pth`** / **`RealESRGAN_x4plus_anime_6B.pth`** *(General Photo & Anime Backup)*
* **`realesr-general-x4v3.pth`** *(Fast Compact Engine)*

---

## 📜 License
Released under the **MIT License**. Created and engineered for professional generative artists and developers.
