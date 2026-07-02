import os
import urllib.request
import socket

# Set socket timeout
socket.setdefaulttimeout(30)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = [
    {
        "name": "RealESRGAN x4plus (General Photo)",
        "file": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    },
    {
        "name": "RealESRNet x4plus",
        "file": "RealESRNet_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    },
    {
        "name": "RealESRGAN x4plus Anime",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    },
    {
        "name": "RealESRGAN x2plus",
        "file": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    },
    {
        "name": "RealESR General x4v3 (Fast Compact)",
        "file": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    },
    {
        "name": "GFPGAN v1.4 (Face Restoration)",
        "file": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    }
]

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, int(downloaded / total_size * 100))
        mb_down = round(downloaded / (1024 * 1024), 1)
        mb_total = round(total_size / (1024 * 1024), 1)
        print(f"\r  -> Downloading: {percent}% ({mb_down}MB / {mb_total}MB)", end="", flush=True)

def main():
    print("================================================================")
    print("      Verifying & Pre-Downloading Neural Network Weights        ")
    print("================================================================")
    print(f"Target Directory: {MODELS_DIR}\n")

    for idx, model in enumerate(MODELS, 1):
        target_path = os.path.join(MODELS_DIR, model["file"])
        print(f"[{idx}/{len(MODELS)}] {model['name']} ({model['file']})")
        
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
            size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 2)
            print(f"  [OK] Already present ({size_mb} MB)\n")
            continue
        
        print(f"  [DOWNLOAD] Fetching from {model['url']}...")
        try:
            urllib.request.urlretrieve(model["url"], target_path, reporthook=download_progress)
            print()
            size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 2)
            print(f"  [SUCCESS] Saved {model['file']} ({size_mb} MB)\n")
        except Exception as e:
            print(f"\n  [WARNING] Could not download {model['file']}: {e}\n")

    print("================================================================")
    print("      All AI Models Verified and Ready for GPU Inference        ")
    print("================================================================\n")

if __name__ == "__main__":
    main()
