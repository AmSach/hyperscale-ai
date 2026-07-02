import sys
import subprocess
import importlib

# Map python import name to pip package name(s)
REQUIRED_LIBS = [
    ("torch", "torch torchvision --index-url https://download.pytorch.org/whl/cu121"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python"),
    ("spandrel", "spandrel"),
    ("basicsr", "basicsr"),
    ("gfpgan", "gfpgan"),
    ("realesrgan", "realesrgan"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("requests", "requests"),
    ("skimage", "scikit-image")
]

def check_and_install():
    print("=== Checking Python Dependencies ===")
    
    # Check pip
    try:
        import pip
    except ImportError:
        print("[ERROR] pip is not installed! Attempting to bootstrap pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--default-pip"])
        except Exception as e:
            print(f"[ERROR] Failed to bootstrap pip: {e}")
            sys.exit(1)

    missing_libs = []
    for import_name, pip_name in REQUIRED_LIBS:
        try:
            importlib.import_module(import_name)
            print(f"[OK] {import_name} is already installed.")
        except ImportError:
            print(f"[MISSING] {import_name} is not installed.")
            missing_libs.append((import_name, pip_name))
            
    if not missing_libs:
        print("[OK] All dependencies are satisfied!")
        return

    print(f"\nInstalling {len(missing_libs)} missing package(s)...")
    for import_name, pip_name in missing_libs:
        print(f"Installing {pip_name}...")
        
        # Build pip install list (splits on whitespace)
        cmd = [sys.executable, "-m", "pip", "install"] + pip_name.split()
        try:
            subprocess.check_call(cmd)
            print(f"[OK] Successfully installed {import_name}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install {pip_name}: {e}")
            # If basicsr fails, try the standard --no-deps workaround
            if import_name == "basicsr":
                print("[TIP] Attempting fallback basicsr installation with --no-deps...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "basicsr", "--no-deps"])
                    print("[OK] Installed basicsr with --no-deps successfully.")
                    continue
                except Exception as e2:
                    print(f"[ERROR] Fallback failed: {e2}")
            sys.exit(1)

    print("\n[SUCCESS] All dependencies checked and installed successfully!")

    # Verify PyTorch has CUDA support — a CPU-only build silently passes import checks
    print("\n=== Verifying CUDA / GPU Support ===")
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[OK] CUDA available — GPU: {gpu} ({vram:.1f} GB VRAM)")
        else:
            print("[WARNING] PyTorch installed but CUDA is NOT available.")
            print("          This usually means a CPU-only PyTorch build was installed.")
            print("          Upscaling will work but will be very slow on CPU.")
            print("          To fix: pip uninstall torch torchvision -y")
            print("          Then: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    except Exception as e:
        print(f"[WARNING] Could not verify CUDA: {e}")

if __name__ == "__main__":
    check_and_install()
