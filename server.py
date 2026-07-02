"""SuperAI Upscaler — Professional GPU Server using Real-ESRGAN Python API
   Uses CUDA directly via PyTorch for genuine super-resolution.
"""
import os
import sys
# Monkeypatch torchvision.transforms.functional_tensor for basicsr compatibility
import types
class MockModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return lambda *args, **kwargs: args[0] if args else None
sys.modules['torchvision.transforms.functional_tensor'] = MockModule('torchvision.transforms.functional_tensor')
import json
import base64
import time
import tempfile
import traceback
import io
import threading
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from PIL import Image, ImageFilter, ImageEnhance, ImageFile

# Robust loading
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

PORT = 8080
DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ======================== LAZY GLOBALS ========================================
# These are initialized on first request to avoid slow startup
_torch = None
_device = None
_upsampler_cache = {}  # model_name -> RealESRGANer instance
_spandrel_cache = {}   # model_name -> spandrel model instance
_face_enhancer = None
_upscale_lock = threading.Lock()  # Serialize GPU inference to prevent VRAM thrashing
_should_abort = False

_current_progress = {
    'active': False,
    'tile_idx': 0,
    'total_tiles': 0,
    'speed': 0.0,
    'eta': '',
    'stage': 'Idle',
    'percent': 0
}

def _init_torch():
    """Lazy-load PyTorch and detect CUDA."""
    global _torch, _device
    if _torch is not None:
        return
    
    import torch
    _torch = torch
    
    if torch.cuda.is_available():
        _device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name} - {vram:.1f} GB VRAM")
        print(f"[GPU] CUDA {torch.version.cuda}")
    else:
        _device = torch.device('cpu')
        print("[WARNING] CUDA not available - running on CPU (slow)")

# ======================== MODEL DEFINITIONS ===================================
# Each model has: network architecture, scale factor, model URL, and description

MODEL_REGISTRY = {
    'realesrgan-x4plus': {
        'arch': 'RRDBNet',
        'num_block': 23,
        'scale': 4,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'desc': 'General purpose photo upscaler (best for most photos)',
        'engine': 'realesrgan',
    },
    'realesrnet-x4plus': {
        'arch': 'RRDBNet',
        'num_block': 23,
        'scale': 4,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
        'desc': 'Sharper, less hallucination variant',
        'engine': 'realesrgan',
    },
    'realesrgan-x4plus-anime': {
        'arch': 'RRDBNet',
        'num_block': 6,
        'scale': 4,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        'desc': 'Optimized for anime/illustration/line art',
        'engine': 'realesrgan',
    },
    'realesrgan-x2plus': {
        'arch': 'RRDBNet',
        'num_block': 23,
        'scale': 2,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'desc': '2x upscale — faster, less artifacts',
        'engine': 'realesrgan',
    },
    'realesr-animevideov3': {
        'arch': 'VGGStyleDiscriminator',
        'num_block': None,
        'scale': 4,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
        'desc': 'Fast anime video model (compact network)',
        'engine': 'realesrgan',
    },
    'realesr-general-x4v3': {
        'arch': 'SRVGGNetCompact',
        'num_block': None,
        'scale': 4,
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        'desc': 'Fast general purpose (compact network)',
        'engine': 'realesrgan',
    },
    # --- HAT Models (state-of-the-art, much better quality) ---
    'hat-sharper': {
        'arch': 'HAT',
        'scale': 4,
        'local_file': 'Real_HAT_GAN_sharper.pth',
        'desc': 'HAT GAN Sharper — best perceptual quality for real-world photos',
        'engine': 'spandrel',
    },
    'hat-imagenet': {
        'arch': 'HAT',
        'scale': 4,
        'local_file': 'HAT_SRx4_ImageNet-pretrain.pth',
        'desc': 'HAT ImageNet - faithful/sharp restoration, minimal hallucination',
        'engine': 'spandrel',
    },
    'cloud-supir': {
        'arch': 'SUPIR',
        'scale': 4,
        'desc': 'Cloud SUPIR - Premium diffusion-based upscaling (requires Replicate key, ~15s)',
        'engine': 'replicate',
    },
    'hat-realesrgan-blend': {
        'arch': 'Blend',
        'scale': 4,
        'desc': 'Model Ensemble Blend (HAT + RealESRGAN) - Ultimate Topaz-Level Sharpness & Detail',
        'engine': 'blend',
    },
}

DEFAULT_MODEL = 'hat-sharper'

def _get_model_path(model_name):
    """Get local path for a model, downloading if needed."""
    info = MODEL_REGISTRY.get(model_name)
    if not info:
        info = MODEL_REGISTRY[DEFAULT_MODEL]
        model_name = DEFAULT_MODEL
    
    filename = info['url'].split('/')[-1]
    path = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(path):
        print(f"[DOWNLOAD] Downloading model from {info['url']} to {path}...")
        import urllib.request
        import socket
        # Set socket timeout to 15 seconds to prevent indefinite hangs
        socket.setdefaulttimeout(15)
        try:
            urllib.request.urlretrieve(info['url'], path)
            print(f"[OK] Model downloaded: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to download model: {e}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
            raise RuntimeError(
                f"Model file not found and auto-download failed.\n"
                f"Please download the model manually from:\n{info['url']}\n"
                f"and save it to:\n{path}"
            )
    
    return path, info


# Maximum output megapixels to prevent OOM crashes
# Dynamic caps: GPU allows high resolutions, CPU is more restricted
MAX_GPU_MEGAPIXELS = 150  # ~12K portrait (e.g. 9000x16000)
MAX_CPU_MEGAPIXELS = 40   # ~6K (e.g. 5000x8000)

def _build_upsampler(model_name, scale, tile_size=400, device_name='cuda', input_megapixels=0):
    """Build or retrieve a cached RealESRGANer instance."""
    global _upsampler_cache
    
    _init_torch()
    
    # Handle CPU fallback
    if device_name == 'cpu' or not _torch.cuda.is_available():
        device_name = 'cpu'
        dev = _torch.device('cpu')
        half = False
        # Use smaller tiles for large images on CPU to reduce peak memory
        if input_megapixels > 4:
            tile_size = 128
        else:
            tile_size = 192
    else:
        device_name = 'cuda'
        dev = _torch.device('cuda')
        half = True
        # Determine tile size based on VRAM
        vram_gb = _torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 4:
            tile_size = 200
        elif vram_gb < 6:
            tile_size = 320
        elif vram_gb < 8:
            tile_size = 400
        else:
            tile_size = 512

    cache_key = f"{model_name}_{scale}_{device_name}"
    if cache_key in _upsampler_cache:
        return _upsampler_cache[cache_key]
    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    
    model_path, info = _get_model_path(model_name)
    native_scale = info['scale']
    
    # Build the appropriate network architecture
    if model_name in ('realesr-general-x4v3', 'realesr-animevideov3'):
        # Compact architecture (SRVGGNetCompact)
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        if model_name == 'realesr-animevideov3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        else:
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    elif model_name == 'realesrgan-x4plus-anime':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    elif model_name == 'realesrgan-x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        # Standard RRDBNet (x4plus, x4net)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    upsampler = RealESRGANer(
        scale=native_scale,
        model_path=model_path,
        model=model,
        tile=tile_size,
        tile_pad=32,
        pre_pad=0,
        half=half,
        device=dev
    )
    
    _upsampler_cache[cache_key] = upsampler
    print(f"[OK] Model loaded: {model_name} on {device_name} (tile={tile_size}, fp16={half})")
    return upsampler


def _get_spandrel_model(model_name, device_name='cuda'):
    """Load a model via spandrel (supports HAT, SwinIR, DAT, etc.)"""
    global _spandrel_cache
    
    cache_key = f"{model_name}_{device_name}"
    if cache_key in _spandrel_cache:
        return _spandrel_cache[cache_key]
    
    _init_torch()
    from spandrel import ModelLoader
    
    info = MODEL_REGISTRY[model_name]
    model_path = os.path.join(MODELS_DIR, info['local_file'])
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}. Please download it first.")
    
    print(f"[SPANDREL] Loading {model_name} from {model_path}...")
    model = ModelLoader().load_from_file(model_path)
    
    if device_name == 'cuda' and _torch.cuda.is_available():
        model = model.cuda().eval()
        model.is_fp16 = False
        model.is_bf16 = False
        
        # Try casting to bfloat16 first (preserves dynamic range, prevents NaN overflows)
        if _torch.cuda.is_bf16_supported():
            try:
                model = model.bfloat16()
                model.is_bf16 = True
                print("[SPANDREL] Model successfully cast to bfloat16.")
            except Exception as e:
                print(f"[SPANDREL] Model does not support bfloat16 casting: {e}")
                
        # Fallback to fp16 if bf16 is not supported or failed
        if not model.is_bf16 and hasattr(model, 'half'):
            try:
                model = model.half()
                model.is_fp16 = True
                print("[SPANDREL] Model successfully cast to fp16.")
            except Exception as e:
                print(f"[SPANDREL] Model does not support fp16 casting: {e}. Falling back to fp32.")
    else:
        model = model.eval()
        model.is_fp16 = False
    _spandrel_cache[cache_key] = model
    print(f"[OK] Spandrel model loaded: {model_name} (arch={info['arch']}, scale={info['scale']}, fp16={getattr(model, 'is_fp16', False)})")
    return model


def _get_adaptive_spandrel_tile_params(device_name):
    """Dynamically determine tile size and pad based on actual free VRAM."""
    if device_name != 'cuda' or not _torch.cuda.is_available():
        return 256, 32  # CPU can handle larger tiles as it uses system RAM
        
    free_mem, total_mem = _torch.cuda.mem_get_info()
    free_gb = free_mem / 1024**3
    vram_gb = total_mem / 1024**3
    is_bf16 = _torch.cuda.is_bf16_supported()
    
    # Adjust tile size dynamically if free VRAM is low
    if free_gb < 2.0:
        sp_tile = 128
        sp_pad = 16
        print(f"[RUN] Low free VRAM ({free_gb:.2f} GB free). Reducing tile size to {sp_tile} to avoid paging.")
    else:
        if is_bf16:
            sp_tile = 192 if vram_gb < 8 else 256
            sp_pad = 16
        else:
            sp_tile = 128 if vram_gb < 8 else (192 if vram_gb < 12 else 256)
            sp_pad = 32
            
    return sp_tile, sp_pad


def _spandrel_tiled_upscale(model, img_np, scale, tile_size=256, tile_pad=32, device_name='cuda'):
    """
    Tiled inference for spandrel models.
    img_np: RGB numpy array (H, W, 3) uint8
    Returns: RGB numpy array (H*scale, W*scale, 3) uint8
    """
    import torch
    h, w, c = img_np.shape
    out_h, out_w = h * scale, w * scale
    
    # Pre-allocate output
    output = np.zeros((out_h, out_w, c), dtype=np.uint8)
    
    # Calculate tiles
    tiles_x = max(1, (w + tile_size - 1) // tile_size)
    tiles_y = max(1, (h + tile_size - 1) // tile_size)
    total_tiles = tiles_x * tiles_y
    
    # Determine adaptive batch size based on device & VRAM
    if device_name == 'cpu':
        batch_size = 1
    else:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        is_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        
        # Check actual available free VRAM dynamically
        free_mem, total_mem = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        free_gb = free_mem / 1024**3
        
        if free_gb < 2.5:
            batch_size = 1
            print(f"[SPANDREL] Tight VRAM detected ({free_gb:.2f} GB free). Forcing batch_size=1 to prevent WDDM paging.")
        else:
            if vram_gb < 5.5:
                batch_size = 1
            elif vram_gb < 7.5:
                batch_size = 1 if not is_bf16 else 2
            else:
                batch_size = 2 if not is_bf16 else 4
            
    print(f"[SPANDREL] Processing {total_tiles} tiles ({tiles_y}x{tiles_x}, tile_size={tile_size}, adaptive batch_size={batch_size})...")
    
    use_fp16 = (device_name == 'cuda' and getattr(model, 'is_fp16', False))
    use_bf16 = (device_name == 'cuda' and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    
    import time
    start_t = time.time()
    
    # Target size for input tiles (including padding)
    target_h = tile_size + 2 * tile_pad
    target_w = tile_size + 2 * tile_pad
    
    # Prepare all tiles coordinates
    tiles_info = []
    for iy in range(tiles_y):
        for ix in range(tiles_x):
            x_start = ix * tile_size
            y_start = iy * tile_size
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)
            
            x_start_pad = max(0, x_start - tile_pad)
            y_start_pad = max(0, y_start - tile_pad)
            x_end_pad = min(w, x_end + tile_pad)
            y_end_pad = min(h, y_end + tile_pad)
            
            tiles_info.append({
                'x_start': x_start, 'y_start': y_start, 'x_end': x_end, 'y_end': y_end,
                'x_start_pad': x_start_pad, 'y_start_pad': y_start_pad, 'x_end_pad': x_end_pad, 'y_end_pad': y_end_pad
            })
            
    # Process in batches
    for i in range(0, len(tiles_info), batch_size):
        global _should_abort
        if _should_abort:
            print("[SPANDREL] Aborting tiled upscaling due to user cancel request.")
            raise RuntimeError("Upscaling process was cancelled by the user.")
            
        batch_items = tiles_info[i:i+batch_size]
        batch_tensors = []
        
        for item in batch_items:
            # Extract tile
            tile = img_np[item['y_start_pad']:item['y_end_pad'], item['x_start_pad']:item['x_end_pad'], :]
            tile_h, tile_w, _ = tile.shape
            
            # Pad tile if it is smaller than target_h x target_w (reflection padding at bottom/right)
            h_pad = target_h - tile_h
            w_pad = target_w - tile_w
            if h_pad > 0 or w_pad > 0:
                tile = np.pad(tile, ((0, h_pad), (0, w_pad), (0, 0)), mode='reflect')
                
            # Convert to tensor: (H,W,C) -> (C,H,W) float32 [0,1]
            t = torch.from_numpy(tile.astype(np.float32) / 255.0).permute(2, 0, 1)
            if device_name == 'cuda':
                t = t.cuda()
            if getattr(model, 'is_bf16', False):
                t = t.to(torch.bfloat16)
            elif use_fp16:
                t = t.half()
            batch_tensors.append(t)
            
        # Stack into a batch tensor (B, C, H, W)
        batch_tensor = torch.stack(batch_tensors)
        
        # Inference
        with torch.no_grad():
            if use_bf16:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out_batch = model(batch_tensor)
            else:
                out_batch = model(batch_tensor)
                
        # Convert to uint8 directly on GPU/CPU to save memory and avoid float32 allocation
        out_batch = (out_batch.float() * 255.0).clamp(0, 255).to(torch.uint8)
        out_batch = out_batch.cpu().permute(0, 2, 3, 1).numpy()
        
        for idx, item in enumerate(batch_items):
            out_tile = out_batch[idx] # (H_out, W_out, C)
            
            # Crop back if it was padded
            tile_h = item['y_end_pad'] - item['y_start_pad']
            tile_w = item['x_end_pad'] - item['x_start_pad']
            out_tile = out_tile[0:tile_h*scale, 0:tile_w*scale, :]
            
            # Coordinates in output space (remove padding)
            out_x_start = item['x_start'] * scale
            out_y_start = item['y_start'] * scale
            out_x_end = item['x_end'] * scale
            out_y_end = item['y_end'] * scale
            
            pad_left = (item['x_start'] - item['x_start_pad']) * scale
            pad_top = (item['y_start'] - item['y_start_pad']) * scale
            tile_out_h = (item['y_end'] - item['y_start']) * scale
            tile_out_w = (item['x_end'] - item['x_start']) * scale
            
            output[out_y_start:out_y_end, out_x_start:out_x_end, :] = \
                out_tile[pad_top:pad_top+tile_out_h, pad_left:pad_left+tile_out_w, :]
                
        # Real-time speed and ETA calculation
        tile_idx = min(i + batch_size, total_tiles)
        elapsed_total = time.time() - start_t
        avg_tile_time = elapsed_total / tile_idx
        remaining_tiles = total_tiles - tile_idx
        eta_seconds = remaining_tiles * avg_tile_time
        
        # Format ETA
        if eta_seconds > 60:
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = f"{eta_seconds:.1f}s"
            
        # Update global progress
        global _current_progress
        _current_progress.update({
            'active': True,
            'tile_idx': tile_idx,
            'total_tiles': total_tiles,
            'speed': round(avg_tile_time, 2),
            'eta': eta_str,
            'percent': int((tile_idx / total_tiles) * 100),
            'stage': f"Processing tiles ({tile_idx}/{total_tiles})"
        })
            
        # Print status every batch (or on first/last tile)
        if i == 0 or tile_idx % (batch_size * 5) == 0 or tile_idx == total_tiles:
            print(f"\tTile {tile_idx}/{total_tiles} | Speed: {avg_tile_time:.2f}s/tile ({1.0/avg_tile_time:.2f} tiles/s) | ETA: {eta_str}", flush=True)
            
    return output


def _replicate_upscale(img_bytes, scale, api_key):
    """
    Call Replicate Cloud API to run SUPIR upscaling.
    Natively runs on A100 GPU and completes in 10-25 seconds.
    """
    import requests
    import time
    
    print(f"[REPLICATE] Preparing request for Cloud SUPIR (scale={scale}x)...")
    
    # Convert image to data URI
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    data_uri = f"data:image/png;base64,{img_b64}"
    
    # Replicate API endpoint
    url = "https://api.replicate.com/v1/predictions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Scale parameter for cjwbw/supir (upscale input parameter must be an integer)
    upscale_val = int(max(1, min(8, scale)))
    
    payload = {
        "version": "1302b550b4f7681da87ed0e405016d443fe1fafd64dabce6673401855a5039b5",
        "input": {
            "image": data_uri,
            "upscale": upscale_val,
            "edm_steps": 30, # Lower steps for speed (30 is very fast and high quality)
            "use_llava": False, # LLaVA captioning is disabled for speed
            "sampler": "Euler",
            "cfg_scale": 4,
            "control_scale": 1.0,
            "min_resolution": 1024
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 201:
            raise RuntimeError(f"Replicate API error: {response.status_code} - {response.text}")
            
        prediction = response.json()
        prediction_id = prediction["id"]
        get_url = prediction["urls"]["get"]
        print(f"[REPLICATE] Prediction created! ID: {prediction_id}. Polling status...")
        
        # Poll for completion
        start_time = time.time()
        max_wait = 300 # 5 minutes max
        
        while time.time() - start_time < max_wait:
            poll_resp = requests.get(get_url, headers=headers, timeout=30)
            if poll_resp.status_code != 200:
                raise RuntimeError(f"Error polling Replicate prediction: {poll_resp.status_code} - {poll_resp.text}")
                
            pred_data = poll_resp.json()
            status = pred_data.get("status")
            print(f"[REPLICATE] Status: {status} ({int(time.time() - start_time)}s)")
            
            if status == "succeeded":
                output_url = pred_data.get("output")
                if not output_url:
                    raise RuntimeError("Replicate succeeded but returned no output URL.")
                
                # Download output image
                print(f"[REPLICATE] Downloading result from: {output_url}")
                img_resp = requests.get(output_url, timeout=60)
                if img_resp.status_code != 200:
                    raise RuntimeError(f"Failed to download output image: {img_resp.status_code}")
                    
                # Load as RGB numpy array
                out_img = Image.open(io.BytesIO(img_resp.content)).convert('RGB')
                return np.array(out_img)
                
            elif status in ("failed", "canceled"):
                error_info = pred_data.get("error", "Unknown error")
                raise RuntimeError(f"Replicate prediction {status}: {error_info}")
                
            time.sleep(2)
            
        raise TimeoutError("Replicate prediction timed out.")
        
    except Exception as e:
        print(f"[REPLICATE] Error: {e}")
        traceback.print_exc()
        raise e


def free_gpu_memory():
    """Unload heavy models to free up VRAM for active steps."""
    global _sd_refine_pipeline, _spandrel_cache, _upsampler_cache, _face_enhancer, _face_enhancer_cache
    import gc
    
    # Unload Stable Diffusion pipeline
    if _sd_refine_pipeline is not None:
        print("[GPU] Unloading Stable Diffusion pipeline to free VRAM...")
        _sd_refine_pipeline = None
        
    # Clear spandrel models cache
    _spandrel_cache.clear()
    _upsampler_cache.clear()
    _face_enhancer_cache.clear()
    _face_enhancer = None
    
    # Run garbage collection and empty CUDA cache
    gc.collect()
    if _torch is not None and _torch.cuda.is_available():
        _torch.cuda.empty_cache()

_sd_refine_pipeline = None

def _get_sd_refine_pipeline(device_name='cuda'):
    global _sd_refine_pipeline
    if _sd_refine_pipeline is not None:
        return _sd_refine_pipeline
        
    import torch
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
    
    print("[SD-REFINE] Loading SD 1.5 + ControlNet Tile from cache/disk...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1e_sd15_tile", 
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    # Configure high-quality DPM++ 2M Karras scheduler for maximum clarity and detail
    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, 
        use_karras_sigmas=True
    )
    
    # Memory Optimizations for 6GB VRAM
    if device_name == 'cuda' and torch.cuda.is_available():
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        try:
            pipe.enable_model_cpu_offload()
            print("[SD-REFINE] Model CPU offload enabled successfully.")
        except Exception as e:
            print(f"[WARNING] enable_model_cpu_offload failed: {e}. Falling back to pipe.to('cuda').")
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
        
    _sd_refine_pipeline = pipe
    return pipe
def apply_sd_controlnet_refine(img_np_bgr, prompt, negative_prompt, denoising_strength=0.20, device_name='cuda', mode='hybrid', tile_size_override=0, controlnet_scale=0.80, detail_boost=1.30, detail_threshold=50.0, sd_refine_stage='post'):
    """
    Refine upscaled image using SD 1.5 + ControlNet Tile in a tiled manner.
    This prevents VRAM OOM, runs at native resolution, and avoids waxy/painterly downscaling artifacts.
    Supports two modes:
      - 'hybrid': preserves original colors/shapes 100% via native frequency separation.
      - 'full': directly blends SD output, allowing structural refinement.
    """
    global _should_abort
    import torch
    from PIL import Image
    import numpy as np
    import cv2
    
    pipe = _get_sd_refine_pipeline(device_name=device_name)
    
    # Calculate inference steps to guarantee at least 12 actual denoising steps
    # (since actual steps = int(num_inference_steps * denoising_strength))
    desired_steps = 12
    steps_to_run = max(20, int(desired_steps / max(0.01, denoising_strength)))
    print(f"[SD-REFINE] Denoising strength={denoising_strength:.2f} -> running {desired_steps} actual steps out of {steps_to_run} total scheduled steps.")
    
    H, W, C = img_np_bgr.shape

    # 1. Run face detection to protect faces from Stable Diffusion warping/distortion
    face_boxes = []
    try:
        dev = torch.device(device_name if (device_name == 'cuda' and torch.cuda.is_available()) else 'cpu')
        enhancer = _get_face_enhancer(dev)
        if enhancer is not None:
            enhancer.face_helper.clean_all()
            enhancer.face_helper.read_image(img_np_bgr)
            # Detect faces with eye distance threshold of 5 pixels (skip extremely small noise)
            enhancer.face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
            if len(enhancer.face_helper.det_faces) > 0:
                print(f"[FACE-PROTECT] Detected {len(enhancer.face_helper.det_faces)} face(s) to protect from SD warping.")
                for box in enhancer.face_helper.det_faces:
                    face_boxes.append(box[:4])
    except Exception as e:
        print(f"[FACE-PROTECT] Face detection failed or skipped: {e}")

    # Build face mask with smooth feathered edges
    face_mask = np.zeros((H, W, 1), dtype=np.float32)
    if len(face_boxes) > 0:
        for box in face_boxes:
            x1, y1, x2, y2 = [int(val) for val in box]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            # Pad the bounding box slightly (30%) to cover the head/hair
            fw = x2 - x1
            fh = y2 - y1
            pad_x = int(fw * 0.3)
            pad_y = int(fh * 0.3)
            
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(W, x2 + pad_x)
            py2 = min(H, y2 + pad_y)
            
            face_mask[py1:py2, px1:px2] = 1.0
            
        # Smooth the mask transition using Gaussian blur
        blur_k = int(min(H, W) * 0.02) | 1
        blur_k = max(15, blur_k)
        face_mask = cv2.GaussianBlur(face_mask, (blur_k, blur_k), 0)
        face_mask = np.expand_dims(face_mask, axis=-1)
    
    # Determine tile size based on GPU memory capacity
    if tile_size_override > 0:
        tile_size = tile_size_override
    elif device_name == 'cuda' and torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if sd_refine_stage == 'post':
            if total_vram < 5.5:
                tile_size = 512
            elif total_vram < 7.5:
                tile_size = 768  # 768x768 is very safe on 6GB VRAM with model cpu offload
            else:
                tile_size = 1024
        else:
            if total_vram < 7.5:
                tile_size = 512
            else:
                tile_size = 768
    else:
        tile_size = 512
        
    # Adjust tile_size if the image is smaller in one dimension
    tile_size = min(tile_size, H, W)
    # Ensure tile_size is a multiple of 8
    tile_size = max(64, (tile_size // 8) * 8)
    
    overlap = min(64, tile_size // 4)
    stride = tile_size - overlap
    
    # Helper to generate tile blending weight mask in 1D
    def get_tile_mask_1d(starts, tile_idx, t_size, ov):
        mask = np.ones(t_size, dtype=np.float32)
        s_i = starts[tile_idx]
        
        # Left boundary transition (with tile tile_idx - 1)
        if tile_idx > 0:
            s_prev = starts[tile_idx - 1]
            overlap_start = s_i
            overlap_end = s_prev + t_size
            mid = (overlap_start + overlap_end) / 2.0
            w = min(ov, overlap_end - overlap_start)
            w = max(1.0, w)
            
            t_start = (mid - w/2.0) - s_i
            t_end = (mid + w/2.0) - s_i
            
            coords = np.arange(t_size, dtype=np.float32)
            mask = np.where(coords < t_start, 0.0, mask)
            ramp_mask = (coords >= t_start) & (coords <= t_end)
            t = (coords - t_start) / w
            mask = np.where(ramp_mask, 0.5 - 0.5 * np.cos(np.pi * t), mask)
            
        # Right boundary transition (with tile tile_idx + 1)
        if tile_idx < len(starts) - 1:
            s_next = starts[tile_idx + 1]
            overlap_start = s_next
            overlap_end = s_i + t_size
            mid = (overlap_start + overlap_end) / 2.0
            w = min(ov, overlap_end - overlap_start)
            w = max(1.0, w)
            
            t_start = (mid - w/2.0) - s_i
            t_end = (mid + w/2.0) - s_i
            
            coords = np.arange(t_size, dtype=np.float32)
            mask = np.where(coords > t_end, 0.0, mask)
            ramp_mask = (coords >= t_start) & (coords <= t_end)
            t = (coords - t_start) / w
            mask = np.where(ramp_mask, 0.5 + 0.5 * np.cos(np.pi * t), mask)
            
        return mask

    # If the image is smaller than or equal to the tile size, we process it in one single pass
    if H <= tile_size and W <= tile_size:
        print(f"[SD-REFINE] Image size {W}x{H} is smaller than tile size {tile_size}. Processing as a single tile...")
        init_image = Image.fromarray(img_np_bgr[:, :, ::-1])
        control_image = init_image.copy()
        
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                control_image=control_image,
                strength=denoising_strength,
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=steps_to_run,
                guidance_scale=7.5
            ).images[0]
            
        sd_out_np = np.array(output)[:, :, ::-1].copy().astype(np.float32)
        
        # Calculate dynamic detail retention based on tile size
        # This prevents 8x upscales from washing out all the macroscopic texture
        dynamic_sigma_hybrid = max(2.0, tile_size / 64.0)
        dynamic_sigma_full = max(1.0, tile_size / 128.0)
        
        if mode == 'hybrid':
            # Frequency separation at native size
            sd_diff = sd_out_np - img_np_bgr.astype(np.float32)
            # High-pass filter the difference map
            sd_diff_blur = cv2.GaussianBlur(sd_diff, (0, 0), sigmaX=dynamic_sigma_hybrid)
            sd_diff_highpass = sd_diff - sd_diff_blur
            
            # Protect sharp structural edges of the GAN input from VAE-induced blurring
            gray_in = cv2.cvtColor(img_np_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray_in, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_in, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            edge_mask = np.clip(grad_mag / 35.0, 0.0, 1.0)
            edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)
            edge_mask = np.expand_dims(edge_mask, axis=-1)
            
            # Add detail-boosted high-pass detail back to GAN, masked by (1 - edge_mask)
            final_img = img_np_bgr.astype(np.float32) + sd_diff_highpass * (detail_boost * (1.0 - edge_mask))
            final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        else:
            if detail_boost > 1.0:
                # Apply high-frequency detail boost to SD output
                sd_blur = cv2.GaussianBlur(sd_out_np, (0, 0), sigmaX=dynamic_sigma_full)
                sd_highpass = sd_out_np - sd_blur
                final_img = sd_out_np + sd_highpass * (detail_boost - 1.0)
                final_img = np.clip(final_img, 0, 255).astype(np.uint8)
            else:
                final_img = np.clip(sd_out_np, 0, 255).astype(np.uint8)
                
        # Protect faces from SD distortion by blending original back over detected face regions
        if len(face_boxes) > 0:
            final_img = (img_np_bgr.astype(np.float32) * face_mask + final_img.astype(np.float32) * (1.0 - face_mask)).astype(np.uint8)
            
        return final_img
            
    # Calculate tile coordinates
    y_starts = []
    y = 0
    while y + tile_size < H:
        y_starts.append(y)
        y += stride
    y_starts.append(H - tile_size)
    y_starts = sorted(list(set(y_starts)))
    
    x_starts = []
    x = 0
    while x + tile_size < W:
        x_starts.append(x)
        x += stride
    x_starts.append(W - tile_size)
    x_starts = sorted(list(set(x_starts)))
    
    total_tiles = len(y_starts) * len(x_starts)
    
    # Pre-calculate Sobel magnitude on a downscaled representation for speed and memory safety
    print(f"[SD-REFINE] Tiled refinement: image={W}x{H}, tile_size={tile_size}, overlap={overlap}, mode={mode}...")
    print("[SD-REFINE] Pre-calculating edge magnitude map to identify detailed vs flat regions...")
    scale_factor = 1.0
    max_dim = max(H, W)
    if max_dim > 2048:
        scale_factor = 2048.0 / max_dim
        small_gray = cv2.resize(cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    else:
        small_gray = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2GRAY)
        
    grad_x = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_small = np.sqrt(grad_x**2 + grad_y**2)
    
    # Prepare all tiles coordinates and check flatness
    tiles_info = []
    flat_count = 0
    for iy, y1 in enumerate(y_starts):
        for ix, x1 in enumerate(x_starts):
            y2 = y1 + tile_size
            x2 = x1 + tile_size
            
            # Check flatness using the small edge magnitude map
            sy1 = int(y1 * scale_factor)
            sy2 = int(y2 * scale_factor)
            sx1 = int(x1 * scale_factor)
            sx2 = int(x2 * scale_factor)
            
            # Ensure valid dimensions in downscaled coordinate space
            sy2 = max(sy2, sy1 + 1)
            sx2 = max(sx2, sx1 + 1)
            
            tile_mag = grad_mag_small[sy1:sy2, sx1:sx2]
            
            # Determine if tile has enough high-frequency detail
            is_flat = False
            if tile_mag.size > 0:
                pct95 = np.percentile(tile_mag, 95)
                if pct95 < detail_threshold:
                    is_flat = True
            
            if is_flat:
                flat_count += 1
                
            tiles_info.append({
                'iy': iy, 'ix': ix,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'flat': is_flat
            })
            
    print(f"[SD-REFINE] Total tiles: {total_tiles} | Flat tiles (skipped): {flat_count} | Detailed tiles (processed): {total_tiles - flat_count}")

    # Accumulators for blending
    accum_image = np.zeros((H, W, 3), dtype=np.float32)
    accum_weight = np.zeros((H, W, 1), dtype=np.float32)
    
    # Blending flat tiles directly (no SD inference needed)
    for item in tiles_info:
        if item['flat']:
            tile_in = img_np_bgr[item['y1']:item['y2'], item['x1']:item['x2']]
            tile_processed = tile_in.astype(np.float32)
            
            mask_y = get_tile_mask_1d(y_starts, item['iy'], tile_size, overlap)
            mask_x = get_tile_mask_1d(x_starts, item['ix'], tile_size, overlap)
            mask = np.outer(mask_y, mask_x)
            mask = np.expand_dims(mask, axis=-1)
            
            accum_image[item['y1']:item['y2'], item['x1']:item['x2']] += tile_processed * mask
            accum_weight[item['y1']:item['y2'], item['x1']:item['x2']] += mask
            
    detailed_tiles = [item for item in tiles_info if not item['flat']]
    total_detailed = len(detailed_tiles)
    
    import time
    start_sd_t = time.time()
    
    tile_count = 0
    # Determine batch size dynamically based on VRAM capacity
    import torch
    if device_name == 'cpu':
        batch_size = 1
    else:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        if vram_gb < 5.5:
            batch_size = 1
        elif vram_gb < 7.5:
            batch_size = 2 # Extremely safe for 6GB VRAM
        else:
            batch_size = 4 # High speed for 8GB+ VRAM

    for i in range(0, total_detailed, batch_size):
        if _should_abort:
            print("[SD-REFINE] Aborting tiled SD refinement due to user cancel request.")
            raise RuntimeError("Upscaling process was cancelled by the user.")
            
        batch_items = detailed_tiles[i:i+batch_size]
        
        # Crop tiles and prepare batch lists
        init_images = []
        control_images = []
        for item in batch_items:
            tile_in = img_np_bgr[item['y1']:item['y2'], item['x1']:item['x2']]
            init_images.append(Image.fromarray(tile_in[:, :, ::-1]))
            control_images.append(Image.fromarray(tile_in[:, :, ::-1]))
            
        # Run SD batch inference
        with torch.inference_mode():
            outputs = pipe(
                prompt=[prompt] * len(batch_items),
                negative_prompt=[negative_prompt] * len(batch_items),
                image=init_images,
                control_image=control_images,
                strength=denoising_strength,
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=steps_to_run,
                guidance_scale=7.5
            ).images
            
        # Calculate dynamic detail retention based on tile size
        dynamic_sigma_hybrid = max(2.0, tile_size / 64.0)
        dynamic_sigma_full = max(1.0, tile_size / 128.0)

        for idx, item in enumerate(batch_items):
            tile_in = img_np_bgr[item['y1']:item['y2'], item['x1']:item['x2']]
            tile_out_np = np.array(outputs[idx])[:, :, ::-1].astype(np.float32)
            
            if mode == 'hybrid':
                sd_diff = tile_out_np - tile_in.astype(np.float32)
                sd_diff_blur = cv2.GaussianBlur(sd_diff, (0, 0), sigmaX=dynamic_sigma_hybrid)
                sd_diff_highpass = sd_diff - sd_diff_blur
                
                # Protect sharp structural edges of the GAN input from VAE-induced blurring
                gray_in = cv2.cvtColor(tile_in.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(gray_in, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_in, cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                edge_mask = np.clip(grad_mag / 150.0, 0.0, 1.0)
                edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)
                edge_mask = np.expand_dims(edge_mask, axis=-1)
                
                # Apply detail boost only in non-edge regions to inject textures without blurring details
                tile_processed = tile_in.astype(np.float32) + sd_diff_highpass * (detail_boost * (1.0 - edge_mask))
            else:
                if detail_boost > 1.0:
                    # Apply high-frequency detail boost to SD output directly
                    sd_blur = cv2.GaussianBlur(tile_out_np, (0, 0), sigmaX=dynamic_sigma_full)
                    sd_highpass = tile_out_np - sd_blur
                    tile_processed = tile_out_np + sd_highpass * (detail_boost - 1.0)
                else:
                    tile_processed = tile_out_np
            
            # Blending mask
            mask_y = get_tile_mask_1d(y_starts, item['iy'], tile_size, overlap)
            mask_x = get_tile_mask_1d(x_starts, item['ix'], tile_size, overlap)
            mask = np.outer(mask_y, mask_x)
            mask = np.expand_dims(mask, axis=-1)
            
            accum_image[item['y1']:item['y2'], item['x1']:item['x2']] += tile_processed * mask
            accum_weight[item['y1']:item['y2'], item['x1']:item['x2']] += mask
            
        tile_count += len(batch_items)
        # Update global progress status
        elapsed_total = time.time() - start_sd_t
        avg_tile_time = elapsed_total / tile_count
        remaining_tiles = total_detailed - tile_count
        eta_seconds = remaining_tiles * avg_tile_time
        if eta_seconds > 60:
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = f"{eta_seconds:.1f}s"
            
        global _current_progress
        _current_progress.update({
            'active': True,
            'tile_idx': tile_count,
            'total_tiles': total_detailed,
            'speed': round(avg_tile_time, 2),
            'eta': eta_str,
            'percent': 80 + int((tile_count / max(1, total_detailed)) * 10),
            'stage': f"SD Texture Refinement ({tile_count}/{total_detailed})"
        })
            
    # Normalize blending
    final_image = accum_image / (accum_weight + 1e-8)
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    
    # Protect faces from SD distortion by blending original back over detected face regions
    if len(face_boxes) > 0:
        final_image = (img_np_bgr.astype(np.float32) * face_mask + final_image.astype(np.float32) * (1.0 - face_mask)).astype(np.uint8)
        
    return final_image.copy()


def apply_color_matching(lr_img_np, hr_img_np):
    """
    Adjust upscaled HR image to match the color and contrast of the original LR image
    using mean and standard deviation alignment in LAB color space.
    """
    import cv2
    import numpy as np
    
    # Compute stats on the low-resolution shape to save gigabytes of system memory
    lr_h, lr_w, _ = lr_img_np.shape
    hr_small = cv2.resize(hr_img_np, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    
    hr_small_lab = cv2.cvtColor(hr_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    lr_small_lab = cv2.cvtColor(lr_img_np, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Compute mean and std on low-res representations
    mean_hr, std_hr = cv2.meanStdDev(hr_small_lab)
    mean_lr, std_lr = cv2.meanStdDev(lr_small_lab)
    
    # Reshape and cast to float32 to prevent automatic numpy upcasting to float64
    mean_hr = mean_hr.reshape(1, 1, 3).astype(np.float32)
    std_hr = std_hr.reshape(1, 1, 3).astype(np.float32)
    mean_lr = mean_lr.reshape(1, 1, 3).astype(np.float32)
    std_lr = std_lr.reshape(1, 1, 3).astype(np.float32)
    
    std_ratio = (std_lr / (std_hr + np.float32(1e-6))).astype(np.float32)
    std_ratio = np.clip(std_ratio, np.float32(0.6), np.float32(1.8))
    
    # Apply color correction to the full high-resolution image in float32
    hr_lab = cv2.cvtColor(hr_img_np, cv2.COLOR_BGR2LAB).astype(np.float32)
    corrected_lab = (hr_lab - mean_hr) * std_ratio + mean_lr
    corrected_lab = np.clip(corrected_lab, 0.0, 255.0).astype(np.uint8)
    
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)


_face_enhancer_cache = {}

def _get_face_enhancer(device):
    """Build GFPGAN face enhancer (lazy loaded per device)."""
    global _face_enhancer_cache
    device_name = str(device)
    if device_name in _face_enhancer_cache:
        return _face_enhancer_cache[device_name]
    
    try:
        from gfpgan import GFPGANer
        
        face_model_path = os.path.join(MODELS_DIR, 'GFPGANv1.4.pth')
        if not os.path.exists(face_model_path):
            print("[DOWNLOAD] Downloading GFPGAN face restoration model...")
            import urllib.request
            urllib.request.urlretrieve(
                'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                face_model_path
            )
            print("[OK] GFPGAN model downloaded.")
        
        enhancer = GFPGANer(
            model_path=face_model_path,
            upscale=1,  # We handle upscaling separately
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=device
        )
        _face_enhancer_cache[device_name] = enhancer
        print(f"[OK] GFPGAN face enhancer loaded on {device_name}.")
        return enhancer
    except Exception as e:
        print(f"[WARNING] GFPGAN not available: {e}")
        return None


# ======================== IMAGE ANALYSIS ======================================

def detect_block_size(img, max_k=8):
    """
    Detect if the image was upscaled using nearest-neighbor.
    Returns the block size K (1 if no blockiness detected).
    """
    try:
        # Convert image to grayscale numpy array for fast analysis
        gray = img.convert('L')
        arr = np.array(gray)
        h, w = arr.shape
        
        if h < 32 or w < 32:
            return 1
            
        # Sample lines to make it extremely fast
        rows = np.linspace(5, h - 6, 30, dtype=int)
        cols = np.linspace(5, w - 6, 30, dtype=int)
        
        run_lengths = []
        
        # Analyze rows
        for r in rows:
            row = arr[r, :]
            # Find runs of nearly identical pixels (handling JPEG compression noise)
            diff = np.abs(row[1:].astype(int) - row[:-1].astype(int))
            zeros = (diff <= 2)
            run_len = 0
            for is_zero in zeros:
                if is_zero:
                    run_len += 1
                else:
                    if run_len > 0:
                        run_lengths.append(run_len + 1)
                        run_len = 0
            if run_len > 0:
                run_lengths.append(run_len + 1)
                
        # Analyze columns
        for c in cols:
            col = arr[:, c]
            diff = np.abs(col[1:].astype(int) - col[:-1].astype(int))
            zeros = (diff <= 2)
            run_len = 0
            for is_zero in zeros:
                if is_zero:
                    run_len += 1
                else:
                    if run_len > 0:
                        run_lengths.append(run_len + 1)
                        run_len = 0
            if run_len > 0:
                run_lengths.append(run_len + 1)
                
        if not run_lengths:
            return 1
            
        # Count frequencies of run lengths
        from collections import Counter
        counts = Counter(run_lengths)
        
        total_runs = len(run_lengths)
        if total_runs < 20:
            return 1
            
        best_k = 1
        best_score = 0.0
        
        for k in range(2, max_k + 1):
            multiples_count = sum(count for val, count in counts.items() if val % k == 0)
            score = multiples_count / total_runs
            
            k_frequency = counts.get(k, 0) / total_runs
            combined_score = score * 0.6 + k_frequency * 0.4
            
            if combined_score > 0.45 and combined_score > best_score:
                best_score = combined_score
                best_k = k
                
        if best_k > 1:
            print(f"[DE-BLOCK] Detected block size: {best_k}x (Confidence score: {best_score:.2f})")
        return best_k
    except Exception as e:
        print(f"[WARNING] Block size detection failed: {e}")
        return 1

def classify_image_type(img):
    """
    Classify image as 'anime' (illustrations/line-art) or 'photo'.
    """
    try:
        small = img.resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(small)
        
        gray = small.convert('L')
        gray_arr = np.array(gray).astype(float)
        dy, dx = np.gradient(gray_arr)
        gradient_mag = np.sqrt(dx*dx + dy*dy)
        
        flat_ratio = np.sum(gradient_mag < 2.0) / gradient_mag.size
        
        quantized = (arr // 16) * 16
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        
        print(f"[ANALYSIS] Image Classification Stats - Flat Ratio: {flat_ratio:.3f}, Quantized Colors: {unique_colors}")
        
        if flat_ratio > 0.72 and unique_colors < 150:
            return 'anime'
        return 'photo'
    except Exception as e:
        print(f"[WARNING] Image type classification failed: {e}")
        return 'photo'

def estimate_noise_level(img):
    """
    Estimate image noise level using block variance in grayscale.
    """
    try:
        w, h = img.size
        if w > 512 or h > 512:
            img = img.resize((512, int(512 * h / w)), Image.Resampling.BILINEAR)
            w, h = img.size
            
        gray = img.convert('L')
        arr = np.array(gray).astype(float)
        
        block_size = 16
        bx = w // block_size
        by = h // block_size
        
        if bx == 0 or by == 0:
            return 0.0
            
        variances = []
        for j in range(by):
            for i in range(bx):
                block = arr[j*block_size:(j+1)*block_size, i*block_size:(i+1)*block_size]
                var = np.var(block)
                variances.append(var)
                
        if not variances:
            return 0.0
            
        variances.sort()
        # Trimmed median: exclude bottom 15% (flat blocks) and top 50% (strong edges)
        # to calculate a robust noise variance representing background texture and noise
        start_idx = int(len(variances) * 0.15)
        end_idx = int(len(variances) * 0.50)
        start_idx = max(0, start_idx)
        end_idx = max(start_idx + 1, end_idx)
        trimmed_variances = variances[start_idx:end_idx]
        median_noise = trimmed_variances[len(trimmed_variances) // 2]
        
        print(f"[ANALYSIS] Estimated noise level (variance, trimmed): {median_noise:.2f}")
        return median_noise
    except Exception as e:
        print(f"[WARNING] Noise level estimation failed: {e}")
        return 0.0

def apply_cinematic_grading(img_np, strength):
    """
    Apply professional Hollywood Teal & Orange split-toning color grade.
    img_np: numpy float array [0, 255] with shape (H, W, 3)
    strength: float [0.0, 1.0]
    """
    # Calculate luminance
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    
    norm_lum = lum / 255.0
    
    # 1. Filmic S-curve on luminance with a subtle black lift (matte cinematic look)
    black_lift = 0.04 * strength
    norm_lifted = black_lift + (1.0 - black_lift) * norm_lum
    s_curve = norm_lifted * norm_lifted * (3.0 - 2.0 * norm_lifted)
    
    # Blend original luminance with s-curve
    lum_target = (norm_lum + (s_curve - norm_lum) * 0.4 * strength) * 255.0
    
    # 2. Teal & Orange Split Toning
    # cool shadows (teal): active in dark regions
    shadow_mask = np.clip(1.0 - norm_lum * 2.2, 0, 1)
    # warm highlights (orange): active in bright regions
    highlight_mask = np.clip((norm_lum - 0.45) * 2.0, 0, 1)
    
    # Teal shift: reduce red, increase blue, keep green neutral
    r_adj = r * (1.0 - shadow_mask * 0.18 * strength)
    g_adj = g * (1.0 - shadow_mask * 0.02 * strength)
    b_adj = b * (1.0 + shadow_mask * 0.12 * strength)
    
    # Orange shift: increase red and green slightly, reduce blue
    r_adj = r_adj * (1.0 + highlight_mask * 0.15 * strength)
    g_adj = g_adj * (1.0 + highlight_mask * 0.03 * strength)
    b_adj = b_adj * (1.0 - highlight_mask * 0.18 * strength)
    
    # 3. Desaturate cool shadows to prevent color noise in black areas
    lum_adj = 0.299 * r_adj + 0.587 * g_adj + 0.114 * b_adj
    shadow_desat = shadow_mask * 0.3 * strength
    r_adj = r_adj * (1.0 - shadow_desat) + lum_adj * shadow_desat
    g_adj = g_adj * (1.0 - shadow_desat) + lum_adj * shadow_desat
    b_adj = b_adj * (1.0 - shadow_desat) + lum_adj * shadow_desat
    
    # Re-calculate luminance after color/saturation adjustments
    lum_adj = 0.299 * r_adj + 0.587 * g_adj + 0.114 * b_adj
    
    # 4. Enforce exact target filmic luminance ratio
    mask_zero = (lum_adj > 0.001)
    ratio = np.ones_like(lum_adj)
    ratio[mask_zero] = lum_target[mask_zero] / lum_adj[mask_zero]
    
    r_final = np.clip(r_adj * ratio, 0, 255)
    g_final = np.clip(g_adj * ratio, 0, 255)
    b_final = np.clip(b_adj * ratio, 0, 255)
    
    return np.stack([r_final, g_final, b_final], axis=2)

def apply_advanced_filters(img, options):
    """
    Apply advanced fine-tuning filters to the upscaled PIL image.
    Options keys: 'sharpening', 'detail', 'contrast', 'colorBoost', 'denoise', 'grain', 'cinematic'
    """
    try:
        denoise_level = int(options.get('denoise', 0))
        detail_level = int(options.get('detail', 0))
        sharpening_level = int(options.get('sharpening', 0))
        contrast_level = int(options.get('contrast', 0))
        color_boost = int(options.get('colorBoost', 0))
        grain_level = int(options.get('grain', 0))
        cinematic_level = int(options.get('cinematic', 0))
        
        # 1. Edge-Preserving Denoise (OpenCV Bilateral Filter) - Moved to Pre-processing to prevent noise amplification
        pass
            
        # 2. Dual-Stage Adaptive Sharpening (Topaz-style micro-sharpness + local clarity pop)
        if sharpening_level > 0:
            factor = (sharpening_level / 100.0)
            # Stage 1: Fine micro-sharpening (captures texture details)
            percent_fine = int(factor * 120)
            img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=percent_fine, threshold=0))
            # Stage 2: Broad clarity/structure (makes details and boundaries pop, no halos)
            percent_broad = int(factor * 45)
            img = img.filter(ImageFilter.UnsharpMask(radius=2.5, percent=percent_broad, threshold=1))
            
        # 3. Local Contrast
        if contrast_level > 0:
            factor = 1.0 + (contrast_level / 100.0) * 0.2
            img = ImageEnhance.Contrast(img).enhance(factor)
            
        # 4. Perceptual Color Boost
        if color_boost > 0:
            factor = 1.0 + (color_boost / 100.0) * 0.35
            img = ImageEnhance.Color(img).enhance(factor)
            
        # 5. Chunked processing for memory-heavy operations (Detail, Cinematic, Grain)
        if detail_level > 0 or cinematic_level > 0 or grain_level > 0:
            width, height = img.size
            # Create a new empty image of the same size to compile the output chunks
            out_img = Image.new('RGB', (width, height))
            
            # Use 1024 or 2048 as chunk height depending on total size
            chunk_h = 1024 if (width * height > 4000 * 4000) else 2048
            margin = 8  # 8px margin to completely avoid Gaussian Blur boundary seams
            
            for y_start in range(0, height, chunk_h):
                y_end = min(height, y_start + chunk_h)
                
                # Crop the chunk with margins for boundary-accurate blur
                crop_y_start = max(0, y_start - margin)
                crop_y_end = min(height, y_end + margin)
                chunk_pil = img.crop((0, crop_y_start, width, crop_y_end))
                
                chunk_arr = np.array(chunk_pil).astype(np.float32)
                
                # A. Apply detail injection on the chunk
                if detail_level > 0:
                    blurred_chunk = chunk_pil.filter(ImageFilter.GaussianBlur(radius=1.0))
                    blurred_arr = np.array(blurred_chunk).astype(np.float32)
                    detail_band = chunk_arr - blurred_arr
                    
                    # Mask detail band near edges (increased threshold to 40.0 for Topaz-level edge clarity)
                    abs_detail = np.abs(detail_band[:, :, :3])
                    threshold = np.float32(40.0)
                    scale_mask = np.ones_like(detail_band[:, :, :3], dtype=np.float32)
                    large_edges = abs_detail > threshold
                    scale_mask[large_edges] = threshold / abs_detail[large_edges]
                    
                    factor = np.float32((detail_level / 100.0) * 0.95)
                    chunk_arr[:, :, :3] = np.clip(chunk_arr[:, :, :3] + detail_band[:, :, :3] * (factor * scale_mask), 0.0, 255.0).astype(np.float32)
                
                # B. Slice back to target coordinates (removing margins)
                # Calculate coordinates of the target chunk relative to the cropped chunk
                rel_y_start = y_start - crop_y_start
                rel_y_end = rel_y_start + (y_end - y_start)
                target_chunk_arr = chunk_arr[rel_y_start:rel_y_end, :, :3].copy()
                
                # C. Apply Cinematic Grading (pixel-wise)
                if cinematic_level > 0:
                    target_chunk_arr = apply_cinematic_grading(target_chunk_arr, cinematic_level / 100.0).astype(np.float32)
                    
                # D. Apply Film Grain (pixel-wise multi-scale micro-texture)
                if grain_level > 0:
                    factor = np.float32((grain_level / 100.0) * 14.0)
                    
                    r_c = target_chunk_arr[:, :, 0]
                    g_c = target_chunk_arr[:, :, 1]
                    b_c = target_chunk_arr[:, :, 2]
                    # Compute luminance in float32
                    lum_c = (np.float32(0.299) * r_c + np.float32(0.587) * g_c + np.float32(0.114) * b_c) / np.float32(255.0)
                    
                    grain_mask = np.clip(np.float32(4.0) * lum_c * (np.float32(1.0) - lum_c), 0.0, 1.0).astype(np.float32)
                    grain_mask = np.expand_dims(grain_mask, axis=2)
                    
                    # 1. Fine-scale sensor grain directly in float32
                    noise_fine = np.random.normal(0, factor * 0.65, (target_chunk_arr.shape[0], target_chunk_arr.shape[1], 1)).astype(np.float32)
                    
                    # 2. Medium-scale organic micro-textures (for skin pores and fabric weaves)
                    h, w = target_chunk_arr.shape[0], target_chunk_arr.shape[1]
                    small_h, small_w = max(2, h // 2), max(2, w // 2)
                    noise_med_small = np.random.normal(0, factor * 0.45, (small_h, small_w, 1)).astype(np.float32)
                    
                    # Convert to temp array to upscale bilinearly with PIL
                    noise_med_pil = Image.fromarray(np.squeeze((noise_med_small + 128.0).clip(0, 255).astype(np.uint8)))
                    noise_med_pil = noise_med_pil.resize((w, h), Image.Resampling.BILINEAR)
                    noise_med = (np.array(noise_med_pil).astype(np.float32) - 128.0)
                    noise_med = np.expand_dims(noise_med, axis=2)
                    
                    # Total organic multi-scale texture in float32
                    total_texture = noise_fine + noise_med
                    target_chunk_arr = np.clip(target_chunk_arr + total_texture * grain_mask, 0.0, 255.0).astype(np.float32)
                    
                # Paste the processed chunk back to final image
                processed_chunk_pil = Image.fromarray(target_chunk_arr.astype(np.uint8))
                out_img.paste(processed_chunk_pil, (0, y_start))
                
                # Free memory immediately to prevent accumulation across chunks
                del chunk_arr, target_chunk_arr, processed_chunk_pil
                import gc; gc.collect()
                
            img = out_img
            
        return img
    except Exception as e:
        print(f"[WARNING] Advanced filtering failed: {e}")
        traceback.print_exc()
        return img


# ======================== UPSCALE PIPELINE ====================================

def upscale_image(img_bytes, model_name, target_scale, options):
    """
    Professional multi-stage upscale pipeline:
    1. Decode & analyze image (Auto model selection, De-blockify check)
    2. Pre-process and apply De-blockify downscaling if needed
    3. Run Real-ESRGAN super-resolution (CUDA/CPU)
    4. Handle scale constraints (stacking / scaling)
    5. Optional face restoration (GFPGAN)
    6. Apply advanced fine-tuning filters in Python
    7. Encode output
    """
    start = time.time()
    global _should_abort
    _should_abort = False
    
    # Dynamically release previously cached SD models to free VRAM for tiled upscaling
    free_gpu_memory()
    
    global _current_progress
    _current_progress.update({
        'active': True,
        'tile_idx': 0,
        'total_tiles': 0,
        'speed': 0.0,
        'eta': '',
        'stage': 'Analyzing image...',
        'percent': 5
    })
    
    # Save input image for debugging/verification
    try:
        input_save_path = os.path.join(DIR, "last_input.png")
        with open(input_save_path, "wb") as f:
            f.write(img_bytes)
        print(f"[DEBUG] Saved last input image to {input_save_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save debug input image: {e}")
        
    # --- 1. Decode input ---
    input_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    orig_w, orig_h = input_img.size
    
    # --- Resolve device and compute input megapixels early ---
    input_megapixels = (orig_w * orig_h) / 1e6
    device_name = options.get('device', 'cuda')
    _init_torch()
    if device_name == 'cpu' or not _torch.cuda.is_available():
        device_name = 'cpu'
        is_bf16 = False
    else:
        device_name = 'cuda'
        is_bf16 = _torch.cuda.is_bf16_supported()

    # --- Enhance Mode (scale=1): AI detail transfer for print quality ---
    enhance_mode = (target_scale == 1)
    print_quality = options.get('print_quality', False)
    
    if enhance_mode:
        print(f"[INPUT] Original size: {orig_w}x{orig_h} -> ENHANCE MODE (AI detail transfer, same resolution)")
    else:
        print(f"[INPUT] Original size: {orig_w}x{orig_h} -> target {target_scale}x = {orig_w*target_scale}x{orig_h*target_scale}")
    
    if print_quality:
        import cv2
        
        # Determine the working scale for AI processing
        work_scale = 2 if enhance_mode else min(target_scale, 4)
        final_w = orig_w if enhance_mode else orig_w * target_scale
        final_h = orig_h if enhance_mode else orig_h * target_scale
        
        print(f"[PRINT] Detail transfer pipeline: AI@{work_scale}x -> blend -> {'downscale to original' if enhance_mode else f'{final_w}x{final_h}'}")
        
        # --- Step 1: Create Lanczos upscale (faithful, clean base) ---
        lanczos_w = orig_w * work_scale
        lanczos_h = orig_h * work_scale
        print(f"[PRINT] Step 1: Lanczos {work_scale}x upscale ({lanczos_w}x{lanczos_h})...")
        lanczos_up = input_img.resize((lanczos_w, lanczos_h), Image.Resampling.LANCZOS)
        lanczos_np = np.array(lanczos_up).astype(np.float32)
        
        model_info = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[DEFAULT_MODEL])
        use_replicate = (model_info.get('engine') == 'replicate')
        use_spandrel = (model_info.get('engine') == 'spandrel')
        native_scale = model_info['scale']
        
        if use_replicate:
            replicate_api_key = options.get('replicate_api_key', '').strip()
            if not replicate_api_key:
                replicate_api_key = os.environ.get('REPLICATE_API_TOKEN', '').strip()
            if not replicate_api_key:
                raise ValueError("Replicate API Token is required for Cloud SUPIR. Please paste it in the UI.")
            ai_output_rgb = _replicate_upscale(img_bytes, native_scale, replicate_api_key)
            ai_pil = Image.fromarray(ai_output_rgb)
        elif use_spandrel:
            spandrel_model = _get_spandrel_model(model_name, device_name=device_name)
            img_rgb = np.array(input_img)
            sp_tile, sp_pad = _get_adaptive_spandrel_tile_params(device_name)
            print(f"[RUN] Selected print tiling params: size={sp_tile}, pad={sp_pad} (bf16={is_bf16})")
            ai_output_rgb = _spandrel_tiled_upscale(spandrel_model, img_rgb, native_scale, tile_size=sp_tile, tile_pad=sp_pad, device_name=device_name)
            ai_pil = Image.fromarray(ai_output_rgb)
        else:
            upsampler = _build_upsampler(model_name, work_scale, device_name=device_name, input_megapixels=input_megapixels)
            img_np_bgr = np.array(input_img)[:, :, ::-1].copy()
            ai_output_np, _ = upsampler.enhance(img_np_bgr, outscale=work_scale)
            ai_pil = Image.fromarray(ai_output_np[:, :, ::-1])
        
        if device_name == 'cuda':
            _torch.cuda.empty_cache()
        
        # Make sure both are the same size
        if ai_pil.size != lanczos_up.size:
            ai_pil = ai_pil.resize(lanczos_up.size, Image.Resampling.LANCZOS)
        ai_np = np.array(ai_pil).astype(np.float32)
        
        # --- Step 3: Extract high-frequency detail from AI output ---
        # Blur the AI output to get its low-frequency component
        # Subtract to get the HIGH-FREQUENCY detail the AI generated
        blur_radius = 3  # Controls detail extraction scale
        print(f"[PRINT] Step 3: Extracting AI high-frequency detail (blur_radius={blur_radius})...")
        ai_lowfreq = cv2.GaussianBlur(ai_np, (0, 0), sigmaX=blur_radius)
        ai_detail = ai_np - ai_lowfreq  # High-frequency detail layer
        
        # Also extract Lanczos high-freq for comparison
        lanczos_lowfreq = cv2.GaussianBlur(lanczos_np, (0, 0), sigmaX=blur_radius)
        
        # --- Step 4: Blend AI detail onto Lanczos base ---
        # detail_strength controls how much AI texture to inject
        detail_strength = options.get('detail', 70) / 100.0  # 0.0 to 1.0
        detail_strength = max(0.3, min(1.0, detail_strength))  # clamp to useful range
        print(f"[PRINT] Step 4: Blending AI detail onto Lanczos base (strength={detail_strength:.0%})...")
        
        # Final = Lanczos base + (AI detail * strength)
        result_np = lanczos_np + (ai_detail * detail_strength)
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)
        
        # --- Step 5: Print-specific sharpening ---
        # Print requires clean sharpening without edge halos
        sharpening = options.get('sharpening', 50)
        if sharpening > 0:
            # Calibrated radius (0.6 - 1.2) prevents halo artifacts
            radius = 0.6 + (sharpening / 100.0) * 0.6
            amount = 30 + (sharpening / 100.0) * 70
            print(f"[PRINT] Step 5: Clean print sharpening (radius={radius:.2f}, amount={amount:.0f}%)")
            result_pil = Image.fromarray(result_np)
            result_pil = result_pil.filter(
                ImageFilter.UnsharpMask(radius=radius, percent=int(amount), threshold=1)
            )
            result_np = np.array(result_pil)
        
        # --- Step 6: CLAHE micro-contrast for print pop ---
        print("[PRINT] Step 6: CLAHE micro-contrast for print definition...")
        lab = cv2.cvtColor(result_np, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        # Reduced and contrast-dependent clip limit prevents grain amplification
        clip_limit = 1.0 + (options.get('contrast', 50) / 100.0) * 0.3
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        result_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # --- Step 7: Resize to final dimensions ---
        output_pil = Image.fromarray(result_np)
        if output_pil.size != (final_w, final_h):
            print(f"[PRINT] Step 7: Resizing to final {final_w}x{final_h}...")
            output_pil = output_pil.resize((final_w, final_h), Image.Resampling.LANCZOS)
        
        out_w, out_h = output_pil.size
        
        # Encode
        elapsed = time.time() - start
        use_jpeg = (out_w * out_h >= 4000 * 4000)
        if use_jpeg:
            buf = io.BytesIO()
            output_pil.save(buf, format='JPEG', quality=95)
            mime_type = 'image/jpeg'
        else:
            buf = io.BytesIO()
            output_pil.save(buf, format='PNG', compress_level=3)
            mime_type = 'image/png'
        
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        data_uri = f"data:{mime_type};base64,{b64}"
        size_mb = len(b64) * 3 / 4 / (1024 * 1024)
        print(f"[DONE] Print-quality {out_w}x{out_h} in {elapsed:.1f}s ({size_mb:.1f} MB)")
        
        # Save debug output
        try:
            output_save_path = os.path.join(DIR, "last_output.png")
            output_pil.save(output_save_path)
            print(f"[DEBUG] Saved last output image to {output_save_path}")
        except Exception:
            pass
        
        return {
            "success": True,
            "image": data_uri,
            "width": out_w,
            "height": out_h,
            "original_width": orig_w,
            "original_height": orig_h,
            "model_used": f"print-detail-transfer ({model_name})",
            "scale_used": target_scale if not enhance_mode else 1,
            "time_seconds": round(elapsed, 1)
        }
    
    # --- 2. Analyze Properties ---
    deblock_enabled = options.get('deblock', False)
    block_size = detect_block_size(input_img) if deblock_enabled else 1
    noise_lvl = estimate_noise_level(input_img)
    detected_type = classify_image_type(input_img)
    
    # Resolve device to run on
    device_name = options.get('device', 'cuda')
    _init_torch()
    if device_name == 'cpu' or not _torch.cuda.is_available():
        device_name = 'cpu'
        dev = _torch.device('cpu')
    else:
        device_name = 'cuda'
        dev = _torch.device('cuda')
    print(f"[RUN] Upscaling executing on: {device_name.upper()}")
    
    # If device is CPU, override heavy models with compact models for processing speed
    if device_name == 'cpu':
        if model_name == 'auto':
            if detected_type == 'anime':
                model_name = 'realesr-animevideov3'
                print("[CPU AUTO-MODEL] Selected compact anime model: realesr-animevideov3")
            else:
                model_name = 'realesr-general-x4v3'
                print("[CPU AUTO-MODEL] Selected compact general model: realesr-general-x4v3")
        elif model_name in ('realesrgan-x4plus', 'realesrnet-x4plus', 'realesrgan-x2plus'):
            print(f"[CPU OVERRIDE] Heavy model '{model_name}' is too slow on CPU. Overriding with 'realesr-general-x4v3' for speed.")
            model_name = 'realesr-general-x4v3'
        elif model_name == 'realesrgan-x4plus-anime':
            print(f"[CPU OVERRIDE] Heavy anime model '{model_name}' is too slow on CPU. Overriding with 'realesr-animevideov3' for speed.")
            model_name = 'realesr-animevideov3'
    else:
        # Auto model selection on GPU
        if model_name == 'auto':
            if detected_type == 'anime':
                model_name = 'realesrgan-x4plus-anime'
                print("[AUTO-MODEL] Selected Anime Model (realesrgan-x4plus-anime)")
            elif options.get('preset') == 'maximum':
                model_name = 'hat-realesrgan-blend'
                print("[AUTO-MODEL] Selected SOTA Model Ensemble Blend (hat-realesrgan-blend) for Maximum detail")
            else:
                # Use SOTA HAT GAN Sharper by default for ultimate details (noise is pre-cleaned by bilateral filter)
                model_name = 'hat-sharper'
                print("[AUTO-MODEL] Selected SOTA HAT GAN Sharper Model (hat-sharper) for maximum details")
                
    # Auto preset settings
    if options.get('preset') == 'auto':
        if detected_type == 'anime':
            options['sharpening'] = 50
            options['detail'] = 10
            options['contrast'] = 30
            options['colorBoost'] = 15
            options['denoise'] = 0
            options['grain'] = 0
        elif noise_lvl > 8.0:
            options['sharpening'] = 30
            options['detail'] = 40
            options['contrast'] = 45
            options['colorBoost'] = 20
            options['denoise'] = 40
            options['grain'] = 8
        elif noise_lvl > 3.0:
            options['sharpening'] = 45
            options['detail'] = 50
            options['contrast'] = 45
            options['colorBoost'] = 25
            options['denoise'] = 12
            options['grain'] = 12
        else:
            options['sharpening'] = 60
            options['detail'] = 75
            options['contrast'] = 50
            options['colorBoost'] = 30
            options['denoise'] = 12
            options['grain'] = 12
            
    model_info = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[DEFAULT_MODEL])
    native_scale = model_info['scale']
    
    # Apply De-Blockify point downscaling
    if block_size > 1 and not enhance_mode:
        print(f"[DE-BLOCK] Applying 1/{block_size}x Box downsampling to restore native low-res grid...")
        down_w = orig_w // block_size
        down_h = orig_h // block_size
        input_img = input_img.resize((down_w, down_h), Image.Resampling.BOX)
        ai_scale = target_scale * block_size
    elif enhance_mode:
        ai_scale = native_scale
    else:
        ai_scale = target_scale
    
    # --- Memory Safety: Cap AI scale to prevent OOM ---
    input_w, input_h = input_img.size
    input_megapixels = (input_w * input_h) / 1e6
    output_megapixels = input_megapixels * (ai_scale ** 2)
    
    max_megapixels = MAX_GPU_MEGAPIXELS if device_name == 'cuda' else MAX_CPU_MEGAPIXELS
    if output_megapixels > max_megapixels:
        safe_scale = int((max_megapixels / input_megapixels) ** 0.5)
        safe_scale = max(safe_scale, 2)  # minimum 2x
        print(f"[SAFETY] Output would be {output_megapixels:.0f} MP ({int(input_w*ai_scale)}x{int(input_h*ai_scale)}) — exceeds {max_megapixels} MP limit.")
        print(f"[SAFETY] Capping AI scale from {ai_scale}x to {safe_scale}x to prevent out-of-memory crash.")
        ai_scale = safe_scale
        output_megapixels = input_megapixels * (ai_scale ** 2)
        
    img_np = np.array(input_img)[:, :, ::-1]  # RGB -> BGR
    
    # --- 3. Pre-processing ---
    denoise_level = int(options.get('denoise', 0))
    if denoise_level > 0:
        import cv2
        d = 3 if denoise_level < 20 else (5 if denoise_level < 50 else 7)
        sigma_color = denoise_level * 0.5
        sigma_space = denoise_level * 0.35
        denoised = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
        alpha = min(1.0, denoise_level / 45.0)
        img_np = cv2.addWeighted(img_np, 1.0 - alpha, denoised, alpha, 0)
        print(f"[PRE-PROCESS] Applied bilateral denoise filter to low-res input (level={denoise_level}, alpha={alpha:.2f})")
        
    deblur = options.get('deblur', 'none')
    if deblur == 'mild':
        input_img_proc = Image.fromarray(img_np[:, :, ::-1])
        input_img_proc = input_img_proc.filter(ImageFilter.UnsharpMask(radius=0.8, percent=80, threshold=0))
        img_np = np.array(input_img_proc)[:, :, ::-1]
    elif deblur == 'heavy':
        input_img_proc = Image.fromarray(img_np[:, :, ::-1])
        input_img_proc = input_img_proc.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=0))
        enhancer = ImageEnhance.Contrast(input_img_proc)
        input_img_proc = enhancer.enhance(1.08)
        img_np = np.array(input_img_proc)[:, :, ::-1]
    
    # --- 3.5. Pre-upscale Generative Refinement (For High Scale Details & Speed) ---
    sd_refine = options.get('sd_refine', False)
    sd_refine_pre = sd_refine and (options.get('sd_refine_stage', 'post') == 'pre')
    if sd_refine_pre:
        _current_progress.update({'stage': 'Pre-upscale SD refinement...', 'percent': 20})
        print("[PRE-PROCESS] Applying Stable Diffusion 1.5 + ControlNet Tile texture refinement to low-res input...")
        try:
            prompt = options.get('sd_prompt') or 'raw photo, highly detailed, sharp focus, 8k, realistic textures, dslr, 35mm lens, film grain'
            neg_prompt = options.get('sd_neg_prompt') or 'airbrushed, plastic, waxy, CGI, 3D, render, digital art, smooth skin, oily skin, blurry, low quality, cartoon, painting, drawing, illustration'
            strength = options.get('sd_strength', 30) / 100.0
            mode = options.get('sd_mode', 'hybrid')
            tile_size_opt = options.get('sd_tile_size', 0)
            controlnet_scale = options.get('sd_controlnet_strength', 0.70)
            detail_boost = options.get('sd_detail_boost', 1.50)
            detail_threshold = options.get('sd_detail_threshold', 50.0)
            
            img_np = apply_sd_controlnet_refine(
                img_np, 
                prompt, 
                neg_prompt, 
                denoising_strength=strength, 
                device_name=device_name, 
                mode=mode,
                tile_size_override=tile_size_opt,
                controlnet_scale=controlnet_scale,
                detail_boost=detail_boost,
                detail_threshold=detail_threshold,
                sd_refine_stage='pre'
            )
            print("[PRE-PROCESS] Pre-upscale Stable Diffusion refinement complete.")
            # Clear SD model from VRAM to make room for Spandrel upscaler
            free_gpu_memory()
        except Exception as e:
            print(f"[WARNING] Pre-upscale Stable Diffusion refinement failed: {e}")
            traceback.print_exc()
            
    # --- 4. Super-Resolution ---
    if device_name == 'cuda':
        print("[GPU] Clearing CUDA cache before upscaling...")
        _torch.cuda.empty_cache()

    model_info = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[DEFAULT_MODEL])
    native_scale = model_info['scale']
    use_replicate = (model_info.get('engine') == 'replicate')
    use_spandrel = (model_info.get('engine') == 'spandrel')
    use_blend = (model_info.get('engine') == 'blend')
    double_pass = options.get('double_pass', False) and not use_spandrel and not use_replicate and not use_blend
    
    if use_replicate:
        # --- Replicate Cloud path (SUPIR) ---
        print(f"[RUN] Using Replicate Cloud engine for {model_name}...")
        replicate_api_key = options.get('replicate_api_key', '').strip()
        if not replicate_api_key:
            replicate_api_key = os.environ.get('REPLICATE_API_TOKEN', '').strip()
        if not replicate_api_key:
            raise ValueError("Replicate API Token is required for Cloud SUPIR. Please paste it in the UI.")
        
        output_rgb = _replicate_upscale(img_bytes, ai_scale, replicate_api_key)
        # Convert RGB to BGR for consistency
        output_np = output_rgb[:, :, ::-1].copy()
    elif use_spandrel:
        # --- Spandrel path (HAT, SwinIR, etc.) ---
        print(f"[RUN] Using spandrel engine for {model_name}...")
        spandrel_model = _get_spandrel_model(model_name, device_name=device_name)
        # Spandrel expects RGB input, img_np is currently BGR
        img_rgb = img_np[:, :, ::-1].copy()
        # Determine tile size and pad based on VRAM and bf16 support
        sp_tile, sp_pad = _get_adaptive_spandrel_tile_params(device_name)
        print(f"[RUN] Selected tiling params: size={sp_tile}, pad={sp_pad} (bf16={is_bf16})")
        output_rgb = _spandrel_tiled_upscale(spandrel_model, img_rgb, native_scale, tile_size=sp_tile, tile_pad=sp_pad, device_name=device_name)
        # If target scale != native scale, resize
        target_h, target_w = int(input_img.size[1] * ai_scale), int(input_img.size[0] * ai_scale)
        if output_rgb.shape[1] != target_w or output_rgb.shape[0] != target_h:
            output_pil_temp = Image.fromarray(output_rgb)
            output_pil_temp = output_pil_temp.resize((target_w, target_h), Image.Resampling.LANCZOS)
            output_rgb = np.array(output_pil_temp)
        # Convert RGB to BGR for consistency with rest of pipeline
        output_np = output_rgb[:, :, ::-1].copy()
    elif use_blend:
        # --- Model Blend path (HAT + RealESRGAN) ---
        print(f"[RUN] Using Model Ensemble Blend engine for {model_name}...")
        # 1. Run HAT-Sharper (Spandrel)
        spandrel_model = _get_spandrel_model('hat-sharper', device_name=device_name)
        img_rgb = img_np[:, :, ::-1].copy()
        sp_tile, sp_pad = _get_adaptive_spandrel_tile_params(device_name)
        output_rgb_hat = _spandrel_tiled_upscale(spandrel_model, img_rgb, 4, tile_size=sp_tile, tile_pad=sp_pad, device_name=device_name)
        
        target_h, target_w = int(input_img.size[1] * ai_scale), int(input_img.size[0] * ai_scale)
        if output_rgb_hat.shape[1] != target_w or output_rgb_hat.shape[0] != target_h:
            output_pil_temp = Image.fromarray(output_rgb_hat)
            output_pil_temp = output_pil_temp.resize((target_w, target_h), Image.Resampling.LANCZOS)
            output_rgb_hat = np.array(output_pil_temp)
        output_np_hat = output_rgb_hat[:, :, ::-1].copy()
        
        # 2. Run RealESRGAN
        upsampler = _build_upsampler('realesrgan-x4plus', ai_scale, device_name=device_name, input_megapixels=input_megapixels)
        output_np_realesrgan, _ = upsampler.enhance(img_np, outscale=ai_scale)
        
        # 3. Blend them (60% HAT, 40% RealESRGAN by default)
        blend_weight = options.get('model_blend_weight', 40) / 100.0
        output_np = (output_np_hat.astype(np.float32) * (1.0 - blend_weight) + output_np_realesrgan.astype(np.float32) * blend_weight).astype(np.uint8)
        print(f"[OK] Model blend complete (weight: {blend_weight:.0%})")
    else:
        # --- RealESRGAN path ---
        upsampler = _build_upsampler(model_name, ai_scale, device_name=device_name, input_megapixels=input_megapixels)
        
        # Set up scale factor splitting for double-pass
        if double_pass:
            if ai_scale >= native_scale:
                pass1_scale = native_scale
                pass2_scale = ai_scale / native_scale
                
                # Safety: check if Pass 2's INPUT fits in VRAM/RAM
                pass1_output_mp = input_megapixels * (pass1_scale ** 2)
                max_pass2_input_mp = 30 if device_name == 'cuda' else 20
                if pass1_output_mp > max_pass2_input_mp:
                    print(f"[SAFETY] Double-pass disabled: Pass 1 output would be {pass1_output_mp:.0f} MP, "
                          f"too large for Pass 2 on {device_name.upper()} (limit {max_pass2_input_mp} MP). Using single pass.")
                    double_pass = False
            else:
                print(f"[RUN] Target scale {ai_scale}x is less than model native scale {native_scale}x. Skipping double pass.")
                double_pass = False
                
        if double_pass:
            print(f"[RUN] Pass 1: {pass1_scale}x super-resolution...")
            output_np, _ = upsampler.enhance(img_np, outscale=pass1_scale)
            
            print(f"[RUN] Pass 2: {pass2_scale:.2f}x super-resolution (stacking)...")
            output_np, _ = upsampler.enhance(output_np, outscale=pass2_scale)
        else:
            # --- CONTEXT-AWARE MULTI-PASS FOR EXTREME UPSCALES ---
            sd_refine = options.get('sd_refine', False)
            sd_refine_pre = sd_refine and (options.get('sd_refine_stage', 'post') == 'pre')
            
            if ai_scale > 4.0 and sd_refine and not sd_refine_pre:
                print(f"[RUN] Splitting {ai_scale}x upscale into Multi-Pass to provide SD texture generator with context...")
                
                # Pass 1: Upscale to 4x (Optimal context resolution)
                print(f"[RUN] Context Pass 1: 4.00x base upscaling...")
                output_np, _ = upsampler.enhance(img_np, outscale=4.0)
                
                # Run SD Refine at 4x
                _current_progress.update({'stage': 'SD context texture refinement...', 'percent': 50})
                print(f"[RUN] Injecting SD texture refinement at 4x for optimal global context...")
                
                prompt = options.get('sd_prompt', 'raw photo, highly detailed, sharp focus, 8k, realistic textures, dslr, 35mm lens, film grain')
                neg_prompt = options.get('sd_neg_prompt', 'airbrushed, plastic, waxy, CGI, 3D, render, digital art, smooth skin, oily skin, blurry, low quality, cartoon, painting, drawing, illustration')
                strength = options.get('sd_strength', 20) / 100.0
                mode = options.get('sd_mode', 'hybrid')
                tile_size_opt = options.get('sd_tile_size', 0)
                controlnet_scale = options.get('sd_controlnet_strength', 0.80)
                detail_boost = options.get('sd_detail_boost', 1.30)
                detail_threshold = options.get('sd_detail_threshold', 50.0)
                
                output_np = apply_sd_controlnet_refine(
                    output_np, prompt, neg_prompt, denoising_strength=strength, 
                    device_name=device_name, mode=mode, tile_size_override=tile_size_opt,
                    controlnet_scale=controlnet_scale, detail_boost=detail_boost,
                    detail_threshold=detail_threshold, sd_refine_stage='post'
                )
                
                # Flag to prevent it running AGAIN at step 7.5
                options['_sd_already_run'] = True
                
                # Pass 2: Upscale from 4x to target scale
                pass2_scale = ai_scale / 4.0
                print(f"[RUN] Context Pass 2: {pass2_scale:.2f}x final mathematical stretch...")
                
                # Clear memory to prevent fragmentation before the massive 8x stretch
                free_gpu_memory()
                upsampler = _build_upsampler(model_name, pass2_scale, device_name=device_name, input_megapixels=input_megapixels)
                output_np, _ = upsampler.enhance(output_np, outscale=pass2_scale)
            else:
                print(f"[RUN] Single pass: {native_scale}x super-resolution (target outscale={ai_scale}x)...")
                output_np, _ = upsampler.enhance(img_np, outscale=ai_scale)
    
    if device_name == 'cuda':
        print("[GPU] Clearing CUDA cache after upscaling...")
        _torch.cuda.empty_cache()
    
    # --- 5. Face Restoration (GFPGAN) ---
    face_restore = options.get('face_restore', False)
    if face_restore:
        try:
            enhancer = _get_face_enhancer(device=dev)
            if enhancer is not None:
                _current_progress.update({'stage': 'Face restoration (GFPGAN)...', 'percent': 70})
                print("[RUN] Face restoration (GFPGAN)...")
                _, _, output_np = enhancer.enhance(
                    output_np,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
                print("[OK] Face restoration complete.")
        except Exception as e:
            print(f"[WARNING] Face restoration failed: {e}")
            
    # --- 4.5. Generative Texture Refinement (Moved to step 7.5 to prevent blurring during 8x resize) ---

    # --- Color Matching ---
    if options.get('color_match', False):
        _current_progress.update({'stage': 'LAB color matching...', 'percent': 90})
        print("[POST] Applying LAB color matching to preserve original color profile...")
        try:
            lr_np = np.array(input_img)[:, :, ::-1].copy() # LR original BGR
            output_np = apply_color_matching(lr_np, output_np)
            print("[OK] LAB color matching complete.")
        except Exception as e:
            print(f"[WARNING] LAB color matching failed: {e}")

    # Convert BGR numpy array to PIL Image
    output_pil = Image.fromarray(output_np[:, :, ::-1])
    
    # --- 6. Advanced Post-processing Filters ---
    _current_progress.update({'stage': 'Applying advanced filters...', 'percent': 95})
    print("[RUN] Applying advanced fine-tuning filters...")
    output_pil = apply_advanced_filters(output_pil, options)
    
    # Support legacy simple sharpening if advanced sliders not present
    if 'sharpening' not in options:
        post_sharp = options.get('post_sharpen', 'light')
        if post_sharp == 'light':
            output_pil = output_pil.filter(ImageFilter.UnsharpMask(radius=0.5, percent=40, threshold=2))
        elif post_sharp == 'medium':
            output_pil = output_pil.filter(ImageFilter.UnsharpMask(radius=0.8, percent=65, threshold=1))
        elif post_sharp == 'strong':
            output_pil = output_pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=1))
            
    # --- 7. Enforce Exact Target Dimensions ---
    out_w, out_h = output_pil.size
    if enhance_mode:
        # Downscale back to original resolution for enhancement
        print(f"[ENHANCE] Downscaling from {out_w}x{out_h} back to original {orig_w}x{orig_h}...")
        output_pil = output_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        # Apply a light sharpening pass to crisp up the downscaled result
        output_pil = output_pil.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=1))
        out_w, out_h = output_pil.size
        expected_w, expected_h = orig_w, orig_h
    else:
        expected_w = orig_w * target_scale
        expected_h = orig_h * target_scale
        if out_w != expected_w or out_h != expected_h:
            print(f"[RESIZE] Final resizing from {out_w}x{out_h} to exact target {expected_w}x{expected_h}...")
            output_pil = output_pil.resize((expected_w, expected_h), Image.Resampling.LANCZOS)
            out_w, out_h = output_pil.size
            
    # --- 7.5. Generative Texture Refinement (Stable Diffusion 1.5 + ControlNet Tile) ---
    sd_refine = options.get('sd_refine', False)
    sd_refine_pre = sd_refine and (options.get('sd_refine_stage', 'post') == 'pre')
    if sd_refine and not sd_refine_pre and not options.get('_sd_already_run', False):
        # Clear the upscaler model (spandrel_model) from GPU memory to make room for SD
        print("[GPU] Unloading upscaler model to allocate VRAM for Stable Diffusion...")
        _spandrel_cache.clear()
        _upsampler_cache.clear()
        if 'spandrel_model' in locals():
            del spandrel_model
        if 'upsampler' in locals():
            del upsampler
        import gc; gc.collect()
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            
        _current_progress.update({'stage': 'SD texture refinement...', 'percent': 80})
        print(f"[RUN] Applying Stable Diffusion 1.5 + ControlNet Tile texture refinement to {out_w}x{out_h} canvas...")
        try:
            # Convert back to numpy for SD processing
            output_np = np.array(output_pil)[:, :, ::-1].copy()
            
            prompt = options.get('sd_prompt', 'raw photo, highly detailed, sharp focus, 8k, realistic textures, dslr, 35mm lens, film grain')
            neg_prompt = options.get('sd_neg_prompt', 'airbrushed, plastic, waxy, CGI, 3D, render, digital art, smooth skin, oily skin, blurry, low quality, cartoon, painting, drawing, illustration')
            strength = options.get('sd_strength', 20) / 100.0
            mode = options.get('sd_mode', 'hybrid')
            tile_size_opt = options.get('sd_tile_size', 0)
            
            controlnet_scale = options.get('sd_controlnet_strength', 0.80)
            detail_boost = options.get('sd_detail_boost', 1.30)
            detail_threshold = options.get('sd_detail_threshold', 50.0)
            
            output_np = apply_sd_controlnet_refine(
                output_np, 
                prompt, 
                neg_prompt, 
                denoising_strength=strength, 
                device_name=device_name, 
                mode=mode,
                tile_size_override=tile_size_opt,
                controlnet_scale=controlnet_scale,
                detail_boost=detail_boost,
                detail_threshold=detail_threshold,
                sd_refine_stage='post'
            )
            print("[OK] Generative texture refinement complete.")
            # Convert back to PIL
            output_pil = Image.fromarray(output_np[:, :, ::-1])
        except Exception as e:
            print(f"[WARNING] Generative texture refinement failed: {e}")
            traceback.print_exc()
        
    # --- 8. Encode Output ---
    # Use JPEG for large images to prevent huge CPU encoding times and massive base64 payloads
    use_jpeg = (out_w * out_h >= 4000 * 4000)
    if use_jpeg:
        print(f"[RUN] Encoding final {out_w}x{out_h} image as JPEG (quality=95, optimize=False)...")
        buf = io.BytesIO()
        output_pil.save(buf, format='JPEG', quality=95, optimize=False)
        mime_type = 'image/jpeg'
    else:
        print(f"[RUN] Encoding final {out_w}x{out_h} image as PNG (compress_level=3)...")
        buf = io.BytesIO()
        output_pil.save(buf, format='PNG', compress_level=3)
        mime_type = 'image/png'
        
    result_bytes = buf.getvalue()
    print("[RUN] Encoding to base64 string...")
    result_b64 = base64.b64encode(result_bytes).decode()
    
    # Save output image for debugging/verification
    try:
        output_save_path = os.path.join(DIR, "last_output.png")
        output_pil.save(output_save_path, format="PNG")
        print(f"[DEBUG] Saved last output image to {output_save_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save debug output image: {e}")
        
    elapsed = time.time() - start
    print(f"[DONE] {out_w}x{out_h} in {elapsed:.1f}s ({len(result_bytes)/1024/1024:.1f} MB)")
    
    return {
        'success': True,
        'image': f'data:{mime_type};base64,{result_b64}',
        'time': round(elapsed, 1),
        'model': model_name,
        'scale': target_scale,
        'output_w': out_w,
        'output_h': out_h,
        'detected_type': detected_type,
        'auto_settings': {
            'sharpening': options.get('sharpening', 50),
            'detail': options.get('detail', 50),
            'contrast': options.get('contrast', 50),
            'colorBoost': options.get('colorBoost', 30),
            'denoise': options.get('denoise', 12),
            'grain': options.get('grain', 12)
        } if options.get('preset') == 'auto' else None
    }


# ======================== HTTP HANDLER ========================================

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DIR, **kw)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_POST(self):
        if self.path == '/api/upscale':
            self._handle_upscale()
        elif self.path == '/api/models':
            self._handle_models()
        elif self.path == '/api/cancel':
            self._handle_cancel()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == '/api/models':
            self._handle_models()
        elif self.path == '/api/health':
            self._send_json({'status': 'ok', 'cuda': _torch is not None and _torch.cuda.is_available()})
        elif self.path == '/api/progress':
            self._handle_progress()
        else:
            super().do_GET()

    def _handle_progress(self):
        global _current_progress
        self._send_json(_current_progress)

    def _handle_cancel(self):
        global _should_abort
        _should_abort = True
        print("[SERVER] Received user abort request via API.")
        self._send_json({'success': True, 'message': 'Abort signal sent.'})

    def _handle_models(self):
        """Return available models list."""
        models = []
        for name, info in MODEL_REGISTRY.items():
            models.append({
                'id': name,
                'desc': info['desc'],
                'scale': info['scale'],
            })
        self._send_json({'models': models})

    def _handle_upscale(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            
            # Decode base64 image
            img_b64 = body['image']
            if ',' in img_b64:
                img_b64 = img_b64.split(',', 1)[1]
            img_bytes = base64.b64decode(img_b64)
            
            model_name = body.get('model', DEFAULT_MODEL)
            scale = int(body.get('scale', 4))
            
            options = {
                'deblur': body.get('deblur', 'none'),
                'double_pass': body.get('double_pass', False),
                'face_restore': body.get('face_restore', False),
                'post_sharpen': body.get('post_sharpen', 'light'),
                'sharpening': body.get('sharpening', 50),
                'detail': body.get('detail', 50),
                'contrast': body.get('contrast', 50),
                'colorBoost': body.get('colorBoost', 30),
                'denoise': body.get('denoise', 12),
                'grain': body.get('grain', 12),
                'preset': body.get('preset', 'auto'),
                'deblock': body.get('deblock', False),
                'cinematic': body.get('cinematic', 0),
                'device': body.get('device', 'cuda'),
                'replicate_api_key': body.get('replicate_api_key', ''),
                'color_match': body.get('color_match', False),
                'sd_refine': body.get('sd_refine', True),
                'sd_prompt': body.get('sd_prompt') or 'raw photo, highly detailed, sharp focus, 8k, skin pores, fabric texture, realistic skin texture, dslr, 35mm lens, film grain',
                'sd_neg_prompt': body.get('sd_neg_prompt') or 'airbrushed, plastic, waxy, CGI, 3D, render, digital art, smooth skin, oily skin, blurry, low quality, cartoon, painting, drawing, illustration',
                'sd_strength': body.get('sd_strength', 30),
                'sd_tile_size': int(body.get('sd_tile_size', 0)),
                'sd_mode': body.get('sd_mode', 'hybrid'),
                'sd_controlnet_strength': float(body.get('sd_controlnet_strength', 0.70)),
                'sd_detail_boost': float(body.get('sd_detail_boost', 1.50)),
                'sd_refine_stage': body.get('sd_refine_stage', 'post'),
                'sd_detail_threshold': float(body.get('sd_detail_threshold', 50.0))
            }
            
            # Map frontend model names to registry keys
            model_map = {
                'auto': 'auto',
                'ultrasharp-4x': 'realesrgan-x4plus',
                'realesrgan-x4plus': 'realesrgan-x4plus',
                'remacri-4x': 'realesrgan-x4plus',
                'digital-art-4x': 'realesrgan-x4plus',
                'realesrgan-x4plus-anime': 'realesrgan-x4plus-anime',
                'upscayl-standard-4x': 'realesr-general-x4v3',
                'realesr-animevideov3-x3': 'realesr-animevideov3',
                'realesrnet-x4plus': 'realesrnet-x4plus',
                'realesrgan-x2plus': 'realesrgan-x2plus',
                'realesr-general-x4v3': 'realesr-general-x4v3',
                'hat-sharper': 'hat-sharper',
                'hat-imagenet': 'hat-imagenet',
                'cloud-supir': 'cloud-supir',
            }
            model_name = model_map.get(model_name, DEFAULT_MODEL)
            
            try:
                with _upscale_lock:
                    result = upscale_image(img_bytes, model_name, scale, options)
                self._send_json(result)
            finally:
                global _current_progress
                _current_progress.update({
                    'active': False,
                    'stage': 'Idle',
                    'percent': 0
                })
            
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            # Provide user-friendly error messages for common failures
            if 'not enough memory' in error_msg or 'DefaultCPUAllocator' in error_msg:
                error_msg = ('Out of memory! The image is too large for the available RAM. '
                             'Try a smaller image, lower scale (2x instead of 4x), or '
                             'disable De-Blockify and Double Pass options.')
            elif 'CUDA out of memory' in error_msg:
                error_msg = ('GPU ran out of VRAM! Try switching to CPU mode, '
                             'using a smaller scale, or a compact model.')
            self._send_json({'success': False, 'error': error_msg}, status=500)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, fmt, *args):
        try:
            req_str = str(args[0]) if args else ''
            # Log all API requests, or errors which format with 'code %d'
            if '/api/' in req_str or 'code ' in fmt:
                super().log_message(fmt, *args)
        except Exception:
            pass


# ======================== STARTUP =============================================

if __name__ == '__main__':
    print("=" * 60)
    print("  SuperAI Upscaler - Professional GPU Server")
    print("=" * 60)
    
    print(f"\n[INIT] Skipping pre-loading default model to speed up server boot...")
    print("[INFO] AI Models will be lazy-loaded on the first upscaling request.")
    print(f"\n{'=' * 60}")
    print(f"  Server running at http://localhost:{PORT}")
    print(f"  Serving from: {DIR}")
    print(f"{'=' * 60}\n")
    
    from socketserver import ThreadingMixIn
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        
    ThreadedHTTPServer(('', PORT), Handler).serve_forever()
