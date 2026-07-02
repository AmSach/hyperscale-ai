/* SuperAI Poster Studio — Core Logic & Real-time Texture Lab */
(function () {
  'use strict';

  // --- DOM Elements ---
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const fileInfo = document.getElementById('fileInfo');
  const btnEnhance = document.getElementById('btnEnhance');
  
  // Toggles and Selectors
  const doublePassToggle = document.getElementById('doublePassToggle');
  
  const stateEmpty = document.getElementById('stateEmpty');
  const stateProcessing = document.getElementById('stateProcessing');
  const stateResult = document.getElementById('stateResult');
  const wsHeader = document.getElementById('wsHeader');

  const progressBar = document.getElementById('progressBar');
  const progressPct = document.getElementById('progressPct');
  const procTitle = document.getElementById('procTitle');
  const procDetail = document.getElementById('procDetail');
  const timerDisplay = document.getElementById('timerDisplay');

  // Result Area
  const compBox = document.getElementById('compBox');
  const compWrapper = document.getElementById('compWrapper');
  const imgBefore = document.getElementById('imgBefore');
  const imgAfter = document.getElementById('imgAfter');
  const compClip = document.getElementById('compClip');
  const compLine = document.getElementById('compLine');
  const compHandle = document.getElementById('compHandle');
  const resBefore = document.getElementById('resBefore');
  const resAfter = document.getElementById('resAfter');
  const resTime = document.getElementById('resTime');
  const btnDownload = document.getElementById('btnDownload');
  const btnNew = document.getElementById('btnNew');
  
  // Canvas and Toast
  const outCanvas = document.getElementById('outCanvas');
  const toast = document.getElementById('toast');
  const toastText = document.getElementById('toastText');
  const tuningStatus = document.getElementById('tuningStatus');

  // Sliders & Values
  const grainSlider = document.getElementById('grainSlider');
  const grainVal = document.getElementById('grainVal');
  const sharpSlider = document.getElementById('sharpSlider');
  const sharpVal = document.getElementById('sharpVal');
  const contrastSlider = document.getElementById('contrastSlider');
  const contrastVal = document.getElementById('contrastVal');
  const saturationSlider = document.getElementById('saturationSlider');
  const saturationVal = document.getElementById('saturationVal');

  // --- State Variables ---
  let selectedFile = null;
  let originalURL = null;
  let enhancedURL = null;
  let origDim = { w: 0, h: 0 };
  let enhDim = { w: 0, h: 0 };
  
  let currentScale = 4;
  let currentModel = 'ultrasharp-4x';
  let currentDeblur = 'none';
  
  let zoom = 1;
  let baseScale = 1;
  let pan = { x: 0, y: 0 };
  let splitPct = 50;
  let processing = false;

  // Real-time rendering state
  let rawEnhancedImg = null; // Unmodified output from AI
  let tuningTimeout = null;

  // --- Pre-calculated 256x256 Fast Noise Buffer for Micro-Texture ---
  const noiseTableSize = 256 * 256;
  const noiseTable = new Float32Array(noiseTableSize);
  for (let i = 0; i < noiseTableSize; i++) {
    // Standard normal distribution approximation
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    noiseTable[i] = num / 3.0; // scale down slightly
  }

  // --- Timer Web Worker ---
  const timerWorkerCode = `
    let start = 0, iv = null;
    self.onmessage = e => {
      if (e.data === 'start') { start = Date.now(); iv = setInterval(() => self.postMessage((Date.now()-start)/1000), 100); }
      else if (e.data === 'stop') { clearInterval(iv); iv = null; }
    };
  `;
  let timerWorker;
  try {
    timerWorker = new Worker(URL.createObjectURL(new Blob([timerWorkerCode], { type: 'application/javascript' })));
    timerWorker.onmessage = e => { timerDisplay.textContent = e.data.toFixed(1) + 's'; };
  } catch (_) {
    timerWorker = { postMessage: () => {} };
  }

  // --- Toast notifications ---
  function showToast(msg, type) {
    toastText.textContent = msg;
    toast.className = 'toast show ' + (type || '');
    setTimeout(() => toast.className = 'toast', 4000);
  }

  function setState(s) {
    stateEmpty.classList.toggle('active', s === 'empty');
    stateProcessing.classList.toggle('active', s === 'processing');
    stateResult.classList.toggle('active', s === 'result');
  }

  function setProgress(pct) {
    const circumference = 2 * Math.PI * 60; // r=60
    progressBar.style.strokeDasharray = circumference;
    progressBar.style.strokeDashoffset = circumference - (pct / 100) * circumference;
    progressPct.textContent = Math.round(pct) + '%';
  }

  function formatBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
    return (b / 1048576).toFixed(1) + ' MB';
  }

  // --- File Upload Handling ---
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      showToast('Please select a valid image file', 'error');
      return;
    }
    selectedFile = file;
    if (originalURL) URL.revokeObjectURL(originalURL);
    originalURL = URL.createObjectURL(file);

    const img = new Image();
    img.onload = () => {
      origDim = { w: img.naturalWidth, h: img.naturalHeight };
      dropZone.classList.add('has-file');
      dropZone.querySelector('.dz-title').textContent = file.name;
      dropZone.querySelector('.dz-sub').textContent = `${origDim.w} × ${origDim.h} • ${formatBytes(file.size)}`;
      fileInfo.style.display = 'block';
      fileInfo.innerHTML = `<strong>${file.name}</strong> — ${origDim.w}×${origDim.h} → <strong style="color:var(--accent)">${origDim.w * currentScale}×${origDim.h * currentScale}</strong>`;
      btnEnhance.disabled = false;
    };
    img.src = originalURL;
  }

  // --- Selection handlers (Option Buttons Grid) ---
  document.querySelectorAll('.scale-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.scale-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentScale = parseInt(btn.dataset.scale);
      if (origDim.w) {
        fileInfo.innerHTML = `<strong>${selectedFile.name}</strong> — ${origDim.w}×${origDim.h} → <strong style="color:var(--accent)">${origDim.w * currentScale}×${origDim.h * currentScale}</strong>`;
      }
    });
  });

  document.querySelectorAll('.model-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentModel = btn.dataset.model;
    });
  });

  document.querySelectorAll('.deblur-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.deblur-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentDeblur = btn.dataset.deblur;
    });
  });

  // --- Real-time Slider Value Handlers ---
  function updateSliderDisplay() {
    grainVal.textContent = (parseFloat(grainSlider.value) / 10).toFixed(1) + '%';
    sharpVal.textContent = sharpSlider.value + '%';
    contrastVal.textContent = (parseFloat(contrastSlider.value) / 100).toFixed(2);
    saturationVal.textContent = (parseFloat(saturationSlider.value) / 100).toFixed(2);
  }

  [grainSlider, sharpSlider, contrastSlider, saturationSlider].forEach(slider => {
    slider.addEventListener('input', () => {
      updateSliderDisplay();
      triggerTuning();
    });
  });

  function triggerTuning() {
    if (!rawEnhancedImg) return;
    tuningStatus.textContent = 'Updating...';
    tuningStatus.style.color = 'var(--warn)';
    
    if (tuningTimeout) clearTimeout(tuningTimeout);
    tuningTimeout = setTimeout(applyTuning, 80); // Debounce to prevent UI lag on large canvases
  }

  // --- Web App Backend Communication ---
  btnEnhance.addEventListener('click', startEnhance);

  async function startEnhance() {
    if (processing || !selectedFile) return;
    processing = true;
    btnEnhance.disabled = true;

    setState('processing');
    setProgress(5);
    procTitle.textContent = 'Preparing Image...';
    procDetail.textContent = 'Encoding photo to base64 buffer...';
    timerWorker.postMessage('start');

    try {
      const base64Data = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(selectedFile);
      });

      setProgress(15);
      procTitle.textContent = 'Initializing Neural Stacks...';
      procDetail.textContent = 'Routing computations to Nvidia discrete GPU...';

      let currentProgress = 15;
      const progressInterval = setInterval(() => {
        if (currentProgress < 88) {
          currentProgress += (88 - currentProgress) * 0.04;
          setProgress(currentProgress);
        }
      }, 350);

      const response = await fetch('/api/upscale', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: base64Data,
          model: currentModel,
          preset: currentPreset,
          scale: currentScale,
          deblur: currentDeblur,
          double_pass: doublePassToggle.checked,
          face_restore: document.getElementById('faceRestoreCheckbox') ? document.getElementById('faceRestoreCheckbox').checked : true,
          sd_refine: document.getElementById('sdRefineCheckbox') ? document.getElementById('sdRefineCheckbox').checked : false,
          sd_refine_stage: document.getElementById('sdStageSelect') ? document.getElementById('sdStageSelect').value : 'post',
          sd_mode: document.getElementById('sdModeSelect') ? document.getElementById('sdModeSelect').value : 'hybrid',
          sd_strength: document.getElementById('sdStrengthSlider') ? parseInt(document.getElementById('sdStrengthSlider').value, 10) : 30,
          sd_controlnet_strength: document.getElementById('sdControlNetSlider') ? parseFloat(document.getElementById('sdControlNetSlider').value) : 0.70,
          sd_detail_boost: document.getElementById('sdDetailBoostSlider') ? parseFloat(document.getElementById('sdDetailBoostSlider').value) : 1.50,
          sd_tile_size: document.getElementById('sdTileSizeSelect') ? parseInt(document.getElementById('sdTileSizeSelect').value, 10) : 0,
          sd_detail_threshold: document.getElementById('sdDetailThresholdSlider') ? parseFloat(document.getElementById('sdDetailThresholdSlider').value) : 50.0,
          sd_prompt: document.getElementById('sdPromptInput') ? document.getElementById('sdPromptInput').value : '',
          sd_neg_prompt: document.getElementById('sdNegPromptInput') ? document.getElementById('sdNegPromptInput').value : '',
          sharpening: document.getElementById('sharpSlider') ? parseInt(document.getElementById('sharpSlider').value, 10) : 0,
          detail: 0, 
          contrast: document.getElementById('contrastSlider') ? parseInt(document.getElementById('contrastSlider').value, 10) : 0,
          colorBoost: document.getElementById('saturationSlider') ? parseInt(document.getElementById('saturationSlider').value, 10) : 0,
          grain: document.getElementById('grainSlider') ? parseInt(document.getElementById('grainSlider').value, 10) : 0
        })
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server failed with status ${response.status}`);
      }

      setProgress(88);
      procTitle.textContent = 'Decoding AI Output...';
      procDetail.textContent = 'Reconstructing multi-megapixel canvas...';

      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || 'Upscaling processing failed');
      }

      // Load resulting base64 to image element
      const resultImg = new Image();
      await new Promise((resolve, reject) => {
        resultImg.onload = resolve;
        resultImg.onerror = reject;
        resultImg.src = result.image;
      });

      // Cache raw enhanced image in memory for real-time post-processing
      rawEnhancedImg = resultImg;
      enhDim = { w: resultImg.naturalWidth, h: resultImg.naturalHeight };
      
      outCanvas.width = enhDim.w;
      outCanvas.height = enhDim.h;

      setProgress(95);
      procTitle.textContent = 'Applying Poster Syntheses...';
      procDetail.textContent = 'Initializing real-time Texture Lab...';

      // Apply initial tuning
      updateSliderDisplay();
      await applyTuningSync();

      timerWorker.postMessage('stop');
      setProgress(100);
      procTitle.textContent = 'Poster Detailed!';
      
      showResult();
      showToast('Poster Enhanced successfully! 🎉', 'success');

    } catch (err) {
      timerWorker.postMessage('stop');
      console.error(err);
      showToast(err.message, 'error');
      setState('empty');
    } finally {
      processing = false;
      btnEnhance.disabled = !selectedFile;
    }
  }

  // --- Real-time Tuning Algorithms (Ultra-Optimized Canvas2D) ---
  async function applyTuningSync() {
    if (!rawEnhancedImg) return;
    const ctx = outCanvas.getContext('2d');
    const w = outCanvas.width;
    const h = outCanvas.height;

    // Draw raw image
    ctx.drawImage(rawEnhancedImg, 0, 0);

    const grainPercent = (parseFloat(grainSlider.value) / 10) / 100;
    const sharpPercent = parseFloat(sharpSlider.value) / 100;
    const contrastValFloat = parseFloat(contrastSlider.value) / 100;
    const saturationValFloat = parseFloat(saturationSlider.value) / 100;

    // 1. Sharpness pass
    if (sharpPercent > 0) {
      // 3x3 Sharpen Kernel
      const edge = -sharpPercent;
      const center = 1 + 4 * sharpPercent;
      const kernel = [
        0, edge, 0,
        edge, center, edge,
        0, edge, 0
      ];
      applyConvolution(ctx, w, h, kernel);
    }

    // 2. Color & Grain pass
    applyPixelLab(ctx, w, h, contrastValFloat, saturationValFloat, grainPercent);
    
    // Update preview link
    if (enhancedURL) URL.revokeObjectURL(enhancedURL);
    
    // Create preview JPEG blob (much faster than PNG)
    const blob = await new Promise(r => outCanvas.toBlob(r, 'image/jpeg', 0.90));
    enhancedURL = URL.createObjectURL(blob);
    imgAfter.src = enhancedURL;
    
    tuningStatus.textContent = 'Synchronized';
    tuningStatus.style.color = 'var(--success)';
  }

  function applyTuning() {
    applyTuningSync().catch(err => {
      console.error('Tuning error:', err);
      tuningStatus.textContent = 'Error';
      tuningStatus.style.color = 'var(--danger)';
    });
  }

  // --- Fast 3x3 Convolution (Tiled for CPU efficiency) ---
  function applyConvolution(ctx, w, h, kernel) {
    const tileSize = 1536;
    for (let ty = 0; ty < h; ty += tileSize) {
      for (let tx = 0; tx < w; tx += tileSize) {
        const tw = Math.min(tileSize, w - tx);
        const th = Math.min(tileSize, h - ty);
        const imageData = ctx.getImageData(tx, ty, tw, th);
        
        const { data, width, height } = imageData;
        const copy = new Uint8ClampedArray(data);
        
        for (let y = 1; y < height - 1; y++) {
          const rowOffset = y * width;
          const prevRowOffset = (y - 1) * width;
          const nextRowOffset = (y + 1) * width;
          
          for (let x = 1; x < width - 1; x++) {
            const idx = (rowOffset + x) * 4;
            
            // Red Channel
            let r = kernel[0]*copy[(prevRowOffset+x-1)*4] + kernel[1]*copy[(prevRowOffset+x)*4] + kernel[2]*copy[(prevRowOffset+x+1)*4] +
                    kernel[3]*copy[(rowOffset+x-1)*4]     + kernel[4]*copy[idx]                    + kernel[5]*copy[(rowOffset+x+1)*4] +
                    kernel[6]*copy[(nextRowOffset+x-1)*4] + kernel[7]*copy[(nextRowOffset+x)*4] + kernel[8]*copy[(nextRowOffset+x+1)*4];
            
            // Green Channel
            let g = kernel[0]*copy[(prevRowOffset+x-1)*4+1] + kernel[1]*copy[(prevRowOffset+x)*4+1] + kernel[2]*copy[(prevRowOffset+x+1)*4+1] +
                    kernel[3]*copy[(rowOffset+x-1)*4+1]     + kernel[4]*copy[idx+1]                    + kernel[5]*copy[(rowOffset+x+1)*4+1] +
                    kernel[6]*copy[(nextRowOffset+x-1)*4+1] + kernel[7]*copy[(nextRowOffset+x)*4+1] + kernel[8]*copy[(nextRowOffset+x+1)*4+1];
            
            // Blue Channel
            let b = kernel[0]*copy[(prevRowOffset+x-1)*4+2] + kernel[1]*copy[(prevRowOffset+x)*4+2] + kernel[2]*copy[(prevRowOffset+x+1)*4+2] +
                    kernel[3]*copy[(rowOffset+x-1)*4+2]     + kernel[4]*copy[idx+2]                    + kernel[5]*copy[(rowOffset+x+1)*4+2] +
                    kernel[6]*copy[(nextRowOffset+x-1)*4+2] + kernel[7]*copy[(nextRowOffset+x)*4+2] + kernel[8]*copy[(nextRowOffset+x+1)*4+2];
            
            data[idx] = r < 0 ? 0 : (r > 255 ? 255 : r);
            data[idx+1] = g < 0 ? 0 : (g > 255 ? 255 : g);
            data[idx+2] = b < 0 ? 0 : (b > 255 ? 255 : b);
          }
        }
        ctx.putImageData(imageData, tx, ty);
      }
    }
  }

  // --- Color Tuning & Tiled Noise Synthesis (Highly Optimized) ---
  function applyPixelLab(ctx, w, h, contrast, saturation, grain) {
    const tileSize = 1536;
    let noiseIdx = 0;
    
    for (let ty = 0; ty < h; ty += tileSize) {
      for (let tx = 0; tx < w; tx += tileSize) {
        const tw = Math.min(tileSize, w - tx);
        const th = Math.min(tileSize, h - ty);
        const imageData = ctx.getImageData(tx, ty, tw, th);
        const data = imageData.data;
        const len = data.length;

        for (let i = 0; i < len; i += 4) {
          let r = data[i];
          let g = data[i+1];
          let b = data[i+2];

          // Contrast curve (S-Curve centering)
          if (contrast !== 1.0) {
            r = contrast * (r - 127) + 127;
            g = contrast * (g - 127) + 127;
            b = contrast * (b - 127) + 127;
          }

          // Saturation (Luminance preserving)
          if (saturation !== 1.0) {
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;
            r = gray + saturation * (r - gray);
            g = gray + saturation * (g - gray);
            b = gray + saturation * (b - gray);
          }

          // Micro-texture (Film Grain injection via bitwise tiled buffer)
          if (grain > 0) {
            const noise = noiseTable[noiseIdx] * grain * 255;
            r += noise;
            g += noise;
            b += noise;
            
            // Circular increment & wrapping with fast bitwise operator
            noiseIdx = (noiseIdx + 1) & 65535;
          }

          data[i] = r < 0 ? 0 : (r > 255 ? 255 : r);
          data[i+1] = g < 0 ? 0 : (g > 255 ? 255 : g);
          data[i+2] = b < 0 ? 0 : (b > 255 ? 255 : b);
        }
        ctx.putImageData(imageData, tx, ty);
      }
    }
  }

  // --- Results and Split slider UI ---
  function showResult() {
    setState('result');
    wsHeader.style.display = 'none';

    imgBefore.src = originalURL;
    resBefore.textContent = `${origDim.w} × ${origDim.h}`;
    resAfter.textContent = `${enhDim.w} × ${enhDim.h}`;
    resTime.textContent = timerDisplay.textContent;

    imgAfter.onload = () => {
      setupComparison();
      resetView();
    };
  }

  function setupComparison() {
    compWrapper.style.width = enhDim.w + 'px';
    compWrapper.style.height = enhDim.h + 'px';

    imgBefore.style.width = enhDim.w + 'px';
    imgBefore.style.height = enhDim.h + 'px';

    compClip.style.width = '50%';
    imgAfter.style.width = enhDim.w + 'px';
    imgAfter.style.height = enhDim.h + 'px';
  }

  function resetView() {
    const boxW = compBox.clientWidth || 800;
    const boxH = compBox.clientHeight || 450;
    if (enhDim.w && enhDim.h) {
      baseScale = Math.min(boxW / enhDim.w, boxH / enhDim.h);
    } else {
      baseScale = 1;
    }
    zoom = 1;
    pan = { x: 0, y: 0 };
    splitPct = 50;
    updateView();
    updateSplit();
  }

  function updateView() {
    const s = baseScale * zoom;
    compWrapper.style.transform = `translate(-50%,-50%) translate(${pan.x}px,${pan.y}px) scale(${s})`;
  }

  function updateSplit() {
    const pct = Math.max(0, Math.min(100, splitPct));
    compClip.style.width = (100 - pct) + '%';
    imgAfter.style.width = enhDim.w + 'px';

    const boxRect = compBox.getBoundingClientRect();
    const wrapperRect = compWrapper.getBoundingClientRect();
    const lineX = wrapperRect.left + wrapperRect.width * (pct / 100) - boxRect.left;

    compLine.style.left = lineX + 'px';
    compHandle.style.left = lineX + 'px';
  }

  // --- Split Handle mouse & touch interactions ---
  let draggingSlider = false;
  let draggingPan = false;
  let panStart = { x: 0, y: 0 };

  compBox.addEventListener('mousedown', e => {
    if (zoom > 1 && !e.target.closest('.comp-handle')) {
      draggingPan = true;
      panStart = { x: e.clientX - pan.x, y: e.clientY - pan.y };
      compBox.style.cursor = 'grabbing';
    } else {
      draggingSlider = true;
      moveSplit(e);
    }
  });

  compHandle.addEventListener('mousedown', e => {
    e.stopPropagation();
    draggingSlider = true;
  });

  window.addEventListener('mousemove', e => {
    if (draggingSlider) moveSplit(e);
    if (draggingPan) {
      pan.x = e.clientX - panStart.x;
      pan.y = e.clientY - panStart.y;
      updateView();
      updateSplit();
    }
  });

  window.addEventListener('mouseup', () => {
    draggingSlider = false;
    draggingPan = false;
    compBox.style.cursor = zoom > 1 ? 'grab' : 'ew-resize';
  });

  compBox.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
      draggingSlider = true;
      moveSplit(e.touches[0]);
    }
  }, { passive: true });
  
  compBox.addEventListener('touchmove', e => {
    if (draggingSlider && e.touches.length === 1) moveSplit(e.touches[0]);
  }, { passive: true });
  
  window.addEventListener('touchend', () => { draggingSlider = false; });

  function moveSplit(e) {
    const wrapperRect = compWrapper.getBoundingClientRect();
    if (wrapperRect.width === 0) return;
    const x = (e.clientX || e.pageX) - wrapperRect.left;
    splitPct = Math.max(0, Math.min(100, (x / wrapperRect.width) * 100));
    updateSplit();
  }

  // --- Zoom Buttons ---
  document.getElementById('zoomIn').addEventListener('click', () => {
    if (zoom >= 10) return;
    zoom = Math.min(10, zoom + 0.5);
    updateView(); updateSplit();
  });
  
  document.getElementById('zoomOut').addEventListener('click', () => {
    zoom = Math.max(1, zoom - 0.5);
    if (zoom === 1) pan = { x: 0, y: 0 };
    updateView(); updateSplit();
  });
  
  document.getElementById('zoomReset').addEventListener('click', resetView);

  compBox.addEventListener('wheel', e => {
    e.preventDefault();
    if (e.deltaY < 0) zoom = Math.min(10, zoom + 0.3);
    else zoom = Math.max(1, zoom - 0.3);
    if (zoom <= 1) { zoom = 1; pan = { x: 0, y: 0 }; }
    updateView(); updateSplit();
  }, { passive: false });

  window.addEventListener('resize', () => {
    if (stateResult.classList.contains('active')) resetView();
  });

  // --- Download (PNG format at full print-quality) ---
  btnDownload.addEventListener('click', () => {
    if (!rawEnhancedImg) return;
    showToast('Rendering final print PNG...', 'info');
    
    // We encode to PNG format for lossy-free print reproduction
    outCanvas.toBlob(blob => {
      const a = document.createElement('a');
      const url = URL.createObjectURL(blob);
      a.href = url;
      const name = selectedFile ? selectedFile.name.replace(/\.[^.]+$/, '') : 'poster';
      a.download = `${name}_poster_${currentScale}x.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast('Poster print saved to Downloads! 🖨️', 'success');
    }, 'image/png');
  });

  // --- Reset UI ---
  btnNew.addEventListener('click', () => {
    selectedFile = null;
    rawEnhancedImg = null;
    if (originalURL) { URL.revokeObjectURL(originalURL); originalURL = null; }
    if (enhancedURL) { URL.revokeObjectURL(enhancedURL); enhancedURL = null; }
    origDim = { w: 0, h: 0 };
    enhDim = { w: 0, h: 0 };
    fileInput.value = '';
    dropZone.classList.remove('has-file');
    dropZone.querySelector('.dz-title').textContent = 'Drag your photo here';
    dropZone.querySelector('.dz-sub').textContent = 'or click to browse • JPG, PNG, WebP';
    fileInfo.style.display = 'none';
    btnEnhance.disabled = true;
    wsHeader.style.display = '';
    setState('empty');
  });

})();
