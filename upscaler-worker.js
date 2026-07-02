'use strict';

// ============================================================================
// SuperAI Upscaler — Professional Algorithmic Image Upscaling Engine
// Runs entirely inside a Web Worker. No AI, no dependencies.
// ============================================================================

const PI = Math.PI;

// ======================== MATH UTILITIES ====================================

function sinc(x) {
  if (x === 0) return 1;
  const px = PI * x;
  return Math.sin(px) / px;
}

function lanczosKernel(x, a) {
  if (x === 0) return 1;
  if (Math.abs(x) >= a) return 0;
  return sinc(x) * sinc(x / a);
}

function clampF(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function clampByte(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// ======================== PRESETS ============================================

const PRESETS = {
  poster:  { sharpening: 65, detail: 70, contrast: 60, colorBoost: 40, denoise: 15 },
  web:     { sharpening: 40, detail: 40, contrast: 40, colorBoost: 25, denoise: 25 },
  photo:   { sharpening: 50, detail: 55, contrast: 45, colorBoost: 30, denoise: 20 },
  maximum: { sharpening: 80, detail: 85, contrast: 70, colorBoost: 50, denoise: 10 }
};

// ======================== PROGRESS REPORTING =================================

let totalStages = 8;
let currentStageIndex = 0;

function sendProgress(stage, percent) {
  self.postMessage({
    type: 'progress',
    stage: stage,
    percent: Math.round(percent),
    stageIndex: currentStageIndex,
    totalStages: totalStages
  });
}

// ======================== SEPARABLE LANCZOS-3 RESAMPLING ====================
// Two-pass (horizontal then vertical) sinc-based interpolation.
// Lanczos-3 uses a 3-lobe windowed sinc — the gold standard for
// traditional image resampling, superior to bicubic.

function separableLanczos3(src, srcW, srcH, dstW, dstH, progressLabel) {
  const A = 3; // Lanczos lobes
  const channels = 4; // RGBA

  // --- Horizontal pass: srcW×srcH → dstW×srcH ---
  const midW = dstW;
  const midH = srcH;
  const mid = new Float32Array(midW * midH * channels);

  const xRatio = srcW / dstW;
  for (let y = 0; y < midH; y++) {
    if (y % 64 === 0) sendProgress(progressLabel || 'Upscaling (horizontal)...', (y / midH) * 45);
    for (let x = 0; x < midW; x++) {
      const srcX = (x + 0.5) * xRatio - 0.5;
      const iStart = Math.ceil(srcX - A);
      const iEnd = Math.floor(srcX + A);

      let sumR = 0, sumG = 0, sumB = 0, sumA = 0, sumW = 0;
      for (let i = iStart; i <= iEnd; i++) {
        const si = clampF(i, 0, srcW - 1) | 0;
        const w = lanczosKernel(srcX - i, A);
        const idx = (y * srcW + si) * channels;
        sumR += src[idx]     * w;
        sumG += src[idx + 1] * w;
        sumB += src[idx + 2] * w;
        sumA += src[idx + 3] * w;
        sumW += w;
      }
      if (sumW !== 0) {
        const inv = 1 / sumW;
        sumR *= inv; sumG *= inv; sumB *= inv; sumA *= inv;
      }
      const dIdx = (y * midW + x) * channels;
      mid[dIdx]     = sumR;
      mid[dIdx + 1] = sumG;
      mid[dIdx + 2] = sumB;
      mid[dIdx + 3] = sumA;
    }
  }

  // --- Vertical pass: dstW×srcH → dstW×dstH ---
  const dst = new Float32Array(dstW * dstH * channels);
  const yRatio = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    if (y % 64 === 0) sendProgress(progressLabel || 'Upscaling (vertical)...', 45 + (y / dstH) * 45);
    for (let x = 0; x < dstW; x++) {
      const srcY = (y + 0.5) * yRatio - 0.5;
      const jStart = Math.ceil(srcY - A);
      const jEnd = Math.floor(srcY + A);

      let sumR = 0, sumG = 0, sumB = 0, sumA = 0, sumW = 0;
      for (let j = jStart; j <= jEnd; j++) {
        const sj = clampF(j, 0, srcH - 1) | 0;
        const w = lanczosKernel(srcY - j, A);
        const idx = (sj * midW + x) * channels;
        sumR += mid[idx]     * w;
        sumG += mid[idx + 1] * w;
        sumB += mid[idx + 2] * w;
        sumA += mid[idx + 3] * w;
        sumW += w;
      }
      if (sumW !== 0) {
        const inv = 1 / sumW;
        sumR *= inv; sumG *= inv; sumB *= inv; sumA *= inv;
      }
      const dIdx = (y * dstW + x) * channels;
      dst[dIdx]     = sumR;
      dst[dIdx + 1] = sumG;
      dst[dIdx + 2] = sumB;
      dst[dIdx + 3] = sumA;
    }
  }

  sendProgress(progressLabel || 'Upscaling...', 95);
  return dst;
}

// ======================== SOBEL EDGE DETECTION ==============================
// Full 3×3 Sobel operator computing gradient magnitude and direction
// on BT.601 luminance. Used to drive adaptive processing in later stages.

function sobelEdgeDetect(img, w, h) {
  const ch = 4;
  // Compute grayscale luminance
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * ch;
    lum[i] = 0.299 * img[idx] + 0.587 * img[idx + 1] + 0.114 * img[idx + 2];
  }

  const magnitude = new Float32Array(w * h);
  const direction = new Float32Array(w * h);
  let maxMag = 0;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const tl = lum[(y - 1) * w + (x - 1)];
      const tc = lum[(y - 1) * w + x];
      const tr = lum[(y - 1) * w + (x + 1)];
      const ml = lum[y * w + (x - 1)];
      const mr = lum[y * w + (x + 1)];
      const bl = lum[(y + 1) * w + (x - 1)];
      const bc = lum[(y + 1) * w + x];
      const br = lum[(y + 1) * w + (x + 1)];

      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      const mag = Math.sqrt(gx * gx + gy * gy);
      magnitude[y * w + x] = mag;
      direction[y * w + x] = Math.atan2(gy, gx);
      if (mag > maxMag) maxMag = mag;
    }
  }

  // Normalize magnitude to [0, 1]
  if (maxMag > 0) {
    const inv = 1 / maxMag;
    for (let i = 0; i < magnitude.length; i++) {
      magnitude[i] *= inv;
    }
  }

  return { magnitude, direction };
}

// ======================== EDGE-DIRECTED INTERPOLATION =======================
// After Lanczos upscaling, detect edge direction via Sobel and apply
// correction that blends along edges (preserving sharpness) instead
// of across them (which creates blur).

function edgeDirectedEnhance(img, w, h, strength) {
  const ch = 4;
  const { magnitude, direction } = sobelEdgeDetect(img, w, h);
  const out = new Float32Array(img.length);
  out.set(img);

  const threshold = 0.05;
  // Direction offsets for 4 quantized edge directions
  // Edge at 0°: sample along Y, Edge at 90°: sample along X
  const dirOffsets = [
    { dx: 0, dy: 1 },   // 0° edge → vertical
    { dx: 1, dy: 1 },   // 45° edge → diagonal
    { dx: 1, dy: 0 },   // 90° edge → horizontal
    { dx: 1, dy: -1 }   // 135° edge → anti-diagonal
  ];

  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const idx = y * w + x;
      const mag = magnitude[idx];
      if (mag < threshold) continue;

      // Quantize direction to 4 bins
      let angle = direction[idx];
      if (angle < 0) angle += PI;
      const bin = Math.round(angle / (PI / 4)) % 4;
      const off = dirOffsets[bin];

      // Sample along the edge (parallel)
      const along1 = ((y + off.dy) * w + (x + off.dx)) * ch;
      const along2 = ((y - off.dy) * w + (x - off.dx)) * ch;
      // Sample across the edge (perpendicular)
      const across1 = ((y + off.dx) * w + (x - off.dy)) * ch;
      const across2 = ((y - off.dx) * w + (x + off.dy)) * ch;

      // Bounds check
      if (x + off.dx < 0 || x + off.dx >= w || y + off.dy < 0 || y + off.dy >= h) continue;
      if (x - off.dx < 0 || x - off.dx >= w || y - off.dy < 0 || y - off.dy >= h) continue;
      if (x - off.dy < 0 || x - off.dy >= w || y + off.dx < 0 || y + off.dx >= h) continue;
      if (x + off.dy < 0 || x + off.dy >= w || y - off.dx < 0 || y - off.dx >= h) continue;

      const pixIdx = idx * ch;
      const blendStrength = clampF(mag * strength, 0, 0.5);

      for (let c = 0; c < 3; c++) {
        const current = img[pixIdx + c];
        // Average along the edge
        const alongAvg = (img[along1 + c] + img[along2 + c]) * 0.5;
        // Average across the edge
        const acrossAvg = (img[across1 + c] + img[across2 + c]) * 0.5;
        // Blend: weight along > across to preserve edge
        const edgeVal = alongAvg * 0.7 + acrossAvg * 0.3;
        // Mix with original based on edge strength
        out[pixIdx + c] = current * (1 - blendStrength) + edgeVal * blendStrength;
      }
    }
  }

  return out;
}

// ======================== SEPARABLE GAUSSIAN BLUR ===========================
// Used for Laplacian pyramid, sharpening, and CLAHE. Separable
// implementation: horizontal then vertical, each O(n×k).

function buildGaussianKernel(sigma) {
  const radius = Math.ceil(sigma * 3);
  const size = radius * 2 + 1;
  const kernel = new Float32Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - radius;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += kernel[i];
  }
  // Normalize
  for (let i = 0; i < size; i++) kernel[i] /= sum;
  return { kernel, radius };
}

function gaussianBlur(img, w, h, sigma) {
  const ch = 4;
  const { kernel, radius } = buildGaussianKernel(sigma);
  const size = kernel.length;
  const temp = new Float32Array(w * h * ch);
  const out = new Float32Array(w * h * ch);

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let k = 0; k < size; k++) {
        const sx = clampF(x + k - radius, 0, w - 1) | 0;
        const idx = (y * w + sx) * ch;
        const kw = kernel[k];
        r += img[idx]     * kw;
        g += img[idx + 1] * kw;
        b += img[idx + 2] * kw;
        a += img[idx + 3] * kw;
      }
      const oIdx = (y * w + x) * ch;
      temp[oIdx]     = r;
      temp[oIdx + 1] = g;
      temp[oIdx + 2] = b;
      temp[oIdx + 3] = a;
    }
  }

  // Vertical pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let k = 0; k < size; k++) {
        const sy = clampF(y + k - radius, 0, h - 1) | 0;
        const idx = (sy * w + x) * ch;
        const kw = kernel[k];
        r += temp[idx]     * kw;
        g += temp[idx + 1] * kw;
        b += temp[idx + 2] * kw;
        a += temp[idx + 3] * kw;
      }
      const oIdx = (y * w + x) * ch;
      out[oIdx]     = r;
      out[oIdx + 1] = g;
      out[oIdx + 2] = b;
      out[oIdx + 3] = a;
    }
  }

  return out;
}

// ======================== NOISE ESTIMATION ==================================
// Estimate image noise level from flat regions. Divides image into
// 16×16 blocks, computes variance per block, takes median of lowest
// quartile as the noise floor.

function estimateNoise(img, w, h) {
  const ch = 4;
  const blockSize = 16;
  const bx = Math.floor(w / blockSize);
  const by = Math.floor(h / blockSize);
  if (bx === 0 || by === 0) return 0;

  const variances = [];
  for (let by_ = 0; by_ < by; by_++) {
    for (let bx_ = 0; bx_ < bx; bx_++) {
      let sum = 0, sumSq = 0, count = 0;
      for (let dy = 0; dy < blockSize; dy++) {
        for (let dx = 0; dx < blockSize; dx++) {
          const px = bx_ * blockSize + dx;
          const py = by_ * blockSize + dy;
          const idx = (py * w + px) * ch;
          const lum = 0.299 * img[idx] + 0.587 * img[idx + 1] + 0.114 * img[idx + 2];
          sum += lum;
          sumSq += lum * lum;
          count++;
        }
      }
      const mean = sum / count;
      const variance = sumSq / count - mean * mean;
      variances.push(variance);
    }
  }

  variances.sort((a, b) => a - b);
  const q1End = Math.max(1, Math.floor(variances.length * 0.25));
  let medianIdx = Math.floor(q1End / 2);
  return variances[medianIdx] || 0;
}

// ======================== LIGHT MEDIAN FILTER ===============================
// 3×3 median filter for noise reduction in noisy areas.
// Only applied when denoise is significant.

function medianFilter3x3(img, w, h) {
  const ch = 4;
  const out = new Float32Array(img.length);
  out.set(img);
  const buf = new Float32Array(9);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      for (let c = 0; c < 3; c++) {
        let k = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            buf[k++] = img[((y + dy) * w + (x + dx)) * ch + c];
          }
        }
        // Sort for median (insertion sort on 9 elements)
        for (let i = 1; i < 9; i++) {
          const val = buf[i];
          let j = i - 1;
          while (j >= 0 && buf[j] > val) {
            buf[j + 1] = buf[j];
            j--;
          }
          buf[j + 1] = val;
        }
        out[(y * w + x) * ch + c] = buf[4]; // median
      }
    }
  }
  return out;
}

// ======================== LAPLACIAN PYRAMID DETAIL ENHANCEMENT ===============
// Builds a 4-level Gaussian pyramid, extracts 3 Laplacian (detail) layers,
// amplifies each at a different strength, then reconstructs. Fine detail
// gets the most boost, coarse structure stays stable.

function laplacianPyramidDetail(img, w, h, detailStrength) {
  sendProgress('Injecting detail...', 0);
  const strength = detailStrength / 100;

  // Build Gaussian pyramid
  const g0 = img;
  const g1 = gaussianBlur(g0, w, h, 1.0);
  sendProgress('Injecting detail...', 20);
  const g2 = gaussianBlur(g1, w, h, 2.0);
  sendProgress('Injecting detail...', 40);
  const g3 = gaussianBlur(g2, w, h, 4.0);
  sendProgress('Injecting detail...', 55);

  // Build Laplacian layers and amplify
  const len = w * h * 4;
  const result = new Float32Array(len);

  // Fine detail boost factors
  const fineBoost   = 1.0 + strength * 0.8;
  const medBoost    = 1.0 + strength * 0.5;
  const coarseBoost = 1.0 + strength * 0.3;

  for (let i = 0; i < len; i++) {
    const l0 = g0[i] - g1[i]; // finest detail
    const l1 = g1[i] - g2[i]; // medium detail
    const l2 = g2[i] - g3[i]; // coarse detail
    result[i] = g3[i] + l2 * coarseBoost + l1 * medBoost + l0 * fineBoost;
  }

  sendProgress('Injecting detail...', 100);
  return result;
}

// ======================== FREQUENCY-AWARE ADAPTIVE SHARPENING ================
// Unlike naive unsharp masking which creates halos on hard edges,
// this uses the Sobel edge map to apply different sharpening strengths:
// light on smooth areas, full on textures, reduced on hard edges.

function frequencyAwareSharpen(img, w, h, sharpStrength) {
  sendProgress('Adaptive sharpening...', 0);
  const strength = sharpStrength / 100;
  const ch = 4;

  // Compute edge map
  const { magnitude } = sobelEdgeDetect(img, w, h);
  sendProgress('Adaptive sharpening...', 25);

  // Compute unsharp mask: highPass = original - blurred
  const blurred = gaussianBlur(img, w, h, 0.7);
  sendProgress('Adaptive sharpening...', 60);

  const out = new Float32Array(img.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const edgeMag = magnitude[idx];
      const pixIdx = idx * ch;

      // Adaptive sharpening factor based on local content
      let factor;
      if (edgeMag < 0.1) {
        factor = strength * 0.3; // smooth area — light sharpen
      } else if (edgeMag < 0.5) {
        factor = strength * 1.0; // texture — full sharpen
      } else {
        factor = strength * 0.4; // hard edge — reduce to avoid halos
      }

      for (let c = 0; c < 3; c++) {
        const highPass = img[pixIdx + c] - blurred[pixIdx + c];
        out[pixIdx + c] = img[pixIdx + c] + highPass * factor * 2.0;
      }
      out[pixIdx + 3] = img[pixIdx + 3]; // preserve alpha
    }
  }

  sendProgress('Adaptive sharpening...', 100);
  return out;
}

// ======================== CLAHE =============================================
// Contrast Limited Adaptive Histogram Equalization. Divides image into
// an 8×8 tile grid, clips each tile's histogram (preventing over-
// enhancement), and uses bilinear interpolation between tiles for
// seamless transitions. Processes luminance only to preserve colors.

function clahe(img, w, h, contrastStrength) {
  sendProgress('Local contrast (CLAHE)...', 0);
  const strength = contrastStrength / 100;
  if (strength < 0.01) return img;

  const ch = 4;
  const tilesX = 8;
  const tilesY = 8;
  const tileW = Math.ceil(w / tilesX);
  const tileH = Math.ceil(h / tilesY);
  const bins = 256;

  // Compute luminance
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * ch;
    lum[i] = 0.299 * img[idx] + 0.587 * img[idx + 1] + 0.114 * img[idx + 2];
  }

  // Compute per-tile clipped CDFs
  const tileCDFs = [];
  const clipLimitFactor = 1.0 + strength * 3.0;

  for (let ty = 0; ty < tilesY; ty++) {
    tileCDFs[ty] = [];
    for (let tx = 0; tx < tilesX; tx++) {
      // Compute histogram for this tile
      const hist = new Float64Array(bins);
      let tilePixels = 0;
      const x0 = tx * tileW;
      const y0 = ty * tileH;
      const x1 = Math.min(x0 + tileW, w);
      const y1 = Math.min(y0 + tileH, h);

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const bin = clampF(Math.floor(lum[y * w + x]), 0, 255) | 0;
          hist[bin]++;
          tilePixels++;
        }
      }

      // Clip histogram
      const clipLimit = Math.max(1, (tilePixels / bins) * clipLimitFactor);
      let excess = 0;
      for (let i = 0; i < bins; i++) {
        if (hist[i] > clipLimit) {
          excess += hist[i] - clipLimit;
          hist[i] = clipLimit;
        }
      }
      // Redistribute excess uniformly
      const redistrib = excess / bins;
      for (let i = 0; i < bins; i++) {
        hist[i] += redistrib;
      }

      // Compute CDF
      const cdf = new Float32Array(bins);
      cdf[0] = hist[0];
      for (let i = 1; i < bins; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
      }
      // Normalize to [0, 255]
      const cdfMin = cdf[0];
      const cdfMax = cdf[bins - 1];
      const cdfRange = cdfMax - cdfMin;
      if (cdfRange > 0) {
        for (let i = 0; i < bins; i++) {
          cdf[i] = ((cdf[i] - cdfMin) / cdfRange) * 255;
        }
      }
      tileCDFs[ty][tx] = cdf;
    }
  }

  sendProgress('Local contrast (CLAHE)...', 50);

  // Apply with bilinear interpolation between tiles
  const out = new Float32Array(img.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const pixIdx = idx * ch;
      const oldL = lum[idx];
      const bin = clampF(Math.floor(oldL), 0, 255) | 0;

      // Find tile coordinates
      const txF = (x / tileW) - 0.5;
      const tyF = (y / tileH) - 0.5;
      const tx0 = clampF(Math.floor(txF), 0, tilesX - 1) | 0;
      const ty0 = clampF(Math.floor(tyF), 0, tilesY - 1) | 0;
      const tx1 = clampF(tx0 + 1, 0, tilesX - 1) | 0;
      const ty1 = clampF(ty0 + 1, 0, tilesY - 1) | 0;
      const fx = clampF(txF - tx0, 0, 1);
      const fy = clampF(tyF - ty0, 0, 1);

      // Bilinear interpolation of mapped values
      const v00 = tileCDFs[ty0][tx0][bin];
      const v10 = tileCDFs[ty0][tx1][bin];
      const v01 = tileCDFs[ty1][tx0][bin];
      const v11 = tileCDFs[ty1][tx1][bin];
      const newL = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                   v01 * (1 - fx) * fy + v11 * fx * fy;

      // Blend with original based on strength
      const finalL = oldL + (newL - oldL) * strength;

      // Apply luminance ratio to RGB (preserves color)
      if (oldL > 0.001) {
        const ratio = finalL / oldL;
        out[pixIdx]     = img[pixIdx]     * ratio;
        out[pixIdx + 1] = img[pixIdx + 1] * ratio;
        out[pixIdx + 2] = img[pixIdx + 2] * ratio;
      } else {
        out[pixIdx]     = finalL;
        out[pixIdx + 1] = finalL;
        out[pixIdx + 2] = finalL;
      }
      out[pixIdx + 3] = img[pixIdx + 3];
    }
  }

  sendProgress('Local contrast (CLAHE)...', 100);
  return out;
}

// ======================== PERCEPTUAL COLOR ENHANCEMENT ======================
// Decomposes to luminance + chroma, applies a smooth sigmoid S-curve
// for contrast, and boosts saturation with gamut protection.

function colorEnhance(img, w, h, colorBoost, contrastAmt) {
  sendProgress('Color enhancement...', 0);
  const ch = 4;
  const satFactor = 1.0 + (colorBoost / 100) * 0.4;
  const contrastS = (contrastAmt / 100) * 0.15;
  const out = new Float32Array(img.length);

  for (let i = 0; i < w * h; i++) {
    const idx = i * ch;
    const r = img[idx], g = img[idx + 1], b = img[idx + 2];

    // Compute luminance
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;

    // S-curve on luminance (sigmoid-based for natural look)
    const normalized = clampF(lum / 255, 0, 1);
    // Smooth S-curve: shift midtones, preserve extremes
    const curved = 1.0 / (1.0 + Math.exp(-6 * (normalized - 0.5)));
    const newL = (normalized + (curved - normalized) * contrastS) * 255;

    // Chroma decomposition
    const cr = r - lum;
    const cg = g - lum;
    const cb = b - lum;

    // Saturation boost with gamut protection
    const newR = clampF(newL + cr * satFactor, 0, 255);
    const newG = clampF(newL + cg * satFactor, 0, 255);
    const newB = clampF(newL + cb * satFactor, 0, 255);

    out[idx]     = newR;
    out[idx + 1] = newG;
    out[idx + 2] = newB;
    out[idx + 3] = img[idx + 3];
  }

  sendProgress('Color enhancement...', 100);
  return out;
}

// ======================== ANTI-ARTIFACT PROCESSING ==========================
// 1. Ringing suppression: clamp overshoots near strong edges
// 2. Ordered dithering: subtle Bayer matrix to prevent color banding

function antiArtifact(img, w, h) {
  sendProgress('Final polish...', 0);
  const ch = 4;
  const out = new Float32Array(img.length);
  out.set(img);

  // --- Ringing suppression ---
  // Clamp pixels that overshoot their local 3×3 min/max
  const ringingThreshold = 8.0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const pixIdx = (y * w + x) * ch;
      for (let c = 0; c < 3; c++) {
        let localMin = Infinity, localMax = -Infinity;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nIdx = ((y + dy) * w + (x + dx)) * ch + c;
            const v = img[nIdx];
            if (v < localMin) localMin = v;
            if (v > localMax) localMax = v;
          }
        }
        const val = out[pixIdx + c];
        if (val > localMax + ringingThreshold) {
          out[pixIdx + c] = localMax + ringingThreshold * 0.5;
        } else if (val < localMin - ringingThreshold) {
          out[pixIdx + c] = localMin - ringingThreshold * 0.5;
        }
      }
    }
  }

  sendProgress('Final polish...', 50);

  // --- Ordered dithering (anti-banding) ---
  // Bayer 4×4 matrix for subtle dithering to prevent color banding
  const bayer = [
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
  ];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const pixIdx = (y * w + x) * ch;
      const ditherVal = (bayer[y & 3][x & 3] / 16.0 - 0.5) * 1.0;
      for (let c = 0; c < 3; c++) {
        out[pixIdx + c] += ditherVal;
      }
    }
  }

  sendProgress('Final polish...', 100);
  return out;
}

// ======================== IMAGE CLASSIFICATION ==============================
// Evaluates block variance to classify image as Photo, Graphic (Line Art/Text),
// Flat, or Noisy, allowing the engine to adapt parameters dynamically.

function classifyImage(img, w, h, noiseLevel) {
  const ch = 4;
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * ch;
    lum[i] = 0.299 * img[idx] + 0.587 * img[idx + 1] + 0.114 * img[idx + 2];
  }

  const sampleStep = Math.max(4, Math.floor(Math.sqrt((w * h) / 10000)));
  let flatCount = 0;
  let detailCount = 0;
  let edgeCount = 0;
  let totalSamples = 0;

  for (let y = 2; y < h - 2; y += sampleStep) {
    for (let x = 2; x < w - 2; x += sampleStep) {
      totalSamples++;
      let sum = 0, sumSq = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const val = lum[(y + dy) * w + (x + dx)];
          sum += val;
          sumSq += val * val;
        }
      }
      const mean = sum / 9;
      const variance = sumSq / 9 - mean * mean;

      if (variance < 6.0) {
        flatCount++;
      } else if (variance < 100.0) {
        detailCount++;
      } else {
        edgeCount++;
      }
    }
  }

  const flatRatio = flatCount / (totalSamples || 1);
  const detailRatio = detailCount / (totalSamples || 1);
  const edgeRatio = edgeCount / (totalSamples || 1);

  let type = 'photo';
  if (flatRatio > 0.72 && detailRatio < 0.22) {
    type = 'graphic';
  } else if (noiseLevel > 35) {
    type = 'noisy';
  } else if (flatRatio > 0.92) {
    type = 'flat';
  }

  return { type, flatRatio, detailRatio, edgeRatio, noiseLevel };
}

// ======================== CINEMATIC COLOR GRADING ============================
// Professional Hollywood Teal & Orange split-toning and S-curve grading.

function applyCinematicGrading(img, w, h, strength) {
  sendProgress('Cinematic grading...', 0);
  const ch = 4;
  const out = new Float32Array(img.length);
  const factor = strength / 100;
  
  for (let i = 0; i < w * h; i++) {
    if (i % 8192 === 0) {
      sendProgress('Cinematic grading...', (i / (w * h)) * 100);
    }
    const idx = i * ch;
    const r = img[idx], g = img[idx + 1], b = img[idx + 2];
    
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    const norm = clampF(lum / 255, 0, 1);
    
    // 1. Filmic S-curve with black lift
    const blackLift = 0.04 * factor;
    const normLifted = blackLift + (1.0 - blackLift) * norm;
    const sCurve = normLifted * normLifted * (3.0 - 2.0 * normLifted);
    const targetLum = (norm + (sCurve - norm) * 0.4 * factor) * 255.0;
    
    // 2. Teal & Orange Split Toning
    const shadowMask = clampF(1.0 - norm * 2.2, 0, 1);
    const highlightMask = clampF((norm - 0.45) * 2.0, 0, 1);
    
    let rAdj = r * (1.0 - shadowMask * 0.18 * factor);
    let gAdj = g * (1.0 - shadowMask * 0.02 * factor);
    let bAdj = b * (1.0 + shadowMask * 0.12 * factor);
    
    rAdj *= (1.0 + highlightMask * 0.15 * factor);
    gAdj *= (1.0 + highlightMask * 0.03 * factor);
    bAdj *= (1.0 - highlightMask * 0.18 * factor);
    
    // 3. Desaturate shadows
    const lumAdj = 0.299 * rAdj + 0.587 * gAdj + 0.114 * bAdj;
    const shadowDesat = shadowMask * 0.3 * factor;
    rAdj = rAdj * (1.0 - shadowDesat) + lumAdj * shadowDesat;
    gAdj = gAdj * (1.0 - shadowDesat) + lumAdj * shadowDesat;
    bAdj = bAdj * (1.0 - shadowDesat) + lumAdj * shadowDesat;
    
    // Recalculate lumAdj
    const finalLumAdj = 0.299 * rAdj + 0.587 * gAdj + 0.114 * bAdj;
    
    if (finalLumAdj > 0.001) {
      const ratio = targetLum / finalLumAdj;
      out[idx]     = clampF(rAdj * ratio, 0, 255);
      out[idx + 1] = clampF(gAdj * ratio, 0, 255);
      out[idx + 2] = clampF(bAdj * ratio, 0, 255);
    } else {
      out[idx]     = clampF(rAdj, 0, 255);
      out[idx + 1] = clampF(gAdj, 0, 255);
      out[idx + 2] = clampF(bAdj, 0, 255);
    }
    out[idx + 3] = img[idx + 3];
  }
  
  sendProgress('Cinematic grading...', 100);
  return out;
}

function applyFilmGrain(img, w, h, strength) {
  if (strength <= 0) return img;

  sendProgress('Applying film grain...', 0);
  const ch = 4;
  const out = new Float32Array(img.length);

  let seed = 12345;
  function lcgRandom() {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  }

  function gaussianNoise() {
    let u = 0, v = 0;
    while (u === 0) u = lcgRandom();
    while (v === 0) v = lcgRandom();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  const factor = (strength / 100) * 14.0;

  for (let i = 0; i < w * h; i++) {
    if (i % 8192 === 0) {
      sendProgress('Applying film grain...', (i / (w * h)) * 100);
    }
    const idx = i * ch;
    const r = img[idx], g = img[idx + 1], b = img[idx + 2];
    
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    const norm = clampF(lum / 255, 0, 1);
    
    // Parabolic midtone mask: 4.0 * Y * (1.0 - Y)
    const grainMask = clampF(4.0 * norm * (1.0 - norm), 0, 1);
    
    // Monochromatic noise
    const noise = gaussianNoise() * factor * grainMask;
    
    out[idx]     = clampF(r + noise, 0, 255);
    out[idx + 1] = clampF(g + noise, 0, 255);
    out[idx + 2] = clampF(b + noise, 0, 255);
    out[idx + 3] = img[idx + 3];
  }

  sendProgress('Applying film grain...', 100);
  return out;
}

// ======================== PIPELINE ORCHESTRATOR ==============================
// Manages the full upscaling pipeline:
// 1. Pre-analysis (noise estimation & classification)
// 2. Multi-pass Lanczos-3 upscaling with intermediate sharpening
// 3. Post-processing: detail → sharpen → CLAHE → color → anti-artifact

function processUpscale(srcBuffer, srcW, srcH, scale, options) {
  const startTime = Date.now();

  // Convert input to Float32Array for precision
  const src = new Float32Array(srcBuffer.length);
  for (let i = 0; i < srcBuffer.length; i++) {
    src[i] = srcBuffer[i];
  }

  // --- Stage 1: Pre-analysis ---
  currentStageIndex = 0;
  sendProgress('Analyzing image...', 0);
  const noiseLevel = estimateNoise(src, srcW, srcH);
  const classification = classifyImage(src, srcW, srcH, noiseLevel);
  const detectedType = classification.type;
  sendProgress('Analyzing image...', 100);

  // Set parameters dynamically if preset is auto
  let sharpening = options.sharpening ?? 50;
  let detail = options.detail ?? 50;
  let contrast = options.contrast ?? 50;
  let colorBoost = options.colorBoost ?? 30;
  let denoise = options.denoise ?? 20;
  let grain = options.grain ?? 0;
  let cinematic = options.cinematic ?? 0;

  if (options.preset === 'auto') {
    if (detectedType === 'graphic') {
      sharpening = 55;
      detail = 10;
      contrast = 40;
      colorBoost = 25;
      denoise = 5;
      grain = 0;
    } else if (detectedType === 'flat') {
      sharpening = 30;
      detail = 0;
      contrast = 30;
      colorBoost = 20;
      denoise = 10;
      grain = 0;
    } else if (detectedType === 'noisy') {
      sharpening = 20;
      detail = 30;
      contrast = 45;
      colorBoost = 20;
      denoise = 55;
      grain = 5;
    } else { // photo
      const noiseMod = clampF(1 - noiseLevel / 60, 0.25, 1.0);
      sharpening = Math.round(55 * noiseMod);
      detail = Math.round(65 * noiseMod);
      contrast = 50;
      colorBoost = 35;
      denoise = Math.round(15 + noiseLevel * 0.4);
      grain = 10;
    }
  } else if (options.preset && PRESETS[options.preset]) {
    const p = PRESETS[options.preset];
    sharpening = p.sharpening;
    detail = p.detail;
    contrast = p.contrast;
    colorBoost = p.colorBoost;
    denoise = p.denoise;
    grain = p.grain ?? 0;
  }

  const noiseReduction = clampF(noiseLevel / 100, 0, 1); // modulate sharpening

  // --- Determine multi-pass strategy ---
  const passes = [];
  let remaining = scale;
  while (remaining > 2) {
    passes.push(2);
    remaining /= 2;
  }
  if (remaining > 1) {
    passes.push(remaining);
  }
  if (passes.length === 0) passes.push(scale);

  // Calculate total stages: analysis + N passes + 6 post-processing stages + 1 grain stage
  totalStages = 1 + passes.length + 7;

  // --- Stage 2+: Multi-pass upscaling ---
  let current = src;
  let curW = srcW;
  let curH = srcH;

  for (let p = 0; p < passes.length; p++) {
    currentStageIndex = 1 + p;
    const passScale = passes[p];
    const newW = Math.round(curW * passScale);
    const newH = Math.round(curH * passScale);
    const label = `Upscaling pass ${p + 1}/${passes.length} (${passScale}×)...`;

    // Optional denoise before upscaling
    if (denoise > 30 && p === 0) {
      sendProgress('Denoising...', 0);
      current = medianFilter3x3(current, curW, curH);
    }

    // Lanczos-3 resize
    if (passScale !== 1) {
      current = separableLanczos3(current, curW, curH, newW, newH, label);
      curW = newW;
      curH = newH;

      // Edge-directed enhancement after each pass
      current = edgeDirectedEnhance(current, curW, curH, 0.3);
    }

    // Light intermediate sharpening to prevent blur accumulation
    if (passes.length > 1 && p < passes.length - 1) {
      const lightBlur = gaussianBlur(current, curW, curH, 0.5);
      const lightStrength = 0.3 * (1 - noiseReduction * 0.5);
      for (let i = 0; i < current.length; i++) {
        if (i % 4 < 3) { // RGB only
          current[i] += (current[i] - lightBlur[i]) * lightStrength;
        }
      }
    }
  }

  // --- Post-processing pipeline ---

  // Stage: Laplacian pyramid detail injection
  currentStageIndex = 1 + passes.length;
  if (detail > 0) {
    current = laplacianPyramidDetail(current, curW, curH, detail);
  }

  // Stage: Frequency-aware adaptive sharpening
  currentStageIndex++;
  const adjustedSharp = sharpening * (1 - noiseReduction * 0.4);
  if (adjustedSharp > 0) {
    current = frequencyAwareSharpen(current, curW, curH, adjustedSharp);
  }

  // Stage: CLAHE local contrast
  currentStageIndex++;
  if (contrast > 0) {
    current = clahe(current, curW, curH, contrast);
  }

  // Stage: Perceptual color enhancement
  currentStageIndex++;
  if (colorBoost > 0 || contrast > 0) {
    current = colorEnhance(current, curW, curH, colorBoost, contrast);
  }

  // Stage: Professional Cinematic Color Grading
  currentStageIndex++;
  if (cinematic > 0) {
    current = applyCinematicGrading(current, curW, curH, cinematic);
  }

  // Stage: Anti-artifact processing
  currentStageIndex++;
  current = antiArtifact(current, curW, curH);

  // Stage: Film Grain
  currentStageIndex++;
  if (grain > 0) {
    current = applyFilmGrain(current, curW, curH, grain);
  }

  // --- Convert to Uint8ClampedArray for output ---
  const output = new Uint8ClampedArray(curW * curH * 4);
  for (let i = 0; i < output.length; i++) {
    output[i] = clampByte(Math.round(current[i]));
  }

  const elapsed = Date.now() - startTime;
  return { 
    output, 
    width: curW, 
    height: curH, 
    elapsed, 
    detectedType, 
    autoSettings: { sharpening, detail, contrast, colorBoost, denoise, grain, cinematic } 
  };
}

// ======================== WORKER MESSAGE HANDLER ============================

self.onmessage = function(e) {
  const msg = e.data;
  if (msg.type !== 'upscale') return;

  try {
    const srcData = new Uint8ClampedArray(msg.imageBuffer);
    const result = processUpscale(srcData, msg.width, msg.height, msg.scale, msg.options || {});

    const buffer = result.output.buffer;
    self.postMessage({
      type: 'complete',
      imageBuffer: buffer,
      width: result.width,
      height: result.height,
      elapsed: result.elapsed,
      detectedType: result.detectedType,
      autoSettings: result.autoSettings
    }, [buffer]);
  } catch (err) {
    self.postMessage({
      type: 'error',
      message: err.message || 'Unknown error during upscaling'
    });
  }
};
