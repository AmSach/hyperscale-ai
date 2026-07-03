function copyCmd() {
  const txt = document.getElementById('installCmd').innerText;
  navigator.clipboard.writeText(txt).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = 'copied!';
    btn.style.background = '#0f0e0c';
    btn.style.color = '#f5d84a';
    setTimeout(() => {
      btn.textContent = 'copy';
      btn.style.background = '';
      btn.style.color = '';
    }, 2000);
  });
}

function initSlider(containerId, layerId, handleId) {
  const container = document.getElementById(containerId);
  const layer     = document.getElementById(layerId);
  const handle    = document.getElementById(handleId);
  if (!container || !layer || !handle) return;

  let dragging = false;

  function move(e) {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const pct = Math.min(98, Math.max(2, ((clientX - rect.left) / rect.width) * 100));
    layer.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    handle.style.left = pct + '%';
  }

  container.addEventListener('mousedown', () => dragging = true);
  container.addEventListener('touchstart', () => dragging = true, { passive: true });
  window.addEventListener('mouseup', () => dragging = false);
  window.addEventListener('touchend', () => dragging = false);
  window.addEventListener('mousemove', move);
  window.addEventListener('touchmove', move, { passive: true });
}

document.addEventListener('DOMContentLoaded', () => {
  initSlider('compContainer', 'layerBefore', 'sliderHandle');
  initSlider('compFull', 'layerFull', 'handleFull');
});
