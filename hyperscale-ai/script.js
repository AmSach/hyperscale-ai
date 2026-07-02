function copyInstallCmd() {
    const cmdText = document.getElementById('installCommand').innerText;
    navigator.clipboard.writeText(cmdText).then(() => {
        const btn = document.getElementById('copyBtn');
        const origContent = btn.innerHTML;
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg><span style="color:#10b981">Copied!</span>`;
        setTimeout(() => { btn.innerHTML = origContent; }, 2000);
    });
}

// Interactive Image Comparison Slider via clip-path
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('compContainer');
    const layerBefore = document.getElementById('layerBefore');
    const handle = document.getElementById('sliderHandle');
    
    if (container && layerBefore && handle) {
        let isDragging = false;

        function moveSlider(e) {
            if (!isDragging) return;
            
            const rect = container.getBoundingClientRect();
            let clientX = e.clientX || (e.touches && e.touches[0].clientX);
            
            let offsetX = clientX - rect.left;
            let widthPercent = (offsetX / rect.width) * 100;
            
            if (widthPercent < 2) widthPercent = 2;
            if (widthPercent > 98) widthPercent = 98;
            
            // Using clip-path guarantees both images stay full-size 1:1 aligned
            let rightInset = 100 - widthPercent;
            layerBefore.style.clipPath = `inset(0 ${rightInset}% 0 0)`;
            handle.style.left = widthPercent + '%';
        }

        container.addEventListener('mousedown', () => isDragging = true);
        container.addEventListener('touchstart', () => isDragging = true, { passive: true });
        
        window.addEventListener('mouseup', () => isDragging = false);
        window.addEventListener('touchend', () => isDragging = false);
        
        window.addEventListener('mousemove', moveSlider);
        window.addEventListener('touchmove', moveSlider, { passive: true });
    }

    // Motion-Primitives 3D Interactive Card Tilt
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = ((y - centerY) / centerY) * -6;
            const rotateY = ((x - centerX) / centerX) * 6;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-6px) scale(1.02)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0) scale(1)`;
        });
    });
});
