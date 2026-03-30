// @ts-nocheck
(function() {
    Toolbox.register({
        id: 'bounce', title: '바운스',
        category: 'play',
        desc: '공을 튕겨 바운스 게임을 합니다',
        layout: 'form',
        icon: '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'app', label: '바운스', build: function(container) {
            Mdd.linePreset('daily_start', { msg: '통통볼 놀이에요!' });
                container.innerHTML = `
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <div style="font-size:var(--font-size-xs); color:var(--text-tertiary);">구석 적중: <span id="cornerCount" style="color:var(--success); font-weight:bold;">0</span></div>
                        <label class="btn-ghost" style="padding:4px 8px; font-size:var(--font-size-xs); cursor:pointer;">
                            <input type="file" id="logoUpload" accept="image/*" style="display:none;">
                            이미지 업로드
                        </label>
                    </div>
                    <canvas class="playground-canvas" id="bounceCanvas" style="background:#000;"></canvas>
                `;
                const canvas = container.querySelector('#bounceCanvas');
                const ctx = canvas.getContext('2d');
                const countLabel = container.querySelector('#cornerCount');

                let canvasWidth = 600; let canvasHeight = 400; // 초기 디폴트
                function resize() {
                    const rect = canvas.parentElement.getBoundingClientRect();
                    canvas.width = rect.width || 600;
                    canvas.height = 400;
                    canvasWidth = canvas.width; canvasHeight = canvas.height;
                }
                resize();

                let img = new Image();
                let imgLoaded = false;
                let textLogo = "ANTIGRAVITY"; // 기본 로고

                img.onload = () => { imgLoaded = true; };
                img.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='30'><text x='0' y='20' font-family='sans-serif' font-size='16' font-weight='bold' fill='%2300ff00'>DVD</text></svg>"; // 임시 기본 DVD 스타일

                // 파일 업로드 처리
                container.querySelector('#logoUpload').onchange = function(e) {
                    const file = e.target.files[0];
                    if (!file) return;
                    const url = URL.createObjectURL(file);
                    imgLoaded = false;
                    img.src = url;
                };

                let x = 50; let y = 50;
                let dx = 1.8; let dy = 1.4;
                let logoWidth = 120; let logoHeight = 40;
                let corners = 0;
                let hue = 0;

                let animId;
                function animate() {
                    if (!canvas.offsetParent) { cancelAnimationFrame(animId); return; }

                    ctx.fillStyle = '#000002';
                    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

                    let hitX = false; let hitY = false;

                    if (imgLoaded) {
                        logoWidth = Math.min(120, img.width || 120);
                        logoHeight = (logoWidth / img.width) * img.height || 40;
                    } else {
                        logoWidth = 120; logoHeight = 30;
                    }

                    // 이동 및 벽 충돌
                    x += dx; y += dy;

                    if (x <= 0 || x + logoWidth >= canvasWidth) {
                        dx = -dx; hitX = true;
                        x = Math.max(0, Math.min(x, canvasWidth - logoWidth)); // 끼임 방지
                    }
                    if (y <= 0 || y + logoHeight >= canvasHeight) {
                        dy = -dy; hitY = true;
                        y = Math.max(0, Math.min(y, canvasHeight - logoHeight));
                    }

                    if (hitX || hitY) hue = (hue + 45) % 360; // 튕길 때 색상 랜덤 업데이트
                    if (hitX && hitY) { corners++; countLabel.textContent = corners; Toolbox.showToast('🎯 구석 적중!'); }

                    ctx.save();
                    if (imgLoaded) {
                        ctx.filter = `hue-rotate(${hue}deg)`;
                        ctx.drawImage(img, x, y, logoWidth, logoHeight);
                    } else {
                        const textHue = (hue + 120) % 360; // 이미지 부재 시 대비 텍스트 색상 전환
                        ctx.fillStyle = `hsl(${textHue}, 100%, 60%)`;
                        ctx.font = 'bold 20px monospace';
                        ctx.fillText(textLogo, x, y + 20);
                    }
                    ctx.restore();

                    animId = requestAnimationFrame(animate);
                }

                const observer = new IntersectionObserver((entries) => {
                    if (entries[0].isIntersecting) { resize(); animate(); }
                    else { cancelAnimationFrame(animId); }
                }, { threshold: 0.1 });
                observer.observe(canvas);
            } }]
    });
})();
