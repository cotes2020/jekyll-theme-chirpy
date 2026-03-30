(function (): void {
  Toolbox.register({
    id: 'bounce',
    title: '바운스',
    category: 'play',
    desc: '공을 튕겨 바운스 게임을 합니다',
    layout: 'form',
    icon: '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" stroke="currentColor" stroke-width="1.5" fill="none"/>',
    tabs: [
      {
        id: 'app',
        label: '바운스',
        build: function (container: HTMLElement): void {
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
          const canvasEl = container.querySelector('#bounceCanvas') as HTMLCanvasElement | null;
          const logoInputEl = container.querySelector('#logoUpload') as HTMLInputElement | null;
          const countLabelEl = container.querySelector('#cornerCount') as HTMLElement | null;
          if (!canvasEl || !logoInputEl || !countLabelEl) return;

          const canvas = canvasEl;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;
          const c2d = ctx;
          const countLabel = countLabelEl;

          let canvasWidth = 600;
          let canvasHeight = 400;
          function resize(): void {
            const parent = canvas.parentElement;
            const rect = parent?.getBoundingClientRect();
            canvas.width = rect?.width || 600;
            canvas.height = 400;
            canvasWidth = canvas.width;
            canvasHeight = canvas.height;
          }
          resize();

          const img = new Image();
          let imgLoaded = false;
          const textLogo = 'ANTIGRAVITY';

          img.onload = () => {
            imgLoaded = true;
          };
          img.src =
            "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='30'><text x='0' y='20' font-family='sans-serif' font-size='16' font-weight='bold' fill='%2300ff00'>DVD</text></svg>";

          logoInputEl.onchange = function (e: Event): void {
            const input = e.target as HTMLInputElement;
            const file = input.files?.[0];
            if (!file) return;
            const url = URL.createObjectURL(file);
            imgLoaded = false;
            img.src = url;
          };

          let x = 50;
          let y = 50;
          let dx = 1.8;
          let dy = 1.4;
          let logoWidth = 120;
          let logoHeight = 40;
          let corners = 0;
          let hue = 0;

          let animId: number | undefined;
          function animate(): void {
            if (!canvas.offsetParent) {
              if (animId !== undefined) cancelAnimationFrame(animId);
              return;
            }

            c2d.fillStyle = '#000002';
            c2d.fillRect(0, 0, canvasWidth, canvasHeight);

            let hitX = false;
            let hitY = false;

            if (imgLoaded) {
              logoWidth = Math.min(120, img.width || 120);
              logoHeight = (logoWidth / img.width) * img.height || 40;
            } else {
              logoWidth = 120;
              logoHeight = 30;
            }

            x += dx;
            y += dy;

            if (x <= 0 || x + logoWidth >= canvasWidth) {
              dx = -dx;
              hitX = true;
              x = Math.max(0, Math.min(x, canvasWidth - logoWidth));
            }
            if (y <= 0 || y + logoHeight >= canvasHeight) {
              dy = -dy;
              hitY = true;
              y = Math.max(0, Math.min(y, canvasHeight - logoHeight));
            }

            if (hitX || hitY) hue = (hue + 45) % 360;
            if (hitX && hitY) {
              corners++;
              countLabel.textContent = String(corners);
              Toolbox.showToast?.('🎯 구석 적중!', undefined, undefined);
            }

            c2d.save();
            if (imgLoaded) {
              c2d.filter = `hue-rotate(${hue}deg)`;
              c2d.drawImage(img, x, y, logoWidth, logoHeight);
            } else {
              const textHue = (hue + 120) % 360;
              c2d.fillStyle = `hsl(${textHue}, 100%, 60%)`;
              c2d.font = 'bold 20px monospace';
              c2d.fillText(textLogo, x, y + 20);
            }
            c2d.restore();

            animId = requestAnimationFrame(animate);
          }

          const observer = new IntersectionObserver(
            (entries) => {
              if (entries[0]?.isIntersecting) {
                resize();
                animate();
              } else if (animId !== undefined) {
                cancelAnimationFrame(animId);
              }
            },
            { threshold: 0.1 }
          );
          observer.observe(canvas);
        }
      }
    ]
  });
})();
