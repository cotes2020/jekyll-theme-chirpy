(function (): void {
  Mdd.injectCSS(
    'particle',
    `
        .playground-canvas { width:100%; flex:1; min-height:300px; background:var(--bg-primary); border:1px solid var(--border); border-radius:var(--radius-lg); cursor:crosshair; display:block; }
    `
  );

  Toolbox.register({
    id: 'particle',
    title: '파티클',
    category: 'play',
    desc: '마우스로 파티클을 움직이고 클릭으로 폭발시킵니다',
    layout: 'form',
    icon: '<circle cx="12" cy="12" r="2" fill="currentColor"/><circle cx="6" cy="6" r="1.5" fill="currentColor"/><circle cx="18" cy="8" r="1" fill="currentColor"/><circle cx="8" cy="18" r="1.5" fill="currentColor"/><circle cx="17" cy="17" r="1" fill="currentColor"/>',
    tabs: [
      {
        id: 'app',
        label: '파티클',
        build: function (container: HTMLElement): void {
          container.innerHTML = `
                <div style="margin-bottom:12px; font-size:var(--font-size-xs); color:var(--text-tertiary);">마우스를 움직이거나 클릭(폭발)하여 파티클을 춤추게 하세요.</div>
                <canvas class="playground-canvas" id="particleCanvas"></canvas>
            `;
          const canvasEl = container.querySelector('#particleCanvas') as HTMLCanvasElement | null;
          if (!canvasEl) return;

          const canvas = canvasEl;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;
          const c2d = ctx;

          Mdd.linePreset('tool_run', { mood: 'idle', msg: '파티클 놀이터예요~' });

          function resize(): void {
            const parent = canvas.parentElement;
            const rect = parent?.getBoundingClientRect();
            canvas.width = rect?.width || 600;
            canvas.height = rect?.height || 450;
          }
          resize();
          window.addEventListener('resize', resize);

          class Particle {
            x: number;
            y: number;
            vx: number;
            vy: number;
            radius: number;
            color: string;
            life: number;

            constructor(x: number, y: number, isExplosion = false) {
              this.x = x;
              this.y = y;
              this.vx = (Math.random() - 0.5) * (isExplosion ? 8 : 2);
              this.vy = (Math.random() - 0.5) * (isExplosion ? 8 : 2);
              this.radius = Math.random() * 2.5 + 1;
              this.color = `hsl(${Math.random() * 360}, 100%, 70%)`;
              this.life = 100;
            }

            update(): void {
              this.x += this.vx;
              this.y += this.vy;
              this.vy += 0.04;
              this.vx *= 0.98;
              this.vy *= 0.98;
              this.life -= 1;
            }

            draw(): void {
              c2d.beginPath();
              c2d.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
              c2d.fillStyle = this.color;
              c2d.fill();
            }
          }

          let particles: Particle[] = [];
          let mouse = { x: canvas.width / 2, y: canvas.height / 2 };

          canvas.onmousemove = (e: MouseEvent) => {
            mouse.x = e.offsetX;
            mouse.y = e.offsetY;
            for (let i = 0; i < 3; i++) particles.push(new Particle(mouse.x, mouse.y));
          };

          canvas.onclick = (e: MouseEvent) => {
            for (let i = 0; i < 40; i++) particles.push(new Particle(e.offsetX, e.offsetY, true));
            Mdd.linePreset('idle_wake', { msg: '폭발이에요!!' });
            Mdd.bounce();
            setTimeout(() => Mdd.setMood('happy'), 1500);
          };

          let animId: number | undefined;
          function animate(): void {
            if (!canvas.offsetParent) {
              if (animId !== undefined) cancelAnimationFrame(animId);
              return;
            }
            c2d.fillStyle = 'rgba(17, 17, 19, 0.15)';
            c2d.fillRect(0, 0, canvas.width, canvas.height);
            particles = particles.filter((p) => p.life > 0);
            particles.forEach((p) => {
              p.update();
              p.draw();
            });
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
