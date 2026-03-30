// @ts-nocheck
(function() {
    Mdd.injectCSS('particle', `
        .playground-canvas { width:100%; flex:1; min-height:300px; background:var(--bg-primary); border:1px solid var(--border); border-radius:var(--radius-lg); cursor:crosshair; display:block; }
    `);

    Toolbox.register({
        id: 'particle', title: '파티클',
        category: 'play',
        desc: '마우스로 파티클을 움직이고 클릭으로 폭발시킵니다',
        layout: 'form',
        icon: '<circle cx="12" cy="12" r="2" fill="currentColor"/><circle cx="6" cy="6" r="1.5" fill="currentColor"/><circle cx="18" cy="8" r="1" fill="currentColor"/><circle cx="8" cy="18" r="1.5" fill="currentColor"/><circle cx="17" cy="17" r="1" fill="currentColor"/>',
        tabs: [{ id: 'app', label: '파티클', build: function(container) {
            container.innerHTML = `
                <div style="margin-bottom:12px; font-size:var(--font-size-xs); color:var(--text-tertiary);">마우스를 움직이거나 클릭(폭발)하여 파티클을 춤추게 하세요.</div>
                <canvas class="playground-canvas" id="particleCanvas"></canvas>
            `;
            const canvas = container.querySelector('#particleCanvas');
            const ctx = canvas.getContext('2d');

            Mdd.linePreset('tool_run', { mood: 'idle', msg: '파티클 놀이터예요~' });

            function resize() {
                const rect = canvas.parentElement.getBoundingClientRect();
                canvas.width = rect.width || 600;
                canvas.height = rect.height || 450;
            }
            resize(); window.addEventListener('resize', resize);

            let particles = [];
            let mouse = { x: canvas.width / 2, y: canvas.height / 2 };

            canvas.onmousemove = (e) => {
                mouse.x = e.offsetX; mouse.y = e.offsetY;
                for (let i = 0; i < 3; i++) particles.push(new Particle(mouse.x, mouse.y));
            };

            canvas.onclick = (e) => {
                for (let i = 0; i < 40; i++) particles.push(new Particle(e.offsetX, e.offsetY, true));
                Mdd.linePreset('idle_wake', { msg: '폭발이에요!!' }); Mdd.bounce();
                setTimeout(() => Mdd.setMood('happy'), 1500);
            };

            class Particle {
                constructor(x, y, isExplosion = false) {
                    this.x = x; this.y = y;
                    this.vx = (Math.random() - 0.5) * (isExplosion ? 8 : 2);
                    this.vy = (Math.random() - 0.5) * (isExplosion ? 8 : 2);
                    this.radius = Math.random() * 2.5 + 1;
                    this.color = `hsl(${Math.random() * 360}, 100%, 70%)`;
                    this.life = 100;
                }
                update() { this.x += this.vx; this.y += this.vy; this.vy += 0.04; this.vx *= 0.98; this.vy *= 0.98; this.life -= 1; }
                draw() { ctx.beginPath(); ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2); ctx.fillStyle = this.color; ctx.fill(); }
            }

            let animId;
            function animate() {
                if (!canvas.offsetParent) { cancelAnimationFrame(animId); return; }
                ctx.fillStyle = 'rgba(17, 17, 19, 0.15)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                particles = particles.filter(p => p.life > 0);
                particles.forEach(p => { p.update(); p.draw(); });
                animId = requestAnimationFrame(animate);
            }

            const observer = new IntersectionObserver((entries) => {
                if (entries[0].isIntersecting) { resize(); animate(); } else { cancelAnimationFrame(animId); }
            }, { threshold: 0.1 });
            observer.observe(canvas);
        } }]
    });
})();
