(function (): void {
  Toolbox.register({
    id: 'moon',
    title: '문페이즈',
    category: 'tool',
    desc: '오늘의 달 위상을 확인합니다',
    layout: 'form',
    icon: '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" stroke="currentColor" stroke-width="1.5" fill="none"/>',
    tabs: [
      {
        id: 'app',
        label: '문페이즈',
        build: function (container: HTMLElement): void {
          Mdd.linePreset('achievement', { msg: '달을 바라보고 있어요...' });
          container.innerHTML = `
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:380px; gap:16px; background:#020205; overflow:hidden; position:relative; border-radius:var(--radius-lg);">
                        <div style="font-size:14px; color:var(--text-secondary); letter-spacing:4px; z-index:2; text-shadow:0 0 4px #000;">THE MOON PHASE</div>
                        <div id="moonVisual" style="font-size:140px; line-height:1; z-index:2; filter:drop-shadow(0 0 20px rgba(255,255,200,0.15)); user-select:none; cursor:default;">🌕</div>
                        <div id="moonDesc" style="font-size:14px; color:var(--text-tertiary); z-index:2; font-family:monospace;">계산 중...</div>
                        <div style="font-size:var(--font-size-xs); color:#555; text-align:center; max-width:80%; margin-top:10px; line-height:1.5; z-index:2;">
                            <span style="color:#aaa; font-weight:bold;">TMI 🌕</span><br>
                            달의 공전과 자전 주기는 약 27.3일로 같아 항상 같은 면만 보입니다.<br>
                            지구에서 달까지의 거리는 약 384,400km로 빛의 속도로 1.28초가 걸리며,<br>
                            매년 지구에서 3.82cm씩 조금씩 조용히 멀어지고 있습니다.
                        </div>
                    </div>
                `;
          const visualEl = container.querySelector('#moonVisual') as HTMLElement | null;
          const descEl = container.querySelector('#moonDesc') as HTMLElement | null;
          if (!visualEl || !descEl) return;

          const visual = visualEl;
          const desc = descEl;

          function getMoonPhase(): number {
            const now = new Date();
            const lp = 2551443;
            const new_moon = new Date(1970, 0, 7, 20, 35, 0).getTime() / 1000;
            return ((now.getTime() / 1000 - new_moon) % lp) / lp;
          }

          let animId: number | undefined;
          function update(): void {
            const phases = [
              '🌑 신월 (New Moon)',
              '🌒 초승달 (Waxing Crescent)',
              '🌓 상현달 (First Quarter)',
              '🌔 차오르는 달 (Waxing Gibbous)',
              '🌕 보름달 (Full Moon)',
              '🌖 이지러지는 달 (Waning Gibbous)',
              '🌗 하현달 (Last Quarter)',
              '🌘 그믐달 (Waning Crescent)'
            ];

            const p = getMoonPhase();
            const phaseIndex = Math.floor(p * 8 + 0.5) % 8;

            visual.textContent = phases[phaseIndex].split(' ')[0];
            desc.textContent = phases[phaseIndex].split(' ').slice(1).join(' ') + ` (${(p * 100).toFixed(6)}%)`;

            animId = requestAnimationFrame(update);
          }

          const observer = new IntersectionObserver((entries) => {
            if (entries[0]?.isIntersecting) update();
            else if (animId !== undefined) cancelAnimationFrame(animId);
          });
          observer.observe(container);
        }
      }
    ]
  });
})();
