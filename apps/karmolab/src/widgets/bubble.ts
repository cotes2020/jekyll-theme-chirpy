(function (): void {
  Mdd.injectCSS(
    'bubble',
    `
        .bubble-wrap-container { padding:24px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-lg); margin-bottom:24px; }
        .bubble-wrap-header { display:flex; justify-content:space-between; margin-bottom:16px; }
        .bubble-wrap-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(40px, 1fr)); gap:12px; flex:1; min-height:200px; overflow-y:auto; padding:4px; user-select:none; -webkit-user-select:none; }
        .bubble { width:40px; height:40px; border-radius:50%; background:radial-gradient(circle at 12px 12px, rgba(255,255,255,0.15), rgba(0,0,0,0.35)); box-shadow:0 4px 6px rgba(0,0,0,0.3), inset 0 -2px 5px rgba(0,0,0,0.4), inset 0 2px 5px rgba(255,255,255,0.1); cursor:pointer; transition:all 80ms ease; }
        .bubble:active { transform:scale(0.85); }
        .bubble.popped { background:rgba(0,0,0,0.32); box-shadow:inset 0 2px 4px rgba(0,0,0,0.6), 0 1px 2px rgba(255,255,255,0.05); transform:scale(0.9); }
    `
  );

  Toolbox.register({
    id: 'bubble',
    title: '뽁뽁이',
    category: 'play',
    desc: '뽁뽁이를 터뜨립니다',
    layout: 'form',
    icon: '<circle cx="9" cy="9" r="4" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="17" cy="9" r="3" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="13" cy="17" r="3.5" stroke="currentColor" stroke-width="1.5" fill="none"/>',
    tabs: [
      {
        id: 'app',
        label: '뽁뽁이',
        build: function (container: HTMLElement): void {
          container.innerHTML = `
                <div class="bubble-wrap-container">
                    <div class="bubble-wrap-header">
                        <span style="font-size:var(--font-size-sm); font-weight:600; color:var(--text-primary);">터진 뽁뽁이: <span id="popCount" style="color:var(--accent)">0</span> / <span id="popTotal">90</span></span>
                        <button class="btn btn-ghost" id="resetBubbleBtn">새 판 깔기</button>
                    </div>
                    <div class="bubble-wrap-grid" id="bubbleGrid"></div>
                </div>
            `;
          const gridEl = container.querySelector('#bubbleGrid') as HTMLElement | null;
          const countLabelEl = container.querySelector('#popCount') as HTMLElement | null;
          const totalLabelEl = container.querySelector('#popTotal') as HTMLElement | null;
          const resetBtnEl = container.querySelector('#resetBubbleBtn') as HTMLButtonElement | null;
          if (!gridEl || !countLabelEl || !totalLabelEl || !resetBtnEl) return;

          const grid = gridEl;
          const countLabel = countLabelEl;
          const totalLabel = totalLabelEl;
          const resetBtn = resetBtnEl;

          let popped = 0;
          const total = 90;
          totalLabel.textContent = String(total);

          Mdd.linePreset('tool_run', { mood: 'idle', msg: '뽁뽁이에요! 눌러보세요~' });

          function createGrid(): void {
            grid.innerHTML = '';
            popped = 0;
            countLabel.textContent = '0';
            for (let i = 0; i < total; i++) {
              const b = document.createElement('div');
              b.className = 'bubble';
              b.onclick = () => {
                if (!b.classList.contains('popped')) {
                  b.classList.add('popped');
                  popped++;
                  countLabel.textContent = String(popped);

                  if (popped % 15 === 0 && popped < total) {
                    Mdd.linePreset('idle_wake', { msg: '뽁! 뽁! 멈출 수 없어요!' });
                    Mdd.bounce();
                  }
                  if (popped === total) {
                    Mdd.linePreset('success', { msg: '올 클리어예요!! 🎉🎉' });
                  }
                }
              };
              grid.appendChild(b);
            }
          }

          resetBtn.onclick = () => {
            createGrid();
            Mdd.linePreset('success', { mood: 'happy', msg: '새 뽁뽁이를 깔았어요!' });
          };
          createGrid();
        }
      }
    ]
  });
})();
