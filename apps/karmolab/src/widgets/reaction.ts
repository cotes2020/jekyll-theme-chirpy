(function (): void {
  Toolbox.register({
    id: 'reaction',
    title: '반응속도',
    category: 'play',
    desc: '반응 속도를 측정합니다',
    layout: 'form',
    icon: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    tabs: [
      {
        id: 'app',
        label: '반응속도',
        build: function (container: HTMLElement): void {
          Mdd.linePreset('tool_run', { msg: '반응 속도를 테스트해요!' });

          const STATES = { WAITING: 0, READY: 1, GREEN: 2, RESULT: 3, EARLY: 4 } as const;
          type ReactionState = (typeof STATES)[keyof typeof STATES];

          let state: ReactionState = STATES.WAITING;
          let greenTime = 0;
          let timeout: ReturnType<typeof setTimeout> | null = null;
          const results: number[] = [];

          container.innerHTML = `
                <div style="display:flex;flex-direction:column;align-items:center;gap:20px;max-width:500px;margin:0 auto;width:100%;">
                    <div id="reactionBox" style="width:100%;height:280px;border-radius:var(--radius-lg);display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;user-select:none;transition:background 0.15s;background:var(--bg-tertiary);border:1px solid var(--border);">
                        <div id="reactionIcon" style="font-size:48px;margin-bottom:16px;">🎯</div>
                        <div id="reactionText" style="font-size:18px;font-weight:700;color:var(--text-primary);">클릭하여 시작</div>
                        <div id="reactionSub" style="font-size:var(--font-size-sm);color:var(--text-tertiary);margin-top:8px;">화면이 초록색이 되면 클릭!</div>
                    </div>
                    <div id="reactionResults" style="width:100%;font-size:var(--font-size-sm);color:var(--text-secondary);text-align:center;min-height:24px;"></div>
                    <div id="reactionHistory" style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;"></div>
                    <div id="reactionBest" style="font-size:var(--font-size-xs);color:var(--text-tertiary);"></div>
                </div>
            `;

          const boxEl = container.querySelector('#reactionBox') as HTMLElement | null;
          const iconEl = container.querySelector('#reactionIcon') as HTMLElement | null;
          const textEl = container.querySelector('#reactionText') as HTMLElement | null;
          const subEl = container.querySelector('#reactionSub') as HTMLElement | null;
          const resultsEl = container.querySelector('#reactionResults') as HTMLElement | null;
          const historyEl = container.querySelector('#reactionHistory') as HTMLElement | null;
          const bestEl = container.querySelector('#reactionBest') as HTMLElement | null;
          if (!boxEl || !iconEl || !textEl || !subEl || !resultsEl || !historyEl || !bestEl) return;

          const box = boxEl;
          const icon = iconEl;
          const text = textEl;
          const sub = subEl;
          const resultsOut = resultsEl;
          const historyOut = historyEl;
          const bestOut = bestEl;

          const bestTime = Toolbox.getProgress?.('reaction_best');
          if (bestTime) bestOut.textContent = `최고 기록: ${bestTime}ms`;

          function setState(newState: ReactionState): void {
            state = newState;
            if (timeout !== null) clearTimeout(timeout);
            timeout = null;

            switch (state) {
              case STATES.WAITING:
                box.style.background = 'var(--bg-tertiary)';
                box.style.borderColor = 'var(--border)';
                icon.textContent = '🎯';
                text.textContent = '클릭하여 시작';
                text.style.color = 'var(--text-primary)';
                sub.textContent = '화면이 초록색이 되면 클릭!';
                break;

              case STATES.READY:
                box.style.background = '#1a1a2e';
                box.style.borderColor = '#2a2a4e';
                icon.textContent = '⏳';
                text.textContent = '기다리세요...';
                text.style.color = '#fbbf24';
                sub.textContent = '';
                timeout = setTimeout(() => setState(STATES.GREEN), 1000 + Math.random() * 4000);
                break;

              case STATES.GREEN:
                greenTime = Date.now();
                box.style.background = '#065f46';
                box.style.borderColor = '#34d399';
                icon.textContent = '⚡';
                text.textContent = '지금 클릭!';
                text.style.color = '#34d399';
                sub.textContent = '';
                break;

              case STATES.EARLY:
                box.style.background = '#450a0a';
                box.style.borderColor = '#f87171';
                icon.textContent = '❌';
                text.textContent = '너무 빨랐습니다!';
                text.style.color = '#f87171';
                sub.textContent = '클릭하여 다시 시도';
                Mdd.linePreset('idle_wake', { msg: '너무 빨라요!' });
                break;

              default:
                break;
            }
          }

          box.onclick = () => {
            switch (state) {
              case STATES.WAITING:
              case STATES.RESULT:
              case STATES.EARLY:
                setState(STATES.READY);
                break;
              case STATES.READY:
                setState(STATES.EARLY);
                break;
              case STATES.GREEN: {
                const reactionTime = Date.now() - greenTime;
                state = STATES.RESULT;
                results.push(reactionTime);

                box.style.background = 'var(--accent-subtle)';
                box.style.borderColor = 'var(--accent)';
                icon.textContent = '🏆';
                text.textContent = `${reactionTime}ms`;
                text.style.color = 'var(--accent)';
                sub.textContent = '클릭하여 다시 시도';

                const avg = Math.round(results.reduce((a, b) => a + b, 0) / results.length);
                resultsOut.textContent = `평균: ${avg}ms · ${results.length}회 시도`;

                const chip = document.createElement('span');
                chip.style.cssText = `padding:4px 10px;border-radius:100px;font-size:var(--font-size-xs);font-weight:600;background:var(--bg-tertiary);color:${reactionTime < 250 ? 'var(--success)' : reactionTime < 400 ? 'var(--accent)' : 'var(--text-secondary)'};`;
                chip.textContent = `${reactionTime}ms`;
                historyOut.appendChild(chip);

                const currentBest = Toolbox.getProgress?.('reaction_best') ?? 0;
                if (!currentBest || reactionTime < currentBest) {
                  Toolbox.setProgress?.('reaction_best', reactionTime);
                  bestOut.textContent = `최고 기록: ${reactionTime}ms`;
                  bestOut.style.color = 'var(--accent)';
                  Mdd.linePreset('success', { msg: '신기록이에요!' });
                } else if (reactionTime < 200) {
                  Mdd.linePreset('idle_wake', { msg: '빨라요!!! 인간 맞아요?!' });
                } else if (reactionTime < 300) {
                  Mdd.linePreset('success', { mood: 'happy', msg: '꽤 빨라요~' });
                } else {
                  Mdd.linePreset('meme_done', { msg: '나보다 느려요 ㅋ' });
                }

                if (reactionTime < 200) {
                  Toolbox.completeAchievement?.('reaction_200', { title: '초고속 반응 200ms' });
                }
                if (reactionTime < 150) {
                  Toolbox.completeAchievement?.('reaction_150', { title: '번개 반응 150ms' });
                }
                Mdd.addAffection(1);
                break;
              }
              default:
                break;
            }
          };
        }
      }
    ]
  });
})();
