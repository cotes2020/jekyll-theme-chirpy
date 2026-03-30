(function (): void {
  const PROGRESS_KEY = 'pet_strokes';

  type MilestoneDef = {
    msg: string;
    mood: string;
    achievement?: string;
    badge?: string;
  };

  const MILESTONES: Record<number, MilestoneDef> = {
    100: { msg: '100번! 겨우 시작이에요~', mood: 'happy', achievement: 'pet_100' },
    1000: { msg: '1,000번! 아직 999,000번 남았어요', mood: 'smug', achievement: 'pet_1000' },
    10000: { msg: '10,000번! 집사 가끔 대단해요', mood: 'happy', achievement: 'pet_10000' },
    100000: { msg: '100,000번!! 진짜로 하고 있었어요?!', mood: 'shock', achievement: 'pet_100000' },
    500000: { msg: '500,000번!!! 반이에요... 설마 진심이에요?!', mood: 'love', achievement: 'pet_500000' },
    1000000: { msg: '', mood: 'love', badge: 'pet_marriage' }
  };

  Toolbox.register({
    id: 'pet',
    title: '쓰다듬기',
    category: 'play',
    desc: '고양이를 쓰다듬고 호감도를 올립니다',
    layout: 'form',
    icon: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 4c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14c-2.67 0-5.18-1.08-7.07-2.83C6.46 15.83 9.11 14 12 14s5.54 1.83 7.07 3.17C17.18 18.92 14.67 20 12 20z" stroke="currentColor" stroke-width="1.5" fill="none"/>',
    tabs: [
      {
        id: 'app',
        label: '쓰다듬기',
        build: function (container: HTMLElement): void {
          container.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:380px; gap:16px; text-align:center; position:relative; overflow:hidden;">
                    <div style="font-size:14px; color:var(--text-secondary);">🐱 고양이 머리를 무한히 쓰다듬어주세요</div>
                    <div style="font-size:var(--font-size-xs); color:var(--text-tertiary);">목표: 1,000,000번</div>
                    <div id="petArea" style="font-size:100px; cursor:grab; user-select:none; filter:drop-shadow(0 4px 4px rgba(0,0,0,0.3)); transition:transform 0.1s;">🐱</div>
                    <div style="font-size:20px; font-weight:bold; color:var(--accent);">쓰담 횟수: <span id="petCount">0</span></div>
                    <div id="petMilestone" style="font-size:var(--font-size-xs); color:var(--success); min-height:16px;"></div>
                </div>
            `;
          const petAreaEl = container.querySelector('#petArea') as HTMLElement | null;
          const countLabelEl = container.querySelector('#petCount') as HTMLElement | null;
          const milestoneEl = container.querySelector('#petMilestone') as HTMLElement | null;
          if (!petAreaEl || !countLabelEl || !milestoneEl) return;

          const petArea = petAreaEl;
          const countLabel = countLabelEl;
          const milestone = milestoneEl;

          let count = Toolbox.getProgress?.(PROGRESS_KEY) ?? 0;
          countLabel.textContent = count.toLocaleString();

          Mdd.linePreset('achievement', { msg: '쓰다듬어달라요~' });

          let isDragging = false;
          petArea.addEventListener('mousedown', () => {
            isDragging = true;
          });
          window.addEventListener('mouseup', () => {
            isDragging = false;
          });

          petArea.addEventListener('mousemove', () => {
            if (!isDragging) return;
            count = Toolbox.incrementProgress?.(PROGRESS_KEY) ?? count + 1;
            countLabel.textContent = count.toLocaleString();
            petArea.style.transform = `scale(${1 + Math.random() * 0.1}) rotate(${(Math.random() - 0.5) * 10}deg)`;

            const m = MILESTONES[count];
            if (m) {
              if (m.badge) {
                Toolbox.unlockBadge?.(m.badge, { title: '검의 서약' });
                showMarriagePopup();
              } else if (m.achievement) {
                Toolbox.completeAchievement?.(m.achievement);
              }
              if (m.msg) {
                milestone.textContent = m.msg;
                Mdd.linePreset('achievement', { mood: m.mood, msg: m.msg });
                Mdd.bounce();
              }
            }
          });

          function showMarriagePopup(): void {
            Mdd.linePreset('achievement', { msg: '결혼합시다!!! 💖💍' });
            const overlay = document.createElement('div');
            overlay.style.cssText =
              'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:9999; display:flex; align-items:center; justify-content:center; flex-direction:column; color:#fff;';
            overlay.innerHTML = `
                    <div style="font-size:60px; animation:mdd-bounce 1s infinite;">💖💍🎉</div>
                    <div style="font-size:32px; font-weight:bold; margin-top:20px;">검의 서약: 결혼합시다!</div>
                    <div style="font-size:14px; margin-top:10px; color:pink;">(주의: 이 창은 영원히 닫히지 않아요 😼)</div>
                `;
            document.body.appendChild(overlay);
          }
        }
      }
    ]
  });
})();
