(function (): void {
  Toolbox.register({
    id: 'stone',
    title: '돌',
    category: 'play',
    desc: '돌을 던져 점을 봅니다',
    layout: 'form',
    icon: '<path d="M12 3C7 3 4 8 4 12s2 8 8 8 8-4 8-8-3-9-8-9z M8 12h8" stroke="currentColor" stroke-width="1.5" fill="none"/>',
    tabs: [
      {
        id: 'app',
        label: '돌',
        build: function (container: HTMLElement): void {
          container.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:380px; gap:20px; text-align:center;">
                    <div style="font-size:14px; color:var(--text-secondary);">🪨 당신의 디지털 반려돌이에요</div>
                    <div id="stoneEmoji" style="font-size:100px; user-select:none; cursor:default; filter:drop-shadow(0 5px 5px rgba(0,0,0,0.5)); transition:transform 0.2s;">🪨</div>
                    <div id="stoneStatus" style="font-size:var(--font-size-sm); color:var(--text-tertiary); min-height:18px;">돌은 가만히 있습니다.</div>
                    <div style="display:flex; gap:10px;">
                        <button class="btn btn-ghost" id="stoneFeed">밥 주기</button>
                        <button class="btn btn-ghost" id="stoneWalk">산책가기</button>
                        <button class="btn btn-ghost" id="stonePraise">칭찬하기</button>
                    </div>
                    <div style="font-size:var(--font-size-xs); color:var(--text-tertiary); margin-top:10px;">함께한 시간: <span id="stoneTime">0</span>초</div>
                </div>
            `;
          const statusEl = container.querySelector('#stoneStatus') as HTMLElement | null;
          const stoneEmojiEl = container.querySelector('#stoneEmoji') as HTMLElement | null;
          const timeElEl = container.querySelector('#stoneTime') as HTMLElement | null;
          const feedEl = container.querySelector('#stoneFeed') as HTMLButtonElement | null;
          const walkEl = container.querySelector('#stoneWalk') as HTMLButtonElement | null;
          const praiseEl = container.querySelector('#stonePraise') as HTMLButtonElement | null;
          if (!statusEl || !stoneEmojiEl || !timeElEl || !feedEl || !walkEl || !praiseEl) return;

          const status = statusEl;
          const stoneEmoji = stoneEmojiEl;
          const timeEl = timeElEl;

          let seconds = 0;

          Mdd.linePreset('idle_sleep', { msg: '돌이에요... 반응이 없는 게 정상이에요.' });

          const reactions: Record<'feed' | 'walk' | 'praise', string[]> = {
            feed: ['...미동조차 하지 않아요.', '...씹지도 않아요.', '...소화기관이 없어요.'],
            walk: ['...굴러가지 않아요.', '...다리가 없어요.', '...움직임을 거부해요.'],
            praise: ['...여전히 돌이에요.', '...수줍지도 않아요.', '...감정이 없어요.']
          };

          function react(type: 'feed' | 'walk' | 'praise'): void {
            const msgs = reactions[type];
            const msg = msgs[Math.floor(Math.random() * msgs.length)];
            status.textContent = msg;
            stoneEmoji.style.transform = 'rotate(3deg)';
            setTimeout(() => {
              stoneEmoji.style.transform = 'rotate(0deg)';
            }, 200);

            if (type === 'feed') {
              Mdd.linePreset('tool_run', { mood: 'eating', msg: '밥을 줬는데... 반응이 없어요...' });
            } else if (type === 'walk') {
              Mdd.linePreset('error', { msg: '산책 시켰는데... 꿈쩍도 안 해요...' });
            } else {
              Mdd.linePreset('tool_run', { mood: 'idle', msg: '칭찬해도 소용없어요... 돌이니까요.' });
            }

            setTimeout(() => Mdd.setMood('sleep'), 2000);
          }

          feedEl.onclick = () => react('feed');
          walkEl.onclick = () => react('walk');
          praiseEl.onclick = () => react('praise');

          const timer = window.setInterval(() => {
            if (!container.offsetParent) {
              clearInterval(timer);
              return;
            }
            seconds++;
            timeEl.textContent = seconds.toLocaleString();
          }, 1000);
        }
      }
    ]
  });
})();
