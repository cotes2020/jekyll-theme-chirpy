(function (): void {
  Mdd.injectCSS(
    'hacker',
    `
        .hacker-container { width:100%; flex:1; min-height:300px; background:#000; color:#00ff00; font-family:'SF Mono','Cascadia Code','Consolas',monospace; font-size:14px; padding:24px; border:1px solid var(--border); border-radius:var(--radius-lg); overflow-y:auto; white-space:pre-wrap; word-break:break-all; position:relative; user-select:none; }
        .hacker-cursor { display:inline-block; width:8px; height:16px; background:#00ff00; animation:hacker-blink 1s step-end infinite; vertical-align:middle; margin-left:2px; }
        @keyframes hacker-blink { 50% { opacity:0; } }
    `
  );

  let hackerText = '';
  let hackerIndex = 0;

  Toolbox.register({
    id: 'hacker',
    title: '해커',
    category: 'play',
    desc: '키보드를 연타해 해커 느낌의 텍스트를 출력합니다',
    layout: 'form',
    icon: '<path d="M4 17l6-6-6-6 M12 19h8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    tabs: [
      {
        id: 'app',
        label: '해커',
        build: function (container: HTMLElement): void {
          container.innerHTML = `
                    <div style="margin-bottom:12px; font-size:var(--font-size-xs); color:var(--text-tertiary);">키보드를 무작위로 연타하세요. (화면 클릭 후 타자)</div>
                    <div class="hacker-container" id="hackerScreen" tabindex="0">드러나지 않은 코드를 로드하려면 아무 키나 누르세요...<span class="hacker-cursor"></span></div>
                `;
          const screenEl = container.querySelector('#hackerScreen') as HTMLElement | null;
          if (!screenEl) return;

          const screen = screenEl;

          Mdd.linePreset('meme_done', { msg: '해킹 실험 개시예요... 히힛' });

          fetch('/apps/karmolab/js/toolbox.js')
            .then((r) => r.text())
            .then((t) => {
              hackerText = t;
            })
            .catch(() => {
              hackerText =
                "/* ACCESS GRANTED */\n\nfunction activateMainframe() {\n    const node = '0xDEADAES';\n    console.log('[SYSTEM]: CONNECTED');\n}";
            });

          let keystrokeCount = 0;
          screen.onkeydown = (e: KeyboardEvent) => {
            e.preventDefault();
            if (!hackerText) return;
            if (hackerIndex === 0) screen.innerHTML = '';

            const chunk = hackerText.substring(hackerIndex, hackerIndex + 5);
            const oldCur = screen.querySelector('.hacker-cursor');
            oldCur?.remove();
            screen.innerHTML += chunk.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            hackerIndex += 5;
            if (hackerIndex >= hackerText.length) hackerIndex = 0;

            const cur = document.createElement('span');
            cur.className = 'hacker-cursor';
            screen.appendChild(cur);
            screen.scrollTop = screen.scrollHeight;

            keystrokeCount++;
            if (keystrokeCount % 30 === 0) {
              Mdd.bounce();
              const quips = ['타다닥... 히힛!', '방화벽 돌파 중이에요...', '거의 다 왔어요!', '키보드에 불이 나요!'];
              Mdd.linePreset('meme_done', { msg: quips[Math.floor(Math.random() * quips.length)] });
            }
          };

          setTimeout(() => screen.focus(), 200);
        }
      }
    ]
  });
})();
