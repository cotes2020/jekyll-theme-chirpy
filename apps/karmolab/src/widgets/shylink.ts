(function (): void {
  Toolbox.register({
    id: 'shylink',
    title: '어그로',
    category: 'play',
    desc: '움직이는 링크를 잡는 미니게임',
    layout: 'form',
    icon: '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71 M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" fill="none"/>',
    tabs: [
      {
        id: 'app',
        label: '어그로',
        build: function (container: HTMLElement): void {
          Mdd.linePreset('meme_done', { msg: '이 링크... 잡을 수 있어요?' });
          container.innerHTML = `
                    <div style="position:relative; width:100%; height:450px; background:#1a1a2e; overflow:hidden; border-radius:var(--radius-lg); cursor:crosshair;" id="shyArea">
                        <a href="#" id="shyTarget" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#ff4757; font-size:24px; font-weight:900; text-decoration:none; text-shadow:0 0 10px rgba(255,71,87,0.5); white-space:nowrap; padding:20px; transition: opacity 0.1s; user-select:none;">
                            ❗❗[속보] 야 이거 봤냐??? 진짜 레전드다 ㅋㅋㅋㅋㅋㅋㅋㅋ❗❗
                        </a>
                    </div>
                
                    <div id="shyModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9); z-index:9999; justify-content:center; align-items:center;">
                        <div style="position:relative; width:80%; max-width:800px; aspect-ratio:16/9; background:#000; border-radius:12px; overflow:hidden; box-shadow:0 0 40px rgba(255,0,0,0.3);">
                            <button id="closeShyModal" style="position:absolute; top:10px; right:15px; color:#fff; font-size:24px; background:none; border:none; cursor:pointer; z-index:10; font-weight:bold;">✕</button>
                            <div id="shyIframeContainer" style="width:100%; height:100%;"></div>
                        </div>
                    </div>
                `;
          const areaEl = container.querySelector('#shyArea') as HTMLElement | null;
          const targetEl = container.querySelector('#shyTarget') as HTMLAnchorElement | null;
          const modalEl = container.querySelector('#shyModal') as HTMLElement | null;
          const closeBtnEl = container.querySelector('#closeShyModal') as HTMLButtonElement | null;
          const iframeBoxEl = container.querySelector('#shyIframeContainer') as HTMLElement | null;
          if (!areaEl || !targetEl || !modalEl || !closeBtnEl || !iframeBoxEl) return;

          const area = areaEl;
          const target = targetEl;
          const modal = modalEl;
          const closeBtn = closeBtnEl;
          const iframeBox = iframeBoxEl;

          area.onmousemove = (e: MouseEvent) => {
            const targetRect = target.getBoundingClientRect();
            const targetX = targetRect.left + targetRect.width / 2;
            const targetY = targetRect.top + targetRect.height / 2;

            const dist = Math.hypot(e.clientX - targetX, e.clientY - targetY);

            if (dist < 200) {
              const op = (dist / 200) ** 2;
              target.style.opacity = String(Math.max(0.01, op));
            } else {
              target.style.opacity = '1';
            }
          };

          area.onmouseleave = () => {
            target.style.opacity = '1';
          };

          target.onclick = (e: MouseEvent) => {
            e.preventDefault();
            iframeBox.innerHTML =
              '<iframe width="100%" height="100%" src="https://www.youtube.com/embed/dQw4w9WgXcQ?autoplay=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>';
            modal.style.display = 'flex';
          };

          closeBtn.onclick = () => {
            modal.style.display = 'none';
            iframeBox.innerHTML = '';
          };
        }
      }
    ]
  });
})();
