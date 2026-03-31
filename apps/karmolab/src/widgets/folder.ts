(function (): void {
  Toolbox.register({
    id: 'folder',
    title: '에러',
    category: 'play',
    desc: '폴더가 무한 증식하는 이스터에그',
    layout: 'form',
    icon: '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" stroke="currentColor" stroke-width="1.5" fill="none"/>',
    tabs: [
      {
        id: 'app',
        label: '에러',
        build: function (container: HTMLElement): void {
          Mdd.linePreset('idle_wake', { msg: '폴더가 무한 증식해요?!' });
          container.innerHTML = `
                    <div style="position:relative; width:100%; height:450px; background:#008080; overflow:hidden; border-radius:var(--radius-lg); font-family:sans-serif; cursor:default; user-select:none;" id="desktop">
                    </div>
                `;
          const desktopEl = container.querySelector('#desktop') as HTMLElement | null;
          if (!desktopEl) return;

          const desktop = desktopEl;
          let count = 0;
          let zIndex = 10;
          const MAX_WINDOWS = 150;

          let draggingWin: HTMLElement | null = null;
          let startX = 0;
          let startY = 0;
          let startLeft = 0;
          let startTop = 0;

          desktop.addEventListener('mousedown', (e: MouseEvent) => {
            const title = (e.target as Element).closest('.win-title');
            if (title?.parentElement) {
              draggingWin = title.parentElement;
              draggingWin.style.zIndex = String(zIndex++);
              startX = e.clientX;
              startY = e.clientY;
              startLeft = parseInt(draggingWin.style.left, 10) || 0;
              startTop = parseInt(draggingWin.style.top, 10) || 0;
            }
          });

          window.addEventListener('mousemove', (e: MouseEvent) => {
            if (!draggingWin) return;
            draggingWin.style.left = `${startLeft + (e.clientX - startX)}px`;
            draggingWin.style.top = `${startTop + (e.clientY - startY)}px`;
          });

          window.addEventListener('mouseup', () => {
            draggingWin = null;
          });

          function spawnWindow(x: number, y: number): void {
            if (count >= MAX_WINDOWS) {
              if (count === MAX_WINDOWS) {
                desktop.innerHTML =
                  '<div style="background:#0000AA; color:#fff; width:100%; height:100%; padding:20px; font-family:monospace; font-size:14px; font-weight:bold;">A fatal exception 0E has occurred at 0028:C0011E36.<br>System memory depleted.<br><br>* Press any key to terminate the current application.<br>* Press CTRL+ALT+DEL to restart your computer.<br>* You will lose any unsaved information in all applications.</div>';
                count++;
              }
              return;
            }

            const win = document.createElement('div');
            win.style.cssText = `position:absolute; left:${x}px; top:${y}px; width:250px; background:#c0c0c0; border:2px solid; border-color:#fff #808080 #808080 #fff; z-index:${zIndex++}; box-shadow: 2px 2px 4px rgba(0,0,0,0.5);`;

            win.innerHTML = `
                        <div class="win-title" style="background:#000080; color:#fff; padding:2px 4px; font-size:var(--font-size-xs); font-weight:bold; display:flex; justify-content:space-between; cursor:default; user-select:none;">
                            <span>Error</span>
                            <button class="win-close" style="background:#c0c0c0; color:#000; border:1px solid; border-color:#fff #808080 #808080 #fff; width:16px; height:16px; font-size:var(--font-size-2xs); line-height:1; cursor:default;">X</button>
                        </div>
                        <div style="padding:15px; text-align:center; color:#000; cursor:default; user-select:none;">
                            <div style="margin-bottom:15px; font-size:var(--font-size-xs); display:flex; align-items:center; gap:10px;">
                                <span style="font-size:24px;">⚠️</span>
                                <span>작업을 완료할 수 없습니다.</span>
                            </div>
                            <button class="win-ok" style="background:#c0c0c0; border:2px solid; border-color:#fff #808080 #808080 #fff; padding:4px 15px; cursor:default; color:#000;">확인</button>
                        </div>
                    `;

            const triggerCascade = (): void => {
              const currentLeft = parseInt(win.style.left, 10) || x;
              const currentTop = parseInt(win.style.top, 10) || y;
              win.style.zIndex = String(zIndex++);
              spawnWindow((currentLeft + 20) % (desktop.clientWidth - 200), (currentTop + 20) % (desktop.clientHeight - 100));
              spawnWindow(Math.max(10, currentLeft - 20), (currentTop + 35) % (desktop.clientHeight - 100));
              Toolbox.showToast('알 수 없는 오류로 창이 증식합니다!', 'error', undefined);
            };

            const closeBtn = win.querySelector('.win-close');
            const okBtn = win.querySelector('.win-ok');
            if (closeBtn) closeBtn.addEventListener('click', triggerCascade);
            if (okBtn) okBtn.addEventListener('click', triggerCascade);

            desktop.appendChild(win);
            count++;
          }

          spawnWindow(30, 30);
        }
      }
    ]
  });
})();
