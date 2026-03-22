(function() {
    Toolbox.register({
        id: 'toast', title: '토스트',
        category: 'play',
        desc: '토스트 알림을 띄웁니다',
        layout: 'form',
        icon: '<path d="M6 4h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z M8 10h8 M8 14h8" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'app', label: '토스트', build: function(container) {
            Mdd.setMood('smug'); Mdd.say('알림이 끝도 없어요...');
                container.innerHTML = `
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:360px; gap:16px;">
                        <div id="toastStatus" style="font-size:15px; font-weight:600; color:var(--text-secondary);">식빵 위에 마우스를 올리고 있으세요 🍞</div>
                        <div id="toastImg" style="font-size:80px; cursor:pointer; transition:all 80ms; user-select:none;">🍞</div>
                        <div style="width:200px; height:8px; background:var(--bg-tertiary); border-radius:4px; overflow:hidden;">
                            <div id="toastProgress" style="width:0%; height:100%; background:var(--accent); transition:width 50ms;"></div>
                        </div>
                        <button class="btn btn-ghost" id="resetToast">태초의 빵으로</button>
                    </div>
                `;
                const img = container.querySelector('#toastImg');
                const status = container.querySelector('#toastStatus');
                const progress = container.querySelector('#toastProgress');
            
                let heat = 0; let isHover = false;
            
                img.onmouseenter = () => { isHover = true; };
                img.onmouseleave = () => { isHover = false; };
                container.querySelector('#resetToast').onclick = () => { heat = 0; update(); };

                let interval;
                function startLoop() {
                    if(interval) return;
                    interval = setInterval(() => {
                        if (isHover) { heat = Math.min(100, heat + 0.6); } 
                        else { heat = Math.max(0, heat - 0.15); }
                        update();
                    }, 50);
                }
                function stopLoop() {
                    clearInterval(interval);
                    interval = null;
                }

                const observer = new IntersectionObserver(entries => {
                    if(entries[0].isIntersecting) startLoop();
                    else stopLoop();
                });
                observer.observe(container);

                function update() {
                    progress.style.width = `${heat}%`;
                
                    // CSS 필터 조합을 통한 굽기 렌더링
                    let sep = heat * 0.7; let br = 1 - (heat * 0.006); let ct = 1 + (heat * 0.005);
                    if (heat > 80) { br = Math.max(0.2, 0.52 - (heat-80)*0.03); } // 80% 이후 타기 시작
                    img.style.filter = `sepia(${sep}%) brightness(${br}) contrast(${ct})`;

                    if (heat < 30) status.textContent = "식빵이 서늘합니다 ❄️";
                    else if (heat < 65) status.textContent = "노릇노릇 익어가는 중... ☀️";
                    else if (heat < 80) status.innerHTML = "<span style='color:var(--success)'>🚨 딱 좋게 구워졌습니다! 골든 토스트! 🚨</span>";
                    else if (heat < 95) status.innerHTML = "<span style='color:var(--warning)'>탄다! 타요! 마우스 떼세요! 🔥</span>";
                    else status.innerHTML = "<span style='color:var(--error)'>석탄이 되었습니다... 💀</span>";
                }
            } }]
    });
})();
