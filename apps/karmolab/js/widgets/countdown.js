(function() {
    Toolbox.register({
        id: 'countdown', title: '카운트다운',
        category: 'tool',
        desc: '카운트다운 타이머를 설정합니다',
        layout: 'form',
        icon: '<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M12 6v6l4 2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>',
        tabs: [{ id: 'app', label: '카운트다운', build: function(container) {
            Mdd.setMood('think'); Mdd.say('시간이 흐르고 있다냥...');
                container.innerHTML = `
                    <div style="display:flex; flex-direction:column; align-items:center; gap:20px; text-align:center; padding-top:40px;">
                        <div style="font-size:14px; color:var(--text-secondary);">⏰ 인생이 낭비되는 속도 (목표 시간까지 남은 ms)</div>
                        <div id="countdownMs" style="font-size:54px; font-variant-numeric: tabular-nums; font-family:monospace; font-weight:900; color:var(--accent); text-shadow:0 0 10px rgba(100,100,250,0.4); letter-spacing:-2px;">0000000000</div>
                        <div style="display:flex; gap:12px; margin-top:20px; align-items:center;">
                            <span style="font-size:var(--font-size-xs); color:var(--text-tertiary);">목표일:</span>
                            <input type="datetime-local" id="countdownTarget" class="input" style="width:200px; padding:6px; font-size:var(--font-size-sm);">
                        </div>
                        <div style="margin-top:15px; font-size:var(--font-size-xs); color:var(--text-tertiary); display:grid; grid-template-columns:1fr; gap:6px; text-align:left; background:rgba(255,255,255,0.03); padding:15px; border-radius:8px; width:100%; max-width:350px;">
                            <div style="font-weight:bold; margin-bottom:4px; color:var(--text-secondary); text-align:center;">--- 낭비 체감 가이드 (ms 전환표) ---</div>
                            <div style="display:flex; justify-content:space-between;"><span>1분</span> <span>60,000 ms</span></div>
                            <div style="display:flex; justify-content:space-between;"><span>1시간</span> <span>3,600,000 ms</span></div>
                            <div style="display:flex; justify-content:space-between;"><span>1일</span> <span>86,400,000 ms</span></div>
                            <div style="display:flex; justify-content:space-between;"><span>1주일</span> <span>604,800,000 ms</span></div>
                            <div style="display:flex; justify-content:space-between;"><span>1달 (30일)</span> <span>2,592,000,000 ms</span></div>
                            <div style="display:flex; justify-content:space-between;"><span>1년 (365일)</span> <span>31,536,000,000 ms</span></div>
                        </div>
                    </div>
                `;
                const msDisplay = container.querySelector('#countdownMs');
                const targetInput = container.querySelector('#countdownTarget');

                // 기본값: 내일 자정 (오늘이 끝나기까지)
                const now = new Date();
                const tomorrow = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1);
            
                // 로컬 시간에 맞춰 날짜-시간 뷰 설정 (YYYY-MM-DDTHH:mm)
                const tzoffset = now.getTimezoneOffset() * 60000;
                const localISOTime = (new Date(tomorrow - tzoffset)).toISOString().slice(0, 16);
                targetInput.value = localISOTime;

                let animId;
                function update() {
                    const targetTime = new Date(targetInput.value).getTime();
                    const diff = targetTime - Date.now();
                
                    if (isNaN(diff) || diff < 0) {
                        msDisplay.textContent = "시간이 다 되었습니다.";
                        msDisplay.style.color = "var(--error)";
                    } else {
                        msDisplay.style.color = "var(--accent)";
                        // 10자리 고정 패딩으로 인생 낭비 시각화 극대화
                        msDisplay.textContent = diff.toString().padStart(10, '0');
                    }
                
                    animId = requestAnimationFrame(update);
                }

                const observer = new IntersectionObserver(entries => {
                    if (entries[0].isIntersecting) update();
                    else cancelAnimationFrame(animId);
                });
                observer.observe(container);
            } }]
    });
})();
