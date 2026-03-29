(function() {
    Toolbox.register({
        ...Toolbox.getLazyWidgetPublicMeta('gacha'),
        tabs: [{ id: 'app', label: '가챠', build: function(container) {
            Mdd.linePreset('idle_wake', { msg: '가챠를 돌려요! 뭐가 나올까요!' });
            container.innerHTML = `
                <div style="display:flex; flex-direction:column; padding:20px; height:380px; box-sizing:border-box; overflow:hidden;">
                    <button class="btn btn-primary" id="gachaBtn" style="margin-bottom:15px;">로또 1등 당첨될 때까지 시뮬레이션 돌리기</button>
                    <div id="gachaLog" style="flex:1; background:rgba(0,0,0,0.3); border-radius:8px; padding:15px; overflow-y:auto; font-size:var(--font-size-sm); font-family:monospace; display:flex; flex-direction:column; gap:6px;">
                        <div style="color:var(--text-tertiary); text-align:center;">--- 시뮬레이션 대기 중 ---</div>
                    </div>
                </div>
            `;
            const btn = container.querySelector('#gachaBtn');
            const log = container.querySelector('#gachaLog');

            let isRunning = false;
            let attempts = 0;

            function generateLotto() {
                const nums = [];
                while(nums.length < 6) {
                    const r = Math.floor(Math.random() * 45) + 1;
                    if(!nums.includes(r)) nums.push(r);
                }
                return nums.sort((a,b) => a-b);
            }

            function simulate() {
                if(!isRunning) return;

                attempts += 100; // 고속 연산 (프레임당 100번)
                const winNums = [1, 2, 3, 4, 5, 6]; // 예시 당첨번호
                
                for(let i=0; i<100; i++) {
                    const pick = generateLotto();
                    const match = pick.filter(n => winNums.includes(n)).length;
                    
                    if (match === 6) {
                        log.innerHTML = `<div style="color:var(--success); font-weight:bold; font-size:15px;">🎉대박! ${attempts.toLocaleString()}번 만에 로또 1등 당첨! 🎉</div>` + log.innerHTML;
                        isRunning = false; btn.textContent = '시뮬레이션 재시작'; btn.classList.remove('danger');
                        return;
                    }
                }

                log.innerHTML = `<div style="color:var(--text-secondary)">🔄 ${attempts.toLocaleString()}회 시도 중... (당첨 확률 1/8,145,060)</div>` + log.innerHTML;
                
                if (log.children.length > 50) log.removeChild(log.lastChild); // 과부하 방지
                requestAnimationFrame(simulate);
            }

            btn.onclick = () => {
                if (isRunning) {
                    isRunning = false;
                    btn.textContent = '시뮬레이션 시작'; btn.classList.remove('danger');
                } else {
                    isRunning = true; attempts = 0; log.innerHTML = '';
                    btn.textContent = '시뮬레이션 정지'; btn.classList.add('danger');
                    simulate();
                }
            };
        } }]
    });
})();
