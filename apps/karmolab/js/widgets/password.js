(function() {
    Toolbox.register({
        id: 'password', title: '비번',
        category: 'feature',
        desc: '랜덤 비밀번호를 생성합니다',
        layout: 'form',
        icon: '<rect x="3" y="11" width="18" height="11" rx="2" ry="2" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M7 11V7a5 5 0 0 1 10 0v4" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="12" cy="16" r="1" fill="currentColor"/>',
        tabs: [{ id: 'app', label: '비번', build: function(container) {
            Mdd.setMood('smug'); Mdd.say('비밀번호 맞춰볼래냥?');
                container.innerHTML = `
                    <div style="display:flex; flex-direction:column; padding:20px; height:100%; box-sizing:border-box;">
                        <div style="font-size:18px; font-weight:bold; color:var(--text-primary); margin-bottom:8px;">비밀번호 야구 ⚾</div>
                        <div style="font-size:var(--font-size-sm); color:var(--text-secondary); margin-bottom:20px;">
                            알파벳(대/소문자) + 숫자 + 일부 기호(!@#$%^&*)가 섞인 <b>4자리</b> 비밀번호를 맞춰보세요냥!<br>
                            결과와 함께 살살 긁는 힌트가 제공됩니다.
                        </div>
                    
                        <div style="display:flex; gap:10px; margin-bottom:20px;">
                            <input type="text" id="pwInput" class="input" style="flex:1; font-family:monospace; font-size:16px; letter-spacing:2px; text-align:center;" maxlength="4" placeholder="4자리 입력">
                            <button class="btn btn-primary" id="pwSubmit">해킹 시도</button>
                            <button class="btn btn-ghost" id="pwReset">포기(새 게임)</button>
                        </div>

                        <div id="pwLogs" style="flex:1; background:rgba(0,0,0,0.3); border-radius:8px; padding:15px; overflow-y:auto; font-size:var(--font-size-sm); font-family:monospace; display:flex; flex-direction:column; gap:8px;">
                        </div>
                    </div>
                `;
                const input = container.querySelector('#pwInput');
                const btnSubmit = container.querySelector('#pwSubmit');
                const btnReset = container.querySelector('#pwReset');
                const logs = container.querySelector('#pwLogs');

                const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*';
                let answer = '';
            
                function generateAnswer() {
                    answer = '';
                    // 4자리 중복 허용 난수 생성
                    for (let i=0; i<4; i++) {
                        answer += chars.charAt(Math.floor(Math.random() * chars.length));
                    }
                    logs.innerHTML = '<div style="color:var(--text-tertiary); text-align:center;">--- 시스템 포트 활성화: 대상의 4자리 비밀번호가 설정됨 ---</div>';
                    input.value = '';
                    input.focus();
                }

                function getSneakyHint(guess) {
                    const hints = [];
                    // 1. 대/소문자/숫자/기호 구성 비율 힌트
                    let hasLower = false, hasUpper = false, hasNum = false, hasSym = false;
                    for (let i=0; i<4; i++) {
                        const c = answer[i];
                        if (/[a-z]/.test(c)) hasLower = true;
                        else if (/[A-Z]/.test(c)) hasUpper = true;
                        else if (/[0-9]/.test(c)) hasNum = true;
                        else hasSym = true;
                    }
                
                    if (hasLower) hints.push("혹시 정답 어딘가에 귀여운 소문자가 숨어있지 않을까냥?");
                    if (hasUpper) hints.push("정답에 크고 우람한 대문자가 포함되어 있는 것 같다냥...");
                    if (hasNum) hints.push("숫자가 하나쯤은 섞여야 제맛이지냥.");
                    if (hasSym) hints.push("특수기호(!@#$ 등)를 안 쓴 보안 허접은 아니라냥!");

                    // 2. 특정 위치의 힌트 (가끔 조롱조)
                    const randPos = Math.floor(Math.random() * 4);
                    const tgt = answer[randPos];
                    let typeStr = '';
                    if (/[a-z]/.test(tgt)) typeStr = '소문자';
                    else if (/[A-Z]/.test(tgt)) typeStr = '대문자';
                    else if (/[0-9]/.test(tgt)) typeStr = '숫자';
                    else typeStr = '특수기호';
                
                    hints.push(`이건 비밀인데... ${randPos + 1}번째 글자는 바로 [${typeStr}] 라냥!`);
                    hints.push(`그딴 비밀번호로는 내 지갑을 못 턴다냥~`);
                    hints.push(`인생의 진리는 삽질에 있다냥. 다시 해라냥.`);

                    // 랜덤하게 하나 선택해서 조롱
                    return hints[Math.floor(Math.random() * hints.length)];
                }

                function checkGuess() {
                    const guess = input.value;
                    if (guess.length !== 4) { Toolbox.showToast('4자리를 정확히 입력해라냥!', 'warning'); return; }

                    let strike = 0, ball = 0;
                    let ansLetters = answer.split('');
                    let guessLetters = guess.split('');
                    let colors = ['#333', '#333', '#333', '#333']; // 기본 회색 (아웃)

                    // 1패스: 스트라이크 (위치+값 일치)
                    for (let i=0; i<4; i++) {
                        if (guessLetters[i] === ansLetters[i]) {
                            colors[i] = 'var(--success)';
                            ansLetters[i] = null; 
                            strike++;
                        }
                    }

                    // 2패스: 볼 (위치는 다르되 값이 포함됨)
                    for (let i=0; i<4; i++) {
                        if (colors[i] !== 'var(--success)' && ansLetters.includes(guessLetters[i])) {
                            colors[i] = 'var(--warning)';
                            ansLetters[ansLetters.indexOf(guessLetters[i])] = null;
                            ball++;
                        }
                    }

                    let tilesHtml = guessLetters.map((g, i) => `
                        <div style="width:32px; height:32px; display:flex; align-items:center; justify-content:center; background:${colors[i]}; font-weight:bold; font-size:18px; border-radius:4px; color:#fff; box-shadow:inset 0 0 4px rgba(0,0,0,0.3);">${g}</div>
                    `).join('');

                    const logEntry = document.createElement('div');
                    logEntry.style.padding = '10px'; logEntry.style.background = 'rgba(255,255,255,0.03)'; logEntry.style.borderRadius = '6px';
                    logEntry.style.borderLeft = strike === 4 ? '3px solid var(--success)' : '3px solid #444';

                    if (strike === 4) {
                        logEntry.innerHTML = `
                            <div style="display:flex; gap:6px; margin-bottom:8px; justify-content:center;">${tilesHtml}</div>
                            <div style="text-align:center; color:var(--success); font-weight:bold;">[해킹 성공] 완벽히 일치한다냥! 🎉</div>
                        `;
                        setTimeout(generateAnswer, 3000);
                    } else {
                        const hintMsg = getSneakyHint(guess);
                        logEntry.innerHTML = `
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <div style="display:flex; gap:6px;">${tilesHtml}</div>
                                <span style="color:var(--text-secondary); font-size:var(--font-size-xs); font-weight:bold; letter-spacing:1px;">${strike}S ${ball}B</span>
                            </div>
                            <div style="color:var(--text-tertiary); font-size:var(--font-size-xs); background:rgba(0,0,0,0.2); padding:8px; border-radius:4px;">💡 ${hintMsg}</div>
                        `;
                    }
                
                    logs.insertBefore(logEntry, logs.firstChild);
                    input.value = ''; input.focus();
                }

                btnSubmit.onclick = checkGuess;
                input.onkeypress = (e) => { if (e.key === 'Enter') checkGuess(); };
                btnReset.onclick = generateAnswer;

                generateAnswer();
            } }]
    });
})();
