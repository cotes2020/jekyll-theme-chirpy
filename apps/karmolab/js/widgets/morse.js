(function() {
    Toolbox.register({
        id: 'morse', title: '모스',
        category: null,
        desc: '모스 부호로 인코딩·디코딩합니다',
        layout: 'form',
        icon: '<path d="M2 12h4 M8 12h8 M18 12h4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        tabs: [{ id: 'app', label: '모스', build: function(container) {
            Mdd.linePreset('tool_run', { msg: '모르스 부호... 삐삐빗!' });
                container.innerHTML = `
                    <div class="field-group">
                        <label class="field-label">텍스트 입력 (영문/숫자/공백)</label>
                        <input type="text" id="morseInput" class="input-field" placeholder="예: SOS / HELLO" style="text-transform:uppercase;">
                    </div>
                    <div style="display:flex; align-items:center; gap:16px; margin:16px 0;">
                        <button class="btn btn-ghost" id="morsePlayBtn">소리 재생</button>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <div id="morseLed" style="width:16px; height:16px; border-radius:50%; background:#2a2a2e; border:2px solid var(--border); transition:all 30ms;"></div>
                            <span style="font-size:var(--font-size-xs); color:var(--text-tertiary);">신호 램프</span>
                        </div>
                    </div>
                    <div class="field-group">
                        <label class="field-label">모스 부호 변환 결과</label>
                        <div id="morseOutput" style="background:var(--bg-tertiary); padding:16px; border-radius:var(--radius-lg); font-family:monospace; font-size:22px; word-break:break-all; letter-spacing:6px; min-height:56px; color:var(--accent); line-height:1.5;"></div>
                    </div>
                `;

                const input = container.querySelector('#morseInput');
                const output = container.querySelector('#morseOutput');
                const playBtn = container.querySelector('#morsePlayBtn');
                const led = container.querySelector('#morseLed');

                const MORSE_MAP = {
                    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
                    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
                    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
                    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
                    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
                    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
                    '9': '----.', '0': '-----', ' ': ' '
                };

                input.addEventListener('input', (e) => {
                    const text = e.target.value.toUpperCase();
                    let result = [];
                    for (let char of text) {
                        if (MORSE_MAP[char] !== undefined) result.push(char === ' ' ? '/' : MORSE_MAP[char]);
                    }
                    output.textContent = result.join(' '); 
                });

                let isPlaying = false;
                playBtn.onclick = async function() {
                    if (isPlaying) return;
                    const morseStr = output.textContent;
                    if (!morseStr) { Toolbox.showToast('변환된 모스 부호가 없습니다.', 'error'); return; }

                    isPlaying = true;
                    playBtn.disabled = true; playBtn.textContent = '재생 중...';

                    try {
                        const actx = new (window.AudioContext || window.webkitAudioContext)();
                        const dotTime = 120; // 1도트 재생 시간(ms)

                        for (let char of morseStr) {
                            if (char === '.') {
                                triggerLed(true); playBeep(actx, dotTime); await sleep(dotTime); triggerLed(false);
                            } else if (char === '-') {
                                triggerLed(true); playBeep(actx, dotTime * 3); await sleep(dotTime * 3); triggerLed(false);
                            } else if (char === ' ') {
                                await sleep(dotTime); // 글자 내 간격
                            } else if (char === '/') {
                                await sleep(dotTime * 4); // 단어 간 공백
                            }
                            await sleep(dotTime); // 기본 딜레이
                        }
                        setTimeout(() => actx.close(), 100);
                    } catch (e) {}

                    isPlaying = false;
                    playBtn.disabled = false; playBtn.textContent = '소리 재생';
                };

                function triggerLed(on) {
                    led.style.background = on ? 'var(--accent)' : '#2a2a2e';
                    led.style.boxShadow = on ? '0 0 10px var(--accent)' : 'none';
                }

                function playBeep(ctx, duration) {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.connect(gain); gain.connect(ctx.destination);
                    osc.type = 'sine'; osc.frequency.setValueAtTime(600, ctx.currentTime);
                    gain.gain.setValueAtTime(0.15, ctx.currentTime);
                    osc.start(); osc.stop(ctx.currentTime + (duration / 1000));
                }

                const sleep = ms => new Promise(res => setTimeout(res, ms));
            } }]
    });
})();
