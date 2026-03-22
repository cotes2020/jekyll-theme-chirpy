(function() {
    Toolbox.register({
        id: 'fontgacha', title: '폰트가챠',
        category: 'play',
        desc: '가챠로 폰트를 바꿉니다',
        layout: 'form',
        icon: '<path d="M4 7V4h16v3 M9 20h6 M12 4v16" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'app', label: '폰트가챠', build: function(container) {
            container.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:380px; gap:16px; text-align:center;">
                    <div style="font-size:14px; color:var(--text-secondary);">🎰 폰트를 바꾸려면 가챠를 돌리세요!</div>
                    <div id="fontDisplay" style="font-size:24px; font-weight:bold; margin:10px 0; min-height:36px; transition:all 0.3s;">가나다라 ABC Hello 123</div>
                    <div id="fontGrade" style="font-size:40px; min-height:50px; transition:transform 0.3s;"></div>
                    <button class="btn btn-accent" id="drawFontBtn">🎰 가챠 돌리기!</button>
                    <div id="fontResult" style="font-size:14px; font-weight:bold; color:var(--accent); min-height:18px;"></div>
                    <div id="fontCollection" style="font-size:var(--font-size-xs); color:var(--text-tertiary);">수집: <span id="fontCollected">0</span> / 7</div>
                </div>
            `;
            const display = container.querySelector('#fontDisplay');
            const btn = container.querySelector('#drawFontBtn');
            const result = container.querySelector('#fontResult');
            const grade = container.querySelector('#fontGrade');
            const collected = container.querySelector('#fontCollected');

            Mdd.setMood('idle'); Mdd.say('폰트 가챠예요! 뭐가 나올까요~');

            const fonts = [
                { name: '기본 폰트', style: 'sans-serif', grade: 'C', color: '#888' },
                { name: '굴림', style: 'Gulim, sans-serif', grade: 'B', color: '#4FC3F7' },
                { name: '궁서', style: 'Gungsuh, serif', grade: 'B', color: '#4FC3F7' },
                { name: '바탕', style: 'Batang, serif', grade: 'A', color: '#AB47BC' },
                { name: '나눔고딕', style: '"Nanum Gothic", sans-serif', grade: 'A', color: '#AB47BC' },
                { name: '궁서체', style: 'GungsuhChe, serif', grade: 'SR', color: '#FFD700' },
                { name: 'Comic Sans', style: '"Comic Sans MS", cursive', grade: 'UR', color: '#FF5722' }
            ];

            const collectedSet = new Set();

            btn.onclick = () => {
                // 연출: 빠르게 폰트 돌리기
                btn.disabled = true;
                Mdd.setMood('shock'); Mdd.say('돌아가요...!!');
                let spinCount = 0;
                const spinInterval = setInterval(() => {
                    const r = fonts[Math.floor(Math.random() * fonts.length)];
                    display.style.fontFamily = r.style;
                    grade.textContent = r.grade;
                    grade.style.color = r.color;
                    spinCount++;
                    if (spinCount > 15) {
                        clearInterval(spinInterval);
                        // 최종 결과
                        const rand = fonts[Math.floor(Math.random() * fonts.length)];
                        display.style.fontFamily = rand.style;
                        document.body.style.fontFamily = rand.style;
                        grade.textContent = `[${rand.grade}]`;
                        grade.style.color = rand.color;
                        grade.style.transform = 'scale(1.3)';
                        setTimeout(() => grade.style.transform = 'scale(1)', 300);

                        result.innerHTML = `획득: <span style="color:${rand.color}">[${rand.grade}]</span> ${rand.name}`;
                        collectedSet.add(rand.name);
                        collected.textContent = collectedSet.size;

                        if (rand.grade === 'UR') {
                            Mdd.setMood('cheer'); Mdd.bounce(); Mdd.say('UR!! 전설급 폰트예요!! 🎉🎉🎉');
                        } else if (rand.grade === 'SR') {
                            Mdd.setMood('happy'); Mdd.bounce(); Mdd.say('SR! 꽤 레어한 득템이에요!');
                        } else if (rand.grade === 'C') {
                            Mdd.setMood('sad'); Mdd.say('C급... 흔해빠진 거예요...');
                        } else {
                            Mdd.setMood('happy'); Mdd.say(`${rand.name} 나왔어요!`);
                        }

                        btn.disabled = false;
                        Toolbox.showToast(`${rand.name} 획득!`);
                    }
                }, 80);
            };
        } }]
    });
})();
