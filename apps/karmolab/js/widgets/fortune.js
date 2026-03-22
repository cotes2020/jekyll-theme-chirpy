(function() {
    Toolbox.register({
        id: 'fortune', title: '운세',
        category: 'play',
        desc: '오늘의 운세를 봅니다',
        layout: 'form',
        icon: '<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14l-5-4.87 6.91-1.01L12 2z" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'app', label: '운세', build: function(container) {
            container.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:380px; gap:20px; text-align:center;">
                    <div style="font-size:16px; font-weight:bold; color:var(--text-primary);">🔮 오늘의 행운을 점쳐보세요냥</div>
                    <div id="fortuneCookie" style="font-size:80px; cursor:pointer; transition:transform 0.2s; user-select:none;">🥠</div>
                    <div id="fortuneResult" style="font-size:15px; font-weight:500; color:var(--accent); min-height:40px; max-width:80%; line-height:1.6;">쿠키를 클릭해서 열어보세요!</div>
                    <button class="btn secondary" id="fortuneReset" style="display:none;">다시 뽑기</button>
                </div>
            `;
            const cookie = container.querySelector('#fortuneCookie');
            const result = container.querySelector('#fortuneResult');
            const reset = container.querySelector('#fortuneReset');

            Mdd.setMood('think'); Mdd.say('운명의 쿠키다냥... 두근');

            const fortunes = [
                "오늘은 집 밖으로 나가지 않는 것이 상책입니다냥.",
                "이불 밖은 위험합니다. 침대와 물아일체가 되세요냥.",
                "오늘의 행운의 장소: 당신의 방 구석지입니다냥.",
                "스마트폰 배터리가 유난히 빨리 닳는 하루가 될 것입니다냥.",
                "길을 걷다 양말이 벗겨질 수 있으니 주의하세요냥.",
                "오늘 산 복권은 꽝일 확률이 99.9%입니다냥.",
                "점심 메뉴 고민이 가장 큰 고비가 될 것입니다냥.",
                "택배가 올 것 같지만 안 올 것입니다냥."
            ];

            cookie.onclick = () => {
                if(cookie.textContent === '🍪') return;
                cookie.style.transform = 'scale(1.2) rotate(15deg)';
                Mdd.setMood('eating'); Mdd.bounce();
                setTimeout(() => {
                    cookie.style.transform = 'scale(1) rotate(0deg)';
                    cookie.textContent = '🍪';
                    const rand = fortunes[Math.floor(Math.random() * fortunes.length)];
                    result.innerHTML = `<span style="color:var(--error); font-weight:bold;">[대흉]</span><br>${rand}`;
                    reset.style.display = 'inline-block';
                    Mdd.setMood('sad'); Mdd.say('역시... 꽝이다냥... 😿');
                }, 300);
            };

            reset.onclick = () => {
                cookie.textContent = '🥠';
                result.textContent = '쿠키를 클릭해서 열어보세요!';
                reset.style.display = 'none';
                Mdd.setMood('think'); Mdd.say('이번엔 다를 거다냥...(아닐걸)');
            };
        } }]
    });
})();
