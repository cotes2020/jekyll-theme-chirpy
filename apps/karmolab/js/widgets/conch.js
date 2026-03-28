(function() {
    Toolbox.register({
        ...Toolbox.getLazyWidgetPublicMeta('conch'),
        tabs: [{ id: 'app', label: '소라고동', build: function(container) {
            container.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:380px; gap:16px; text-align:center;">
                    <div style="font-size:14px; color:var(--text-secondary);">🐚 마법의 소라고동에게 질문을 속삭이세요</div>
                    <input type="text" id="conchInput" class="input" style="width:80%; max-width:300px; text-align:center;" placeholder="질문을 입력한 뒤 버튼을 누르세요">
                    <div id="conchVisual" style="font-size:70px; cursor:pointer; transition:transform 0.3s; user-select:none;">🐚</div>
                    <div id="conchResult" style="font-size:16px; font-weight:bold; color:var(--accent); min-height:24px;"></div>
                    <button class="btn primary" id="conchBtn">소라고동님께 여쭤보기</button>
                </div>
            `;
            const input = container.querySelector('#conchInput');
            const visual = container.querySelector('#conchVisual');
            const result = container.querySelector('#conchResult');
            const btn = container.querySelector('#conchBtn');

            Mdd.setMood('idle'); Mdd.say('소라고동님은 모든 걸 알고 있어요...');

            const answers = [
                "그래.", "안 돼.", "가만히 있어.", "다시 한번 물어봐.",
                "좋아.", "그럼.", "절대 안 돼.", "하늘을 봐.", "훗.",
                "다음에 다시 와.", "글쎄...", "그건 비밀이야."
            ];

            function ask() {
                const text = input.value.trim();
                if (!text) { Toolbox.showToast('질문을 입력하셔야 해요!', 'warning'); return; }

                visual.style.transform = 'scale(1.2) rotate(15deg)';
                result.textContent = '음...';
                Mdd.setMood('think'); Mdd.say('소라고동님이 고민 중이세요...');

                setTimeout(() => {
                    visual.style.transform = 'scale(1) rotate(0deg)';
                    const rand = answers[Math.floor(Math.random() * answers.length)];
                    result.textContent = `"${rand}"`;
                    input.value = '';
                    Mdd.setMood('smug'); Mdd.bounce(); Mdd.say(`소라고동님의 답변: "${rand}"`);
                }, 800);
            }

            btn.onclick = ask;
            input.onkeypress = (e) => { if (e.key === 'Enter') ask(); };
        } }]
    });
})();
