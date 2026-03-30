// @ts-nocheck
(function() {
    Toolbox.register({
        id: 'shylink', title: '어그로',
        category: 'play',
        desc: '움직이는 링크를 잡는 미니게임',
        layout: 'form',
        icon: '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71 M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" fill="none"/>',
        tabs: [{ id: 'app', label: '어그로', build: function(container) {
            Mdd.linePreset('meme_done', { msg: '이 링크... 잡을 수 있어요?' });
                container.innerHTML = `
                    <div style="position:relative; width:100%; height:450px; background:#1a1a2e; overflow:hidden; border-radius:var(--radius-lg); cursor:crosshair;" id="shyArea">
                        <a href="#" id="shyTarget" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#ff4757; font-size:24px; font-weight:900; text-decoration:none; text-shadow:0 0 10px rgba(255,71,87,0.5); white-space:nowrap; padding:20px; transition: opacity 0.1s; user-select:none;">
                            ❗❗[속보] 야 이거 봤냐??? 진짜 레전드다 ㅋㅋㅋㅋㅋㅋㅋㅋ❗❗
                        </a>
                    </div>
                
                    <!-- 릭롤 등 밈 동영상을 틀기 위한 풀스크린 모달 오버레이 -->
                    <div id="shyModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9); z-index:9999; justify-content:center; align-items:center;">
                        <div style="position:relative; width:80%; max-width:800px; aspect-ratio:16/9; background:#000; border-radius:12px; overflow:hidden; box-shadow:0 0 40px rgba(255,0,0,0.3);">
                            <button id="closeShyModal" style="position:absolute; top:10px; right:15px; color:#fff; font-size:24px; background:none; border:none; cursor:pointer; z-index:10; font-weight:bold;">✕</button>
                            <div id="shyIframeContainer" style="width:100%; height:100%;"></div>
                        </div>
                    </div>
                `;
                const area = container.querySelector('#shyArea');
                const target = container.querySelector('#shyTarget');
                const modal = container.querySelector('#shyModal');
                const closeBtn = container.querySelector('#closeShyModal');
                const iframeBox = container.querySelector('#shyIframeContainer');

                // 포인터가 다가갈수록 투명해지는 핵심 계산 로직
                area.onmousemove = (e) => {
                    const targetRect = target.getBoundingClientRect();
                    const targetX = targetRect.left + (targetRect.width / 2);
                    const targetY = targetRect.top + (targetRect.height / 2);
                
                    const dist = Math.hypot(e.clientX - targetX, e.clientY - targetY);
                
                    // 거리 200px 이내로 들어오면 투명화 시작
                    if (dist < 200) {
                        // 완전히 가까워지면(dist 0) opacity 0. 거리 200이면 opacity 1. 극강의 난이도를 위해 opacity 곡선을 가파르게 줌.
                        let op = (dist / 200) ** 2;
                        // 단, 0.02 이하는 클릭도 막히므로 최소한의 형태는 남김 (하지만 눈엔 안 보임)
                        target.style.opacity = Math.max(0.01, op);
                    } else {
                        target.style.opacity = 1;
                    }
                };

                area.onmouseleave = () => { target.style.opacity = 1; };

                // 극적으로 클릭 성공 시 이벤트
                target.onclick = (e) => {
                    e.preventDefault();
                    // Never Gonna Give You Up 릭롤 밈 링크 (autoplay=1 로 강제 재생)
                    iframeBox.innerHTML = '<iframe width="100%" height="100%" src="https://www.youtube.com/embed/dQw4w9WgXcQ?autoplay=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>';
                    modal.style.display = 'flex';
                };

                closeBtn.onclick = () => {
                    modal.style.display = 'none';
                    iframeBox.innerHTML = ''; // iframe 파괴하여 영상 종료
                };
            } }]
    });
})();
