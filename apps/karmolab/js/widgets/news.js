(function() {
    Toolbox.register({
        id: 'news', title: '뉴스',
        category: 'play',
        desc: '가짜 뉴스 헤드라인을 생성합니다',
        layout: 'form',
        icon: '<rect x="4" y="4" width="16" height="16" rx="2" ry="2" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M4 8h16 M8 4v4" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M8 12h8 M8 16h6" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'app', label: '뉴스', build: function(container) {
            Mdd.linePreset('tool_run', { msg: '뉴스를 읽고 있어요... 어디서 봤어요?' });
                container.innerHTML = `
                    <div style="height:350px; background:#111; border:4px solid #333; border-radius:12px; overflow:hidden; display:flex; flex-direction:column; position:relative;">
                        <div style="background:var(--error); color:#fff; font-weight:bold; padding:10px; text-align:center; letter-spacing:2px; z-index:10; box-shadow:0 2px 10px rgba(0,0,0,0.5);">🚨 긴급 속보 🚨</div>
                        <div id="newsMarquee" style="flex:1; position:relative; overflow:hidden; background:#0a0a0a; color:#ccc; font-family:serif;">
                            <div id="newsContent" style="position:absolute; width:100%; padding:20px; font-size:16px; line-height:1.8; text-align:justify;">
                            </div>
                        </div>
                    </div>
                `;
                const content = container.querySelector('#newsContent');
                const article = "오늘 낮, 익명의 집사가 고양이 간식을 훔쳐 먹은 것으로 밝혀져 충격을 주고 있습니다. 고양이 측 대변인은 '믿었던 집사에게 발등을 찍혔다'며 강하게 규탄했습니다. 이에 대해 집사 측은 '너무 맛있어 보여서 그만...' 이라며 말끝을 흐렸습니다. 한편, 이 소식을 접한 옆 동네 강아지는 '그럴 줄 알았다'며 냉소적인 반응을 보였습니다. 전문가들은 이번 사태가 반려동물과 반려인 간의 신뢰 관계에 큰 파장을 일으킬 것으로 전망하고 있습니다. ";
            
                // Generate extremely long identical news
                content.innerHTML = (article + "<br><br>").repeat(20);

                let y = 0;
                let animId;
                function scroll() {
                    y -= 0.5;
                    if (y < -1000) y = 0; // seamless reset 방지용 편법이나 20번 채웠으므로 오랫동안 버팀
                    content.style.transform = `translateY(${y}px)`;
                    animId = requestAnimationFrame(scroll);
                }
                const observer = new IntersectionObserver(e => {
                    if (e[0].isIntersecting) scroll();
                    else cancelAnimationFrame(animId);
                });
                observer.observe(container);
            } }]
    });
})();
