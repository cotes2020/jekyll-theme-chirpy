// @ts-nocheck
(function() {
    Toolbox.register({
        id: 'darkroom', title: '다크룸',
        category: 'play',
        desc: '어두운 방에서 마우스로 빛을 비춥니다',
        layout: 'form',
        icon: '<path d="M12 12m-9 0a9 9 0 1 0 18 0a9 9 0 1 0 -18 0 M12 3v18 M3 12h18" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'app', label: '다크룸', build: function(container) {
            Mdd.linePreset('idle_wake', { msg: '깜깜해요... 무서워요...' });
                container.innerHTML = `
                    <div style="position:relative; width:100%; flex:1; min-height:300px; background:#000; overflow:hidden; border-radius:var(--radius-lg); cursor:none;" id="darkArea">
                        <canvas id="darkCanvas" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;"></canvas>
                        <div id="escapeDoor" style="position:absolute; font-size:18px; opacity:0; cursor:none; width:24px; height:24px; display:flex; align-items:center; justify-content:center; transition:opacity 0.2s; user-select:none;">🚪</div>
                        <div id="darkWin" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#fff; font-size:18px; font-weight:bold; text-align:center; display:none; background:rgba(0,0,0,0.8); padding:20px; border-radius:12px; border:1px solid #333; cursor:default; user-select:none;">
                             🎉 탈출 성공! 축하해요 🎉<br>
                            <button class="btn" style="margin-top:16px; font-size:var(--font-size-xs); cursor:default;" id="resetDark">다시 갇히기</button>
                        </div>
                    </div>
                `;
                const canvas = container.querySelector('#darkCanvas');
                const ctx = canvas.getContext('2d');
                const area = container.querySelector('#darkArea');
                const door = container.querySelector('#escapeDoor');
                const winBox = container.querySelector('#darkWin');

                function resize() {
                    canvas.width = area.clientWidth; canvas.height = area.clientHeight;
                    resetDoor();
                }
                setTimeout(resize, 100);

                let mx = -100, my = -100;
                area.onmousemove = (e) => {
                    const rect = area.getBoundingClientRect();
                    mx = e.clientX - rect.left; my = e.clientY - rect.top;
                
                    // 마우스가 문 근처에 오면 문이 아주 살짝 보이기 시작
                    const dist = Math.hypot(mx - (door.offsetLeft + 12), my - (door.offsetTop + 12));
                    door.style.opacity = dist < 50 ? (1 - dist/50) : 0;
                };

                function resetDoor() {
                    door.style.left = `${Math.random() * (canvas.width - 40) + 20}px`;
                    door.style.top = `${Math.random() * (canvas.height - 40) + 20}px`;
                }

                door.onclick = () => { winBox.style.display = 'block'; };
                container.querySelector('#resetDark').onclick = () => { winBox.style.display = 'none'; resetDoor(); };

                let animId;
                function animate() {
                    ctx.fillStyle = '#010101'; ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.save();
                    ctx.globalCompositeOperation = 'destination-out';
                    ctx.beginPath();
                    ctx.arc(mx, my, 40, 0, Math.PI * 2); // 탐색을 위한 약간 넓은 스포트라이트 (40px)
                    ctx.fill();
                    ctx.restore();

                    animId = requestAnimationFrame(animate);
                }

                const observer = new IntersectionObserver((entries) => {
                    if (entries[0].isIntersecting) { resize(); animate(); }
                    else { cancelAnimationFrame(animId); }
                }, { threshold: 0.1 });
                observer.observe(area);
            } }]
    });
})();
