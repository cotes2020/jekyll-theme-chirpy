// @ts-nocheck
(function() {
    Toolbox.register({
        id: 'eyes', title: '눈동자',
        category: 'play',
        desc: '마우스를 따라오는 눈동자',
        layout: 'form',
        icon: '<path d="M12 5c-7 0-10 7-10 7s3 7 10 7 10-7 10-7-3-7-10-7z" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'app', label: '눈동자', build: function(container) {
            Mdd.linePreset('tool_run', { mood: 'idle', msg: '눈동자가 따라가요...' });
                container.innerHTML = `
                    <div id="eyesArena" style="position:relative; width:100%; flex:1; min-height:300px; background:#1e1e2d; border-radius:var(--radius-lg); overflow:hidden; cursor:crosshair; box-shadow:inset 0 0 20px rgba(0,0,0,0.5);">
                    </div>
                `;
                const arena = container.querySelector('#eyesArena');
                const eyes = [];
            
                // Create eye grid
                for (let i=0; i<40; i++) {
                    const eyeBox = document.createElement('div');
                    eyeBox.style.cssText = `
                        position:absolute; width:36px; height:36px; background:#fff; border-radius:50%;
                        box-shadow: 0 0 5px rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center;
                        overflow:hidden;
                    `;
                    eyeBox.style.left = `${Math.random() * 90}%`;
                    eyeBox.style.top = `${Math.random() * 90}%`;
                
                    const pupil = document.createElement('div');
                    pupil.style.cssText = `
                        width:14px; height:14px; background:#111; border-radius:50%; transition: transform 0.05s linear;
                    `;
                
                    const eyelidTop = document.createElement('div');
                    eyelidTop.style.cssText = `
                        position:absolute; top:0; left:0; width:100%; height:50%; background:#1e1e2d; transform-origin:top; transition:transform 0.15s ease-in-out; transform:scaleY(0); border-bottom:2px solid #000;
                    `;
                    const eyelidBot = document.createElement('div');
                    eyelidBot.style.cssText = `
                        position:absolute; bottom:0; left:0; width:100%; height:50%; background:#1e1e2d; transform-origin:bottom; transition:transform 0.15s ease-in-out; transform:scaleY(0); border-top:2px solid #000;
                    `;

                    eyeBox.appendChild(pupil);
                    eyeBox.appendChild(eyelidTop);
                    eyeBox.appendChild(eyelidBot);
                    arena.appendChild(eyeBox);
                
                    eyes.push({ box: eyeBox, pupil: pupil, top: eyelidTop, bot: eyelidBot });
                }

                let mx = 0, my = 0;
                arena.onmousemove = (e) => {
                    const rect = arena.getBoundingClientRect();
                    mx = e.clientX - rect.left;
                    my = e.clientY - rect.top;
                
                    eyes.forEach(eye => {
                        const ex = eye.box.offsetLeft + 18;
                        const ey = eye.box.offsetTop + 18;
                        const angle = Math.atan2(my - ey, mx - ex);
                        // 동공 가동 범위 제한 (반지름 이내)
                        const dist = Math.min(8, Math.hypot(mx - ex, my - ey) / 10);
                        const px = Math.cos(angle) * dist;
                        const py = Math.sin(angle) * dist;
                        eye.pupil.style.transform = `translate(${px}px, ${py}px)`;
                    });
                };

                // 클릭하면 깜짝 놀라서 전체 눈을 스르륵 감고 뜹니다.
                arena.onmousedown = () => {
                    eyes.forEach(eye => {
                        eye.top.style.transform = 'scaleY(1)';
                        eye.bot.style.transform = 'scaleY(1)';
                    });
                };
                arena.onmouseup = () => {
                    setTimeout(() => {
                        eyes.forEach(eye => {
                            eye.top.style.transform = 'scaleY(0)';
                            eye.bot.style.transform = 'scaleY(0)';
                        });
                    }, 200);
                };
                arena.onmouseleave = arena.onmouseup;
            } }]
    });
})();
