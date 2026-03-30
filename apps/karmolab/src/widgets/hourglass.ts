// @ts-nocheck
(function() {
    Toolbox.register({
        id: 'hourglass', title: '모래시계',
        category: 'tool',
        desc: '모래시계 타이머를 실행합니다',
        layout: 'form',
        icon: '<path d="M6 2h12 M6 22h12 M6 2l6 10-6 10 M18 2l-6 10 18 10" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'app', label: '모래시계', build: function(container) {
            Mdd.linePreset('tool_run', { msg: '모래시계... 졸려요...' });
            container.innerHTML = `
                <div style="display:flex; flex-direction:column; padding:20px; height:380px; box-sizing:border-box; overflow:hidden;">
                    <div style="font-size:var(--font-size-xs); color:var(--text-tertiary); margin-bottom:8px;">픽셀 모래가 다 떨어지면 페이지가 새로고침돼요...</div>
                    <canvas id="hourglassCanvas" style="flex:1; background:#0a0a0c; border-radius:8px;"></canvas>
                </div>
            `;
            const canvas = container.querySelector('#hourglassCanvas');
            const ctx = canvas.getContext('2d');
            
            function resize() {
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;
            }
            requestAnimationFrame(resize);

            const grid = [];
            const cols = 60, rows = 60;
            const w = canvas.width / cols, h = canvas.height / rows;

            // 모래 생성 (상단 절반)
            for(let y=0; y<30; y++) {
                for(let x=10; x<50; x++) {
                    if (Math.random() > 0.3) grid.push({ x, y, color: `hsl(${20 + Math.random()*20}, 70%, 60%)` });
                }
            }

            let animId;
            function animate() {
                if (!container.offsetParent) { cancelAnimationFrame(animId); return; }

                ctx.fillStyle = '#0a0a0c';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                const currentW = canvas.width / cols;
                const currentH = canvas.height / rows;

                // 물리 피직스 (아래로 낙하)
                for (let i = grid.length - 1; i >= 0; i--) {
                    const p = grid[i];
                    if (p.y < rows - 1) {
                        // 바로 하단 빈자리 검사
                        const isBlocked = grid.some(o => o.x === p.x && o.y === p.y + 1);
                        if (!isBlocked) {
                            p.y++;
                        } else {
                            // 대각선 낙하
                            const leftBlocked = grid.some(o => o.x === p.x - 1 && o.y === p.y + 1);
                            const rightBlocked = grid.some(o => o.x === p.x + 1 && o.y === p.y + 1);
                            
                            if (!leftBlocked && p.x > 0 && Math.random() > 0.5) p.x--;
                            else if (!rightBlocked && p.x < cols - 1) p.x++;
                        }
                    }
                    
                    ctx.fillStyle = p.color;
                    ctx.fillRect(p.x * currentW, p.y * currentH, currentW - 1, currentH - 1);
                }

                const allFalled = grid.every(p => p.y >= rows - 5); // 거의 다 떨어지면 새로고침
                if (allFalled && grid.length > 0) {
                    setTimeout(() => window.location.reload(), 1500);
                    return;
                }

                animId = requestAnimationFrame(animate);
            }

            const observer = new IntersectionObserver(e => {
                if (e[0].isIntersecting) animate();
                else cancelAnimationFrame(animId);
            });
            observer.observe(canvas);
        } }]
    });
})();
