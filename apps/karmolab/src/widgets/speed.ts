// @ts-nocheck
(function() {
    Toolbox.register({
        id: 'speed', title: '속도측정',
        category: 'play',
        desc: '드래그 속도를 측정합니다',
        layout: 'form',
        icon: '<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'app', label: '속도측정', build: function(container) {
            Mdd.linePreset('tool_run', { msg: '속도 측정이에요! 빨리 드래그하세요!' });
            container.innerHTML = `
                <div style="display:flex; flex-direction:column; padding:20px; height:380px; box-sizing:border-box; text-align:center;">
                    <div style="font-size:14px; color:var(--text-secondary); margin-bottom:10px;">💾 [1MB] 블럭을 마우스로 잡고 골인 지점까지 끌고 가세요!</div>
                    <div id="dropZone" style="flex:1; background:rgba(0,0,0,0.3); border:2px dashed #444; border-radius:8px; position:relative; overflow:hidden; display:flex; align-items:center; justify-content:center;">
                        <div id="targetArea" style="position:absolute; right:20px; width:80px; height:80px; background:rgba(0, 200, 0, 0.1); border:2px dashed var(--success); border-radius:8px; display:flex; align-items:center; justify-content:center; color:var(--success); font-size:var(--font-size-xs); font-weight:bold;">GOAL</div>
                        <div id="dragBlock" style="position:absolute; left:20px; width:60px; height:60px; background:var(--accent); border-radius:8px; display:flex; align-items:center; justify-content:center; color:#fff; font-weight:bold; font-size:var(--font-size-xs); cursor:grab; box-shadow:0 4px 6px rgba(0,0,0,0.3); user-select:none;">1 MB</div>
                    </div>
                    <div id="speedResult" style="margin-top:15px; font-size:15px; font-weight:bold; color:var(--text-primary); min-height:20px;"></div>
                </div>
            `;
            const dragBlock = container.querySelector('#dragBlock');
            const targetArea = container.querySelector('#targetArea');
            const dropZone = container.querySelector('#dropZone');
            const result = container.querySelector('#speedResult');

            let startTime = null;
            let isDragging = false;

            dragBlock.onmousedown = (e) => {
                isDragging = true;
                if (!startTime) startTime = performance.now();
                dragBlock.style.cursor = 'grabbing';
                
                const offsetX = e.clientX - dragBlock.getBoundingClientRect().left;
                const offsetY = e.clientY - dragBlock.getBoundingClientRect().top;

                function onMouseMove(moveEvent) {
                    if (!isDragging) return;
                    const r = dropZone.getBoundingClientRect();
                    let x = moveEvent.clientX - r.left - offsetX;
                    let y = moveEvent.clientY - r.top - offsetY;

                    // 범위 제한
                    x = Math.max(0, Math.min(x, r.width - 60));
                    y = Math.max(0, Math.min(y, r.height - 60));

                    dragBlock.style.left = `${x}px`;
                    dragBlock.style.top = `${y}px`;

                    // Goal 충돌 검사
                    const tr = targetArea.getBoundingClientRect();
                    const br = dragBlock.getBoundingClientRect();

                    if (br.right > tr.left && br.left < tr.right && br.bottom > tr.top && br.top < tr.bottom) {
                        isDragging = false;
                        dragBlock.style.cursor = 'default';
                        document.removeEventListener('mousemove', onMouseMove);
                        document.removeEventListener('mouseup', onMouseUp);

                        const endTime = performance.now();
                        const tookMs = endTime - startTime;
                        const speed = 1000 / tookMs; // 1MB / sec
                        
                        result.innerHTML = `이동 시간: 무려 <span style="color:var(--warning)">${tookMs.toFixed(0)} ms</span>!<br>당신의 수동 손목 속도는 <span style="color:var(--success)">${speed.toFixed(2)} MB/s</span> 이에요!`;
                        startTime = null; // 리셋
                    }
                }

                function onMouseUp() {
                    isDragging = false;
                    dragBlock.style.cursor = 'grab';
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                }

                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            };
        } }]
    });
})();
