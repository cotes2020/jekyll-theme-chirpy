(function () {
    /* ===== 유틸 ===== */
    function getModelDisplayName(modelId) {
        const all = [
            ...(Gemini.MODELS.gemini || []),
            ...(Gemini.MODELS.geminiImage || []),
            ...(Gemini.MODELS.imagen || [])
        ];
        const found = all.find(m => m.id === modelId);
        return found ? found.name : modelId;
    }

    const formatTimestamp = Toolbox.formatTimestamp;
    const escapeHtml = Toolbox.escapeHtml;

    function showLightbox(imageUrl) {
        let lb = document.getElementById('ilLightbox');
        if (!lb) {
            lb = document.createElement('div');
            lb.id = 'ilLightbox';
            lb.className = 'il-lightbox';
            lb.innerHTML = `
                <img id="ilLightboxImg" src="" alt="Full Size">
                <div class="il-lightbox-actions">
                    <button class="btn btn-accent" id="ilLightboxDl">⬇️ 다운로드</button>
                    <button class="btn btn-ghost" id="ilLightboxClose">닫기</button>
                </div>`;
            lb.onclick = (e) => { if (e.target === lb) lb.classList.remove('open'); };
            document.body.appendChild(lb);
        }
        document.getElementById('ilLightboxImg').src = imageUrl;
        document.getElementById('ilLightboxDl').onclick = () => downloadImage(imageUrl);
        document.getElementById('ilLightboxClose').onclick = () => lb.classList.remove('open');
        lb.classList.add('open');
    }

    function downloadImage(url) {
        const a = document.createElement('a');
        a.href = url;
        a.download = `ai-image-${Date.now()}.png`;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        Toolbox.showToast('다운로드 시작');
    }

    function copyToClipboard(text) {
        if (navigator.clipboard?.writeText) {
            navigator.clipboard.writeText(text).then(() => Toolbox.showToast('클립보드에 복사됨'));
        } else {
            const ta = document.createElement('textarea');
            ta.value = text; document.body.appendChild(ta); ta.select();
            document.execCommand('copy'); document.body.removeChild(ta);
            Toolbox.showToast('클립보드에 복사됨');
        }
    }

    /* ===== CSS 주입 ===== */
    Mdd.injectCSS('imagelib', `
        .il-lib-header {
            display:flex; align-items:center; justify-content:space-between;
            margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid var(--border);
        }
        .il-lib-count { font-size:var(--font-size-sm); color:var(--text-secondary); font-weight:600; }
        .il-lib-grid {
            display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr));
            gap:12px; overflow-y:auto; max-height:calc(100vh - 200px); padding:2px;
        }
        .il-lib-card {
            position:relative; border-radius:var(--radius-md); overflow:hidden;
            border:1px solid var(--border); background:var(--bg-tertiary);
            cursor:pointer; transition:all 0.25s ease; aspect-ratio:1;
        }
        .il-lib-card:hover {
            border-color:var(--accent); transform:translateY(-3px);
            box-shadow:0 8px 28px rgba(0,0,0,0.5);
        }
        .il-lib-card img { width:100%; height:100%; object-fit:cover; display:block; }
        .il-lib-card-overlay {
            position:absolute; inset:0;
            background:linear-gradient(transparent 30%, rgba(0,0,0,0.88) 100%);
            display:flex; flex-direction:column; justify-content:flex-end;
            padding:14px; opacity:0; transition:opacity 0.25s;
        }
        .il-lib-card:hover .il-lib-card-overlay { opacity:1; }
        .il-lib-card-model {
            display:inline-block; font-size:var(--font-size-2xs); padding:2px 8px; border-radius:4px;
            background:var(--accent); color:#fff; font-weight:600;
            margin-bottom:6px; width:fit-content;
        }
        .il-lib-card-prompt {
            font-size:var(--font-size-2xs); color:rgba(255,255,255,0.8); line-height:1.4;
            display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
        }
        .il-lib-card-date {
            font-size:var(--font-size-2xs); color:rgba(255,255,255,0.5); margin-top:4px;
        }

        .il-lib-empty {
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            padding:80px 20px; color:var(--text-tertiary); text-align:center;
        }
        .il-lib-empty-icon { font-size:56px; opacity:0.2; margin-bottom:20px; }
        .il-lib-empty-text { font-size:15px; font-weight:500; margin-bottom:6px; }
        .il-lib-empty-sub { font-size:var(--font-size-xs); opacity:0.6; }

        .il-detail { display:flex; gap:24px; min-height:480px; }
        .il-detail-image {
            flex:1; display:flex; align-items:center; justify-content:center;
            background:var(--bg-tertiary); border-radius:var(--radius-lg);
            border:1px solid var(--border); overflow:hidden; min-height:400px;
        }
        .il-detail-image img {
            max-width:100%; max-height:540px; border-radius:var(--radius-md);
            cursor:zoom-in; transition:transform 0.2s;
        }
        .il-detail-image img:hover { transform:scale(1.02); }
        .il-detail-info {
            width:320px; flex-shrink:0; display:flex; flex-direction:column; gap:16px;
        }
        .il-detail-model-badge {
            display:inline-flex; align-items:center; gap:6px;
            padding:8px 14px; border-radius:var(--radius-sm);
            background:var(--accent-subtle); color:var(--accent);
            font-size:var(--font-size-xs); font-weight:600; width:fit-content;
        }
        .il-detail-date { font-size:var(--font-size-xs); color:var(--text-tertiary); }
        .il-detail-section-label {
            font-size:var(--font-size-xs); font-weight:600; color:var(--text-secondary); margin-bottom:6px;
        }
        .il-detail-prompt {
            font-size:var(--font-size-xs); line-height:1.6; color:var(--text-primary);
            background:var(--bg-tertiary); border:1px solid var(--border);
            border-radius:var(--radius-sm); padding:12px;
            max-height:220px; overflow-y:auto; word-break:break-word; white-space:pre-wrap;
        }
        .il-detail-stats {
            display:flex; gap:16px; font-size:var(--font-size-xs); color:var(--text-tertiary); font-family:monospace;
        }
        .il-detail-actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:auto; }

        .il-lightbox {
            display:none; position:fixed; inset:0; z-index:9999;
            background:rgba(0,0,0,0.92); backdrop-filter:blur(4px);
            align-items:center; justify-content:center; flex-direction:column;
        }
        .il-lightbox.open { display:flex; }
        .il-lightbox img { max-width:92vw; max-height:85vh; border-radius:var(--radius-md); }
        .il-lightbox-actions { margin-top:16px; display:flex; gap:12px; }

        .il-search-bar {
            display:flex; gap:8px; align-items:center; margin-bottom:12px;
        }
        .il-search-bar input {
            flex:1; font-size:var(--font-size-xs); padding:8px 12px; border:1px solid var(--border); border-radius:var(--radius-sm);
            background:var(--bg-primary); color:var(--text-primary); outline:none; font-family:inherit;
        }
        .il-search-bar input:focus { border-color:var(--accent); }
        .il-search-bar input::placeholder { color:var(--text-tertiary); }

        @media (max-width:768px) {
            .il-lib-grid { grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:8px; }
            .il-detail { flex-direction:column; min-height:auto; }
            .il-detail-info { width:100%; }
            .il-detail-image { min-height:280px; }
        }
    `);

    /* ===== 메인 빌드 ===== */
    function buildMain(container) {
        container.innerHTML = `
            <div id="ilGridView">
                <div class="il-lib-header">
                    <span class="il-lib-count" id="ilCount"></span>
                    <button class="btn btn-danger" id="ilClearBtn">🗑️ 전체 삭제</button>
                </div>
                <div class="il-search-bar">
                    <input type="text" id="ilSearchInput" placeholder="프롬프트, 모델명으로 검색...">
                </div>
                <div class="il-lib-grid" id="ilGridContent"></div>
                <div class="il-lib-empty" id="ilEmpty" style="display:none">
                    <div class="il-lib-empty-icon">🖼️</div>
                    <div class="il-lib-empty-text">아직 생성된 이미지가 없습니다</div>
                    <div class="il-lib-empty-sub">이미지 생성에서 이미지를 만들어보세요</div>
                </div>
            </div>
            <div id="ilDetailView" style="display:none">
                <button class="btn btn-ghost" id="ilBackBtn" style="margin-bottom:16px;">← 돌아가기</button>
                <div class="il-detail">
                    <div class="il-detail-image">
                        <img id="ilDetailImg" src="" alt="">
                    </div>
                    <div class="il-detail-info">
                        <div class="il-detail-model-badge" id="ilDetailModel"></div>
                        <div class="il-detail-date" id="ilDetailDate"></div>
                        <div>
                            <div class="il-detail-section-label">📝 프롬프트</div>
                            <div class="il-detail-prompt" id="ilDetailPrompt"></div>
                        </div>
                        <div class="il-detail-stats" id="ilDetailStats"></div>
                        <div class="il-detail-actions" id="ilDetailActions"></div>
                    </div>
                </div>
            </div>`;

        requestAnimationFrame(() => {
            document.getElementById('ilClearBtn').onclick = async () => {
                if (!confirm('모든 이미지를 삭제하시겠습니까?')) return;
                try {
                    await ImageDB.clear();
                    loadGrid();
                    Toolbox.showToast('라이브러리를 비웠습니다.');
                } catch (e) {
                    Toolbox.showToast('삭제 실패', 'error');
                }
            };

            document.getElementById('ilBackBtn').onclick = () => {
                document.getElementById('ilGridView').style.display = '';
                document.getElementById('ilDetailView').style.display = 'none';
            };

            const searchInput = document.getElementById('ilSearchInput');
            let searchDebounce = null;
            if (searchInput) {
                searchInput.addEventListener('input', () => {
                    clearTimeout(searchDebounce);
                    searchDebounce = setTimeout(() => renderGrid(), 200);
                });
            }

            loadGrid();

            window.addEventListener('imagedb-change', () => loadGrid());
        });
    }

    let _allItems = [];

    async function loadGrid() {
        try {
            _allItems = await ImageDB.getAll();
            renderGrid();
        } catch (e) {
            console.error('Library load error:', e);
        }
    }

    function renderGrid() {
        const searchInput = document.getElementById('ilSearchInput');
        const query = (searchInput?.value || '').trim().toLowerCase();
        const items = query
            ? _allItems.filter(item => (item.prompt || '').toLowerCase().includes(query) || (item.modelName || item.model || '').toLowerCase().includes(query))
            : _allItems;

        const countEl = document.getElementById('ilCount');
        const gridEl = document.getElementById('ilGridContent');
        const emptyEl = document.getElementById('ilEmpty');
        const clearBtn = document.getElementById('ilClearBtn');

        if (countEl) countEl.textContent = query ? `${items.length} / ${_allItems.length}장` : `${items.length}장의 이미지`;
        if (clearBtn) clearBtn.style.display = _allItems.length > 0 ? '' : 'none';

        if (items.length === 0) {
            if (gridEl) gridEl.style.display = 'none';
            if (emptyEl) { emptyEl.style.display = ''; const et = emptyEl.querySelector('.il-lib-empty-text'); if (et) et.textContent = query ? '검색 결과가 없습니다' : '아직 생성된 이미지가 없습니다'; }
            return;
        }

        if (gridEl) gridEl.style.display = '';
        if (emptyEl) emptyEl.style.display = 'none';
        gridEl.innerHTML = '';

        items.forEach(item => {
            const card = document.createElement('div');
            card.className = 'il-lib-card';
            card.innerHTML = `
                <img src="${escapeHtml(item.url)}" alt="" loading="lazy">
                <div class="il-lib-card-overlay">
                    <div class="il-lib-card-model">${escapeHtml(item.modelName || item.model || '')}</div>
                    <div class="il-lib-card-prompt">${escapeHtml(item.prompt || '')}</div>
                    <div class="il-lib-card-date">${formatTimestamp(item.timestamp)}</div>
                </div>`;
            card.onclick = () => showDetail(item);
            gridEl.appendChild(card);
        });
    }

    function showDetail(item) {
        document.getElementById('ilGridView').style.display = 'none';
        document.getElementById('ilDetailView').style.display = '';

        const detailImg = document.getElementById('ilDetailImg');
        detailImg.src = item.url;
        detailImg.onclick = () => showLightbox(item.url);
        document.getElementById('ilDetailModel').textContent = '🤖 ' + (item.modelName || item.model || 'Unknown');
        document.getElementById('ilDetailDate').textContent = formatTimestamp(item.timestamp);
        document.getElementById('ilDetailPrompt').textContent = item.prompt || '(프롬프트 없음)';

        const stats = [];
        if (item.tokens) stats.push(`${Number(item.tokens).toLocaleString()} tokens`);
        if (item.elapsed) stats.push(`${item.elapsed}s`);
        document.getElementById('ilDetailStats').textContent = stats.join(' · ') || '';

        const actionsEl = document.getElementById('ilDetailActions');
        actionsEl.innerHTML = '';

        const actions = [
            {
                label: '🔄 프롬프트 재사용', cls: 'btn-accent',
                fn: () => {
                    const promptEl = document.getElementById('igPrompt');
                    if (promptEl) promptEl.value = item.prompt || '';
                    Toolbox.switchPage('imagegen');
                    Toolbox.showToast('프롬프트를 불러왔습니다.');
                }
            },
            {
                label: '📋 프롬프트 복사', cls: '',
                fn: () => copyToClipboard(item.prompt || '')
            },
            {
                label: '⬇️ 다운로드', cls: '',
                fn: () => downloadImage(item.url)
            },
            {
                label: '🗑️ 삭제', cls: 'btn-danger',
                fn: async () => {
                    if (!confirm('이 이미지를 삭제하시겠습니까?')) return;
                    try {
                        await ImageDB.remove(item.id);
                        document.getElementById('ilGridView').style.display = '';
                        document.getElementById('ilDetailView').style.display = 'none';
                        loadGrid();
                        Toolbox.showToast('삭제되었습니다.');
                    } catch (e) {
                        Toolbox.showToast('삭제 실패', 'error');
                    }
                }
            }
        ];

        actions.forEach(({ label, cls, fn }) => {
            const btn = document.createElement('button');
            btn.className = 'btn ' + (cls || 'btn-ghost');
            btn.textContent = label;
            btn.onclick = fn;
            actionsEl.appendChild(btn);
        });
    }

    /* ===== 위젯 등록 ===== */
    Toolbox.register({
        ...Toolbox.getLazyWidgetPublicMeta('imagelib'),
        tabs: [
            { id: 'imagelib-main', label: '라이브러리', build: buildMain }
        ]
    });
})();
