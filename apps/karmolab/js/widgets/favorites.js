/**
 * 즐겨찾기 모음 — 사이트 파비콘/아이콘으로 빠른 접속
 * - Google Favicon API 또는 Simple Icons CDN 사용
 * - 사용자 추가/삭제 (localStorage)
 */
(function () {
    const STORAGE_KEY = 'toolbox_favorites';
    const VIEW_KEY = 'toolbox_fav_view';
    const FAVICON_API = 'https://www.google.com/s2/favicons?domain=';
    const FAVICON_SZ = '64';

    const DEFAULT_ITEMS = [
        { group: '개발', items: [
            { url: 'https://github.com', label: 'GitHub', icon: 'https://cdn.simpleicons.org/github' },
            { url: 'https://discord.com/developers/applications/', label: 'Discord Developer', icon: 'https://cdn.simpleicons.org/discord' },
            { url: 'https://my.vultr.com', label: 'Vultr', icon: 'https://cdn.simpleicons.org/vultr' },
            { url: 'https://stackoverflow.com', label: 'Stack Overflow', icon: 'https://cdn.simpleicons.org/stackoverflow' },
            { url: 'https://developer.mozilla.org', label: 'MDN', icon: 'https://cdn.simpleicons.org/mdnwebdocs' },
            { url: 'https://www.npmjs.com', label: 'npm', icon: 'https://cdn.simpleicons.org/npm' },
            { url: 'https://codepen.io', label: 'CodePen', icon: 'https://www.google.com/s2/favicons?domain=codepen.io&sz=64' },
            { url: 'https://solved.ac/class?class=5', label: 'solved.ac CLASS', icon: null },
        ]},
        { group: '채용·커리어', items: [
            { url: 'https://blog.maplestory.nexon.com/Employment', label: '메이플 채용', icon: 'https://cdn.simpleicons.org/nexon' },
            { url: 'https://maplecareer.stibee.com', label: '메이플 커리어 레터', icon: 'https://www.google.com/s2/favicons?domain=stibee.com&sz=64' },
            { url: 'https://careers.nexon.com', label: '넥슨 채용', icon: 'https://cdn.simpleicons.org/nexon' },
            { url: 'https://www.nexon-tutorial.com', label: '넥토리얼', icon: 'https://cdn.simpleicons.org/nexon' },
            { url: 'https://www.gamejob.co.kr/User/resumemng/portfolio', label: '게임잡 포트폴리오', icon: null },
            { url: 'https://inditor.co.kr', label: '인디터웹', icon: null },
        ]},
        { group: '메이플', items: [
            { url: 'https://blog.maplestory.nexon.com/Tech', label: '메이플 테크 블로그', icon: 'https://cdn.simpleicons.org/nexon' },
            { url: 'https://blog.maplestory.nexon.com/Tech/Content/10', label: '메이플 테크 (1)', icon: 'https://cdn.simpleicons.org/nexon' },
            { url: 'https://blog.maplestory.nexon.com/Tech/Content/2', label: '메이플 테크 (2)', icon: 'https://cdn.simpleicons.org/nexon' },
            { url: 'https://maplescouter.com/ko', label: '환산주스탯', icon: null },
        ]},
        { group: '검색·AI', items: [
            { url: 'https://www.google.com', label: 'Google', icon: 'https://cdn.simpleicons.org/google' },
            { url: 'https://duckduckgo.com', label: 'DuckDuckGo', icon: 'https://cdn.simpleicons.org/duckduckgo' },
            { url: 'https://chat.openai.com', label: 'ChatGPT', icon: 'https://www.google.com/s2/favicons?domain=chat.openai.com&sz=64' },
            { url: 'https://claude.ai', label: 'Claude', icon: 'https://cdn.simpleicons.org/anthropic' },
            { url: 'https://gemini.google.com', label: 'Gemini', icon: 'https://cdn.simpleicons.org/google' },
            { url: 'https://aistudio.google.com', label: 'AI Studio', icon: 'https://cdn.simpleicons.org/google' },
            { url: 'https://notebooklm.google.com/', label: 'NotebookLM', icon: 'https://cdn.simpleicons.org/google' },
        ]},
        { group: 'AI 아트', items: [
            { url: 'https://pixai.art/', label: 'PixAI', icon: null },
            { url: 'https://tensor.art/', label: 'Tensor.art', icon: null },
            { url: 'https://novelai.net/', label: 'NovelAI', icon: null },
        ]},
        { group: '소셜·미디어', items: [
            { url: 'https://www.youtube.com', label: 'YouTube', icon: 'https://cdn.simpleicons.org/youtube' },
            { url: 'https://music.youtube.com', label: 'YouTube Music', icon: 'https://cdn.simpleicons.org/youtube' },
            { url: 'https://kr.pinterest.com', label: 'Pinterest', icon: 'https://cdn.simpleicons.org/pinterest' },
            { url: 'https://chzzk.naver.com', label: '치지직', icon: null },
            { url: 'https://sooplive.co.kr', label: '숲 (SOOP)', icon: null },
            { url: 'https://x.com', label: 'X (Twitter)', icon: 'https://cdn.simpleicons.org/x' },
            { url: 'https://www.reddit.com', label: 'Reddit', icon: 'https://cdn.simpleicons.org/reddit' },
            { url: 'https://discord.com', label: 'Discord', icon: 'https://cdn.simpleicons.org/discord' },
        ]},
        { group: '서로이웃', items: [
            { url: 'https://orbit3230.github.io', label: 'orbit3230', icon: 'https://orbit3230.github.io/favicon.ico' },
        ]},
        { group: '짝이웃', items: [
            { url: 'https://hyngng.github.io', label: 'HYNGNG', icon: null },
        ]},
        { group: '도구', items: [
            { url: 'https://www.dhlottery.co.kr/main', label: '동행복권', icon: null },
            { url: 'https://www.notion.so', label: 'Notion', icon: 'https://cdn.simpleicons.org/notion' },
            { url: 'https://figma.com', label: 'Figma', icon: 'https://cdn.simpleicons.org/figma' },
            { url: 'https://excalidraw.com', label: 'Excalidraw', icon: 'https://cdn.simpleicons.org/excalidraw' },
            { url: 'https://regex101.com', label: 'Regex101', icon: null },
        ]},
    ];

    function getToolboxToolsGroup() {
        const tools = typeof Toolbox !== 'undefined' && Toolbox.getTools ? Toolbox.getTools() : [];
        const items = tools
            .filter(t => !t.hidden)
            .map(t => ({ type: 'tool', toolId: t.id, label: t.title, icon: t.icon }));
        return items.length ? { group: 'Toolbox', items } : null;
    }

    function getFaviconUrl(item) {
        if (item.icon) return item.icon;
        try {
            const u = new URL(item.url);
            return FAVICON_API + u.hostname + '&sz=' + FAVICON_SZ;
        } catch (_) {
            return FAVICON_API + 'example.com&sz=' + FAVICON_SZ;
        }
    }

    function loadFavorites() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) return JSON.parse(raw);
        } catch (_) {}
        return null;
    }

    function saveFavorites(data) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
        } catch (_) {}
    }

    function getViewMode() {
        try { return localStorage.getItem(VIEW_KEY) || 'icon'; } catch (_) { return 'icon'; }
    }
    function setViewMode(mode) {
        try { localStorage.setItem(VIEW_KEY, mode); } catch (_) {}
    }

    function buildGroups(defaultGroups, customGroups) {
        const merged = [];
        const toolboxGroup = getToolboxToolsGroup();
        if (toolboxGroup) merged.push(toolboxGroup);
        defaultGroups.forEach(g => merged.push({
            group: g.group,
            items: g.items.map(it => ({ ...it, isCustom: false }))
        }));
        if (customGroups && Array.isArray(customGroups)) {
            customGroups.forEach(cg => {
                const existing = merged.find(m => m.group === cg.group);
                const customItems = (cg.items || []).map(it => ({ ...it, isCustom: true }));
                if (existing) {
                    existing.items.push(...customItems);
                } else {
                    merged.push({ group: cg.group, items: customItems });
                }
            });
        }
        return merged;
    }


    Mdd.injectCSS('favorites', `
        .fav-layout { display:flex; flex-direction:column; gap:24px; }
        .fav-group { display:flex; flex-direction:column; gap:12px; }
        .fav-group-title { font-size:var(--font-size-xs); font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.06em; }
        .fav-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(72px, 1fr)); gap:12px; }
        .fav-item { display:flex; flex-direction:column; align-items:center; gap:6px; padding:12px 8px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-md); cursor:pointer; transition:all var(--transition); text-decoration:none; color:inherit; }
        .fav-item:hover { background:var(--bg-hover); border-color:var(--border-hover); transform:translateY(-12px); box-shadow:var(--shadow-float); }
        .fav-item:active { transform:translateY(0); }
        .fav-icon { width:52px; height:52px; border-radius:10px; object-fit:contain; background:var(--bg-secondary); }
        .fav-icon-svg { display:flex; align-items:center; justify-content:center; padding:6px; }
        .fav-icon-svg svg { width:32px; height:32px; stroke:var(--text-secondary); }
        .fav-label { font-size:var(--font-size-2xs); color:var(--text-secondary); text-align:center; max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .fav-actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
        .fav-add-form { display:flex; gap:8px; flex-wrap:wrap; align-items:flex-end; padding:16px; background:var(--bg-tertiary); border:1px dashed var(--border); border-radius:var(--radius-md); }
        .fav-add-form input, .fav-add-form select { flex:1; min-width:180px; padding:8px 12px; font-size:var(--font-size-xs); background:var(--bg-primary); border:1px solid var(--border); border-radius:4px; color:var(--text-primary); }
        .fav-add-form .form-group { display:flex; flex-direction:column; gap:4px; }
        .fav-add-form label { font-size:var(--font-size-2xs); color:var(--text-tertiary); }
        .fav-item-wrap { position:relative; }
        .fav-item-wrap .fav-remove { position:absolute; top:4px; right:4px; width:20px; height:20px; border-radius:50%; background:var(--error); color:#fff; border:none; cursor:pointer; font-size:12px; line-height:1; opacity:0; transition:opacity var(--transition); display:flex; align-items:center; justify-content:center; z-index:1; }
        .fav-item-wrap:hover .fav-remove { opacity:1; }
        .fav-remove:hover { background:#dc2626; }
        .fav-view-toggle-btn { padding:8px; background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-secondary); cursor:pointer; transition:var(--transition); border-radius:4px; display:flex; align-items:center; justify-content:center; }
        .fav-view-toggle-btn:hover { background:var(--bg-hover); color:var(--text-primary); }
        .fav-view-toggle-btn.active { background:var(--accent-subtle); border-color:var(--accent); color:var(--accent); }
        .fav-view-toggle-btn svg { width:18px; height:18px; }
        .fav-view-toggle-btn .fav-view-icon-card { display:none; }
        .fav-view-toggle-btn .fav-view-icon-grid { display:block; }
        .fav-view-toggle-btn[data-view="card"] .fav-view-icon-card { display:block; }
        .fav-view-toggle-btn[data-view="card"] .fav-view-icon-grid { display:none; }
        .fav-top-row { position:relative; display:flex; justify-content:center; align-items:center; margin-bottom:var(--space-md); }
        .fav-top-row .landing-search-wrap { flex:none; width:220px; margin:0; position:relative; }
        .fav-top-row .fav-view-toggle-wrap { position:absolute; right:0; top:50%; transform:translateY(-50%); display:flex; gap:6px; align-items:center; }
        .fav-add-btn { padding:8px; background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-secondary); cursor:pointer; transition:var(--transition); border-radius:4px; display:flex; align-items:center; justify-content:center; }
        .fav-add-btn:hover { background:var(--bg-hover); color:var(--accent); }
        .fav-add-btn svg { width:18px; height:18px; }
        .fav-add-modal-backdrop { position:fixed; inset:0; background:rgba(0,0,0,0.5); z-index:9998; display:flex; align-items:center; justify-content:center; padding:20px; opacity:0; pointer-events:none; transition:opacity var(--transition); }
        .fav-add-modal-backdrop.open { opacity:1; pointer-events:auto; }
        .fav-add-modal { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg); padding:var(--space-lg); max-width:360px; width:100%; box-shadow:var(--shadow-float); }
        .fav-add-modal h3 { font-size:var(--font-size-sm); margin-bottom:var(--space-md); color:var(--text-primary); }
        .fav-add-modal .fav-add-form { display:flex; flex-direction:column; gap:12px; padding:0; border:none; background:none; align-items:stretch; }
        .fav-add-modal .fav-add-form .form-group { display:flex; flex-direction:column; gap:4px; width:100%; }
        .fav-add-modal .fav-add-form label { font-size:var(--font-size-2xs); color:var(--text-tertiary); }
        .fav-add-modal .fav-add-form input, .fav-add-modal .fav-add-form select { width:100%; min-width:0; box-sizing:border-box; padding:8px 12px; font-size:var(--font-size-xs); background:var(--bg-primary); border:1px solid var(--border); border-radius:4px; color:var(--text-primary); }
        .fav-add-modal .fav-add-form .btn { width:100%; margin-top:4px; }
        .fav-top-row .landing-search-wrap .landing-search-icon { position:absolute; left:10px; top:50%; transform:translateY(-50%); width:16px; height:16px; color:var(--text-tertiary); pointer-events:none; flex-shrink:0; }
        .fav-top-row .landing-search-wrap .landing-search { width:100%; padding:9px 12px 9px 36px; font-size:var(--font-size-xs); background:var(--bg-tertiary); border:1px solid var(--border); border-radius:4px; color:var(--text-primary); }
        .fav-top-row .landing-search-wrap .landing-search:focus { outline:none; border-color:var(--accent); }
        .fav-top-row .landing-search-wrap .landing-search::placeholder { color:var(--text-tertiary); }
        .fav-grid.fav-grid-card { grid-template-columns:repeat(auto-fill, minmax(100px, 1fr)); gap:16px; }
        .fav-grid.fav-grid-card .fav-item { padding:16px 12px; }
        .fav-grid.fav-grid-card .fav-icon { width:60px; height:60px; border-radius:12px; }
        .fav-grid.fav-grid-card .fav-icon-svg svg { width:40px; height:40px; }
        .fav-grid.fav-grid-card .fav-label { font-size:var(--font-size-xs); }
    `);

    function buildFavorites(container) {
        Mdd.setMood('happy');
        Mdd.say('자주 가는 곳을 모아뒀어요~ 클릭해서 가봐요!');

        function render() {
            const customNow = loadFavorites();
            const groupsNow = buildGroups(DEFAULT_ITEMS, customNow);
            const esc = Toolbox.escapeHtml;
            const viewMode = getViewMode();
            const isCard = viewMode === 'card';

            container.innerHTML = `
                <div class="fav-layout">
                    <div class="fav-top-row">
                        <div class="landing-search-wrap">
                            <svg class="landing-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                            <input type="text" class="landing-search" placeholder="도구·사이트 검색..." id="favSearch" autocomplete="off">
                        </div>
                        <div class="fav-view-toggle-wrap">
                            <button type="button" class="fav-add-btn" id="fav-add-open" title="추가하기">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                            </button>
                            <button type="button" class="fav-view-toggle-btn ${isCard ? 'active' : ''}" data-view="${isCard ? 'card' : 'icon'}" title="${isCard ? '작게 보기' : '크게 보기'}">
                                <svg class="fav-view-icon fav-view-icon-grid" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
                                <svg class="fav-view-icon fav-view-icon-card" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="7" rx="1"/><rect x="3" y="14" width="18" height="6" rx="1"/></svg>
                            </button>
                        </div>
                    </div>
                    ${groupsNow.map(g => `
                        <div class="fav-group" data-fav-group="${esc(g.group)}">
                            <div class="fav-group-title">${esc(g.group)}</div>
                            <div class="fav-grid ${isCard ? 'fav-grid-card' : ''}">
                                ${g.items.map(it => {
                                    const isTool = it.type === 'tool';
                                    const metaDesc = (it.toolId && Toolbox.getToolMeta && Toolbox.getToolMeta(it.toolId)?.desc) || '';
                                    const searchable = [it.label, g.group, it.url || '', it.toolId || '', metaDesc].join(' ').toLowerCase();
                                    const removeBtn = it.isCustom ? `<button type="button" class="fav-remove" data-group="${esc(g.group)}" data-url="${esc(it.url)}" title="삭제">×</button>` : '';
                                    const iconHtml = isTool
                                        ? `<div class="fav-icon fav-icon-svg"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${it.icon || ''}</svg></div>`
                                        : `<img class="fav-icon" src="${esc(getFaviconUrl(it))}" alt="" loading="lazy" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22%23666%22%3E%3Cpath d=%22M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z%22/%3E%3C/svg%3E'">`;
                                    const linkAttrs = isTool
                                        ? `href="#" class="fav-item" title="${esc(it.label)}" data-tool-id="${esc(it.toolId)}"`
                                        : `href="${esc(it.url)}" class="fav-item" target="_blank" rel="noopener noreferrer" title="${esc(it.label)}"`;
                                    return `
                                    <div class="fav-item-wrap" data-searchable="${esc(searchable)}">
                                        ${removeBtn}
                                        <a ${linkAttrs}>
                                            ${iconHtml}
                                            <span class="fav-label">${esc(it.label)}</span>
                                        </a>
                                    </div>`;
                                }).join('')}
                            </div>
                        </div>
                    `).join('')}
                </div>

                <div class="fav-add-modal-backdrop" id="fav-add-modal">
                    <div class="fav-add-modal" onclick="event.stopPropagation()">
                        <h3>즐겨찾기 추가</h3>
                        <div class="fav-add-form">
                            <div class="form-group">
                                <label for="fav-url">URL</label>
                                <input type="url" id="fav-url" placeholder="https://example.com" class="input">
                            </div>
                            <div class="form-group">
                                <label for="fav-label">이름 (선택)</label>
                                <input type="text" id="fav-label" placeholder="사이트 이름" class="input">
                            </div>
                            <div class="form-group">
                                <label for="fav-icon">아이콘 URL (선택)</label>
                                <input type="url" id="fav-icon" placeholder="https://example.com/favicon.ico" class="input">
                            </div>
                            <div class="form-group">
                                <label for="fav-group">그룹</label>
                                <select id="fav-group" class="input">
                                    <option value="개발">개발</option>
                                    <option value="채용·커리어">채용·커리어</option>
                                    <option value="메이플">메이플</option>
                                    <option value="검색·AI">검색·AI</option>
                                    <option value="AI 아트">AI 아트</option>
                                    <option value="소셜·미디어">소셜·미디어</option>
                                    <option value="서로이웃">서로이웃</option>
                                    <option value="짝이웃">짝이웃</option>
                                    <option value="도구">도구</option>
                                    <option value="기타">기타</option>
                                </select>
                            </div>
                            <button type="button" class="btn btn-primary" id="fav-add-btn">추가</button>
                        </div>
                    </div>
                </div>`;

            const modal = container.querySelector('#fav-add-modal');
            const openAddBtn = container.querySelector('#fav-add-open');
            if (openAddBtn) {
                openAddBtn.onclick = () => modal?.classList.add('open');
            }
            if (modal) {
                modal.onclick = (e) => { if (e.target === modal) modal.classList.remove('open'); };
            }

            container.querySelector('#fav-add-btn').onclick = () => {
                const urlInput = container.querySelector('#fav-url');
                const labelInput = container.querySelector('#fav-label');
                const iconInput = container.querySelector('#fav-icon');
                const groupSelect = container.querySelector('#fav-group');
                const url = (urlInput.value || '').trim();
                if (!url) {
                    Toolbox.showToast('URL을 입력해주세요', 'error');
                    return;
                }
                let label = (labelInput.value || '').trim();
                if (!label) {
                    try {
                        label = new URL(url).hostname.replace(/^www\./, '');
                    } catch (_) {
                        label = url;
                    }
                }
                const iconUrl = (iconInput?.value || '').trim() || null;
                const group = groupSelect.value || '기타';
                const data = loadFavorites() || [];
                let g = data.find(d => d.group === group);
                if (!g) {
                    g = { group, items: [] };
                    data.push(g);
                }
                g.items.push({ url, label, icon: iconUrl });
                saveFavorites(data);
                urlInput.value = '';
                labelInput.value = '';
                if (iconInput) iconInput.value = '';
                modal?.classList.remove('open');
                Toolbox.showToast('추가되었습니다');
                render();
            };

            container.querySelectorAll('.fav-remove').forEach(btn => {
                btn.onclick = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const group = btn.dataset.group;
                    const url = btn.dataset.url;
                    const data = loadFavorites() || [];
                    const g = data.find(d => d.group === group);
                    if (g) {
                        g.items = g.items.filter(it => it.url !== url);
                        if (!g.items.length) data.splice(data.indexOf(g), 1);
                        saveFavorites(data);
                        Toolbox.showToast('삭제되었습니다');
                        render();
                    }
                };
            });

            container.querySelectorAll('.fav-item[data-tool-id]').forEach(a => {
                a.onclick = (e) => {
                    e.preventDefault();
                    const id = a.dataset.toolId;
                    if (id && typeof Toolbox !== 'undefined' && Toolbox.switchPage) Toolbox.switchPage(id);
                };
            });

            const searchInput = container.querySelector('#favSearch');
            if (searchInput) {
                searchInput.oninput = () => {
                    const q = searchInput.value.toLowerCase().trim();
                    container.querySelectorAll('.fav-item-wrap').forEach(wrap => {
                        const match = !q || (wrap.dataset.searchable || '').includes(q);
                        wrap.style.display = match ? '' : 'none';
                    });
                    container.querySelectorAll('.fav-group').forEach(grp => {
                        const visible = grp.querySelectorAll('.fav-item-wrap:not([style*="display: none"])');
                        grp.style.display = visible.length ? '' : 'none';
                    });
                };
            }

            const viewToggleBtn = container.querySelector('.fav-view-toggle-btn');
            if (viewToggleBtn) {
                viewToggleBtn.onclick = () => {
                    const next = viewToggleBtn.dataset.view === 'card' ? 'icon' : 'card';
                    setViewMode(next);
                    viewToggleBtn.dataset.view = next;
                    viewToggleBtn.title = next === 'card' ? '작게 보기' : '크게 보기';
                    viewToggleBtn.classList.toggle('active', next === 'card');
                    container.querySelectorAll('.fav-grid').forEach(grid => {
                        grid.classList.toggle('fav-grid-card', next === 'card');
                    });
                };
            }
        }

        render();
    }

    Toolbox.register({
        id: 'favorites',
        title: '즐겨찾기',
        category: 'feature',
        desc: '자주 가는 사이트와 도구를 모아 빠르게 접속합니다',
        layout: 'wide',
        icon: '<path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>',
        tabs: [{ id: 'fav-main', label: '즐겨찾기', build: buildFavorites }]
    });
})();
