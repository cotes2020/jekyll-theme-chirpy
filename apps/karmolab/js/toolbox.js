/**
 * KarmoLab — 도구 레지스트리 기반 모듈 시스템
 *
 * ┌─ 아키텍처 ─────────────────────────────────────────────────┐
 * │                                                            │
 * │  index.html ─→ toolbox.js (코어)                           │
 * │                  ├─ 랜딩 페이지 (히어로 + 즐겨찾기 CTA)       │
 * │                  ├─ 사이드바 (도구/놀이/기타)               │
 * │                  ├─ 검색, breadcrumb, 테마, 사용량 추적      │
 * │                  └─ 도전과제/뱃지/진행도 시스템              │
 * │               ─→ mdd.js (마스코트)                          │
 * │                  ├─ 이미지 마스코트 (12 감정)               │
 * │                  ├─ 말풍선, 바운스 인터랙션                  │
 * │                  └─ 호감도/스토리 진행 시스템                │
 * │               ─→ gemini.js (AI API)                        │
 * │               ─→ widgets/*.js (개별 도구)                   │
 * │                                                            │
 * │  카테고리:  tool (도구)  /  play (놀이)  /  null (기타)  │
 * └────────────────────────────────────────────────────────────┘
 *
 * 새 도구 추가 방법:
 * 1. widgets/ 폴더에 새 JS 파일 생성
 * 2. widgets-manifest.js(boot) + widgets-lazy-meta.js(지연 메타 단일 출처)
 * 3. Toolbox.register({ id, title, icon, category, desc, hidden?, tabs }) 호출
 *    - icon: SVG path 문자열 (viewBox 0 0 24 24 기준)
 *    - category: 'tool' | 'play' | null
 *    - desc: 한 줄 설명 (검색·즐겨찾기용)
 *    - hidden: true면 사이드바 비표시 (user 등)
 *    - tabs: [{ id, label, build(container) }]
 *
 * 마스코트 연동:
 *   Mdd.setMood('happy')   — 감정 변경
 *   Mdd.say('메시지')       — 말풍선 표시
 *   Mdd.linePreset('success', { msg?, mood?, duration? }) — 티메토 대사 프리셋 (`mdd.js`의 LINE_PRESETS)
 *   Mdd.bounce()           — 바운스 애니메이션
 *   Mdd.addAffection(n)    — 호감도 증가 (스토리 해금 트리거)
 */

const Toolbox = (() => {
    const tools = [];

    /* ===== 카테고리 & 메타데이터 ===== */

    const CATEGORIES = [
        { id: 'tool', label: '도구', icon: '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94L6.73 20.15a2.1 2.1 0 0 1-3-3l6.72-6.72a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>' },
        { id: 'play', label: '놀이', icon: '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 12h4m-2-2v4"/><circle cx="15" cy="11" r="1"/><circle cx="18" cy="13" r="1"/>' },
    ];

    /** 위젯별 메타데이터 (category, desc, hidden 등) — 각 위젯 register에서 정의 */
    function getToolMeta(id) {
        const t = tools.find(x => x.id === id);
        return t ? { category: t.category, desc: t.desc, hidden: t.hidden } : null;
    }
    const SIDEBAR_GROUP_KEY = 'toolbox-sidebar-groups';
    const SIDEBAR_COLLAPSED_KEY = 'toolbox_sidebar_collapsed';
    const LAST_PAGE_KEY = 'toolbox_last_page';

    function getGroupState() {
        try {
            const raw = localStorage.getItem(SIDEBAR_GROUP_KEY);
            if (raw) return JSON.parse(raw);
        } catch (_) {}
        return { tool: true, play: false };
    }

    function setGroupState(state) {
        try { localStorage.setItem(SIDEBAR_GROUP_KEY, JSON.stringify(state)); } catch (_) {}
    }

    /* ===== Public API ===== */

    const lazyLoadPromises = new Map();

    function register(config) {
        const deferredIdx = tools.findIndex(t => t.id === config.id && t._deferred);
        if (deferredIdx >= 0) {
            tools[deferredIdx] = { ...config, _deferred: false };
            rebuildToolPageIfInDom(config.id);
            return;
        }
        if (tools.some(t => t.id === config.id)) return;
        tools.push(config);
    }

    /** 사이드바·초기화용 — 스크립트는 첫 방문 시 loadDeferredWidget에서 로드 */
    function registerDeferred(stub) {
        const { lazyScriptPaths, ...rest } = stub;
        tools.push({
            ...rest,
            _deferred: true,
            lazyScriptPaths: lazyScriptPaths || [],
            tabs: [{
                id: '__lazy',
                label: '…',
                build(container) {
                    container.innerHTML = '<p class="tb-lazy-loading" style="padding:32px;text-align:center;color:var(--text-secondary);">불러오는 중…</p>';
                },
            }],
        });
    }

    function rebuildToolPageIfInDom(pageId) {
        const toolPages = document.getElementById('tool-pages');
        if (!toolPages) return;
        const tool = tools.find(t => t.id === pageId);
        if (!tool || !tool.tabs) return;
        const old = document.getElementById('page-' + pageId);
        if (!old) return;
        const wasActive = old.classList.contains('active');
        const nu = buildToolPage(tool);
        if (wasActive) nu.classList.add('active');
        old.replaceWith(nu);
    }

    function getWidgetScriptBase() {
        const b = typeof window !== 'undefined' && window.KARMOLAB_WIDGET_SCRIPT_BASE;
        if (b) return b;
        try {
            const origin = location.origin || '';
            return origin + '/apps/karmolab/js/widgets/';
        } catch (_) {
            return '/apps/karmolab/js/widgets/';
        }
    }

    const widgetScriptsLoaded = new Set();
    const widgetScriptsLoading = new Map();

    function loadScriptOnce(src) {
        if (widgetScriptsLoaded.has(src)) return Promise.resolve();
        if (widgetScriptsLoading.has(src)) return widgetScriptsLoading.get(src);
        const p = new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = src;
            s.async = false;
            s.onload = () => {
                widgetScriptsLoaded.add(src);
                widgetScriptsLoading.delete(src);
                resolve();
            };
            s.onerror = () => {
                widgetScriptsLoading.delete(src);
                reject(new Error('load failed: ' + src));
            };
            document.body.appendChild(s);
        });
        widgetScriptsLoading.set(src, p);
        return p;
    }

    function loadDeferredWidget(pageId) {
        const tool = tools.find(t => t.id === pageId && t._deferred);
        if (!tool) return Promise.resolve();
        if (lazyLoadPromises.has(pageId)) return lazyLoadPromises.get(pageId);

        const paths = tool.lazyScriptPaths;
        if (!paths || !paths.length) {
            return Promise.resolve();
        }

        const base = getWidgetScriptBase();
        const p = (async () => {
            let waitIdx = (window.KARMOLAB_WIDGET_LOADER_WAIT || []).length;
            for (let i = 0; i < paths.length; i++) {
                const url = base + paths[i] + '.js';
                await loadScriptOnce(url);
                const w = window.KARMOLAB_WIDGET_LOADER_WAIT || [];
                if (w.length > waitIdx) {
                    await Promise.allSettled(w.slice(waitIdx));
                    waitIdx = w.length;
                }
            }
        })()
            .finally(() => {
                lazyLoadPromises.delete(pageId);
            })
            .catch((err) => {
                try { showToast('도구 로드 실패', 'error', err); } catch (_) {}
                throw err;
            });

        lazyLoadPromises.set(pageId, p);
        return p;
    }

    function kickLazyLoad(pageId) {
        return loadDeferredWidget(pageId);
    }

    /** 지연 위젯용 — lazy-meta에 정의된 공개 필드만 (lazyScriptPaths 제외). 위젯 register 시 스프레드 */
    function getLazyWidgetPublicMeta(id) {
        const m = typeof window !== 'undefined' && window.KARMOLAB_LAZY_META_BY_ID && window.KARMOLAB_LAZY_META_BY_ID[id];
        if (!m) {
            console.warn('[KarmoLab] getLazyWidgetPublicMeta: 정의 없음 —', id);
            return { id };
        }
        const { lazyScriptPaths: _paths, ...rest } = m;
        return rest;
    }

    function setNotifyInvokeDebugPayload(payload) {
        if (typeof window.__karmolabSetNotifyInvokeDebug === 'function') {
            window.__karmolabSetNotifyInvokeDebug(payload);
            return;
        }
        try {
            const pre = document.getElementById('karmolab-notify-debug-pre');
            const sum = document.querySelector('.karmolab-notify-debug-summary');
            const det = document.querySelector('.karmolab-notify-debug');
            const line = JSON.stringify(payload);
            if (pre) pre.textContent = JSON.stringify(payload, null, 2);
            if (sum) sum.textContent = line.length > 100 ? line.slice(0, 97) + '…' : line;
            if (det) det.open = true;
        } catch (_) {}
    }

    function injectDesktopBadge() {
        if (typeof window === 'undefined' || !window.__KARMOLAB_DESKTOP__) return;
        const left = document.querySelector('.header-bar-left');
        if (left && !left.querySelector('.karmolab-desktop-chrome')) {
            const row = document.createElement('span');
            row.className = 'karmolab-desktop-chrome';
            row.setAttribute('aria-label', '데스크톱 앱 모드');
            const span = document.createElement('span');
            span.className = 'karmolab-desktop-badge';
            span.textContent = '앱';
            span.title = 'Tauri 데스크톱 앱에서 실행 중입니다. 웹에서는 이 배지가 보이지 않습니다.';
            const browserA = document.createElement('a');
            browserA.className = 'karmolab-open-browser';
            browserA.href = 'https://mascari4615.github.io/karmolab/';
            browserA.target = '_blank';
            browserA.rel = 'noopener noreferrer';
            browserA.textContent = '브라우저';
            browserA.title = '기본 브라우저에서 KarmoLab 열기';
            row.appendChild(span);
            row.appendChild(browserA);
            left.appendChild(row);
        }
    }

    function isDesktopApp() {
        return typeof window !== 'undefined' && !!window.__KARMOLAB_DESKTOP__;
    }

    function mirrorToastToDesktop(msg, type, detailText) {
        if (!isDesktopApp()) return;
        if (type !== 'error' && type !== 'success') return;
        if (type === 'success') {
            const m = String(msg);
            if (m.includes('클립보드') || m.includes('코드 테마')) return;
        }
        const invokeFn = window.__TAURI__?.core?.invoke;
        if (typeof invokeFn !== 'function') return;
        const title = String(msg).trim();
        if (!title) return;
        let body = typeof detailText === 'string' ? detailText.trim() : '';
        if (body.length > 240) body = body.slice(0, 237) + '…';
        const payload = { title: title.slice(0, 120), body: body || 'KarmoLab' };
        if (type === 'error') payload.sound = 'Mail';
        setNotifyInvokeDebugPayload(payload);
        invokeFn('desktop_notify', payload).catch(function () {});
    }

    function init() {
        const sidebarNav = document.getElementById('sidebar-nav');
        const mobileNav = document.getElementById('mobile-nav');
        const toolPages = document.getElementById('tool-pages');
        const hiddenSet = new Set(tools.filter(t => t.hidden).map(t => t.id));
        const groupState = getGroupState();

        function addNavItem(container, tool) {
            const a = document.createElement('a');
            a.className = 'nav-item';
            a.dataset.page = tool.id;
            a.title = tool.title;
            a.innerHTML = `<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${tool.icon}</svg><span class="nav-item-text">${tool.title}</span>`;
            a.onclick = () => switchPage(tool.id);
            container.appendChild(a);
        }

        function addMobileNavItem(tool) {
            const m = document.createElement('a');
            m.className = 'nav-item';
            m.dataset.page = tool.id;
            m.textContent = tool.title;
            m.onclick = () => switchPage(tool.id);
            mobileNav.appendChild(m);
        }

        // Mobile home button
        const mHome = document.createElement('a');
        mHome.className = 'nav-item active';
        mHome.dataset.page = 'home';
        mHome.textContent = '홈';
        mHome.onclick = () => switchPage('home');
        mobileNav.appendChild(mHome);

        function buildSidebarGroup(cat) {
            const catTools = tools
                .filter(t => !hiddenSet.has(t.id) && t.category === cat.id)
                .sort((a, b) => (a.title || '').localeCompare(b.title || '', 'ko-KR'));
            if (!catTools.length) return;

            const isOpen = groupState[cat.id] !== undefined ? groupState[cat.id] : (cat.id === 'tool');
            const wrap = document.createElement('div');
            wrap.className = 'sidebar-group';
            const trigger = document.createElement('button');
            trigger.type = 'button';
            trigger.className = 'sidebar-group-trigger' + (isOpen ? ' open' : '');
            trigger.setAttribute('aria-expanded', isOpen);
            trigger.innerHTML = '<span class="chevron" aria-hidden="true"></span><span class="sidebar-group-label">' + cat.label + '</span>';
            const body = document.createElement('div');
            body.className = 'sidebar-group-body' + (isOpen ? ' open' : '');
            catTools.forEach(tool => addNavItem(body, tool));
            trigger.onclick = () => {
                const open = body.classList.toggle('open');
                trigger.classList.toggle('open', open);
                trigger.setAttribute('aria-expanded', open);
                setGroupState({ ...getGroupState(), [cat.id]: open });
            };
            wrap.appendChild(trigger);
            wrap.appendChild(body);
            sidebarNav.appendChild(wrap);
        }

        CATEGORIES.forEach(cat => buildSidebarGroup(cat));

        // Uncategorized tools
        const uncategorized = tools
            .filter(t => !hiddenSet.has(t.id) && !t.category)
            .sort((a, b) => (a.title || '').localeCompare(b.title || '', 'ko-KR'));
        if (uncategorized.length) {
            const wrap = document.createElement('div');
            wrap.className = 'sidebar-group';
            const trigger = document.createElement('button');
            trigger.type = 'button';
            trigger.className = 'sidebar-group-trigger';
            trigger.innerHTML = '<span class="chevron" aria-hidden="true"></span><span class="sidebar-group-label">기타</span>';
            const body = document.createElement('div');
            body.className = 'sidebar-group-body';
            uncategorized.forEach(tool => addNavItem(body, tool));
            trigger.onclick = () => {
                const open = body.classList.toggle('open');
                trigger.classList.toggle('open', open);
            };
            wrap.appendChild(trigger);
            wrap.appendChild(body);
            sidebarNav.appendChild(wrap);
        }

        // Build landing page
        toolPages.appendChild(buildLanding());

        // Build tool pages (가나다순)
        const sortedTools = [...tools].sort((a, b) => (a.title || '').localeCompare(b.title || '', 'ko-KR'));
        sortedTools.forEach(tool => {
            if (!hiddenSet.has(tool.id)) addMobileNavItem(tool);
            toolPages.appendChild(buildToolPage(tool));
        });

        document.getElementById('userPageBtn')?.addEventListener('click', () => switchPage('user'));

        window.addEventListener('gemini-active-profile-changed', () => {
            const name = typeof Gemini !== 'undefined' ? (Gemini.getActiveProfileName() || '기본') : '-';
            const cb = document.getElementById('cbActiveProfileName');
            if (cb) cb.textContent = name;
            const ig = document.getElementById('igActiveProfileName');
            if (ig) ig.textContent = name;
        });

        const hashPage = location.hash ? location.hash.slice(1) : null;
        const lastPage = (() => { try { return localStorage.getItem(LAST_PAGE_KEY); } catch (_) { return null; } })();
        const isValidPage = (id) => id === 'home' || id === 'user' || tools.some(t => t.id === id);
        const initialPage = (hashPage && isValidPage(hashPage))
            ? hashPage
            : (lastPage && isValidPage(lastPage) ? lastPage : 'home');

        switchPage(initialPage, { pushHistory: false });
        history.replaceState({ pageId: initialPage }, '', location.pathname + (location.search || '') + '#' + initialPage);

        window.addEventListener('popstate', () => {
            const pageId = pageIdFromHash();
            if (isValidPage(pageId)) switchPage(pageId, { pushHistory: false });
        });

        document.getElementById('sidebar')?.classList.add('collapsed');

        injectDesktopBadge();

        const serverDot = document.getElementById('server-status-dot');
        if (serverDot && typeof getServerBase === 'function') {
            function updateServerDot() {
                const base = getServerBase();
                serverDot.classList.remove('server-status-ok', 'server-status-offline', 'server-status-none');
                if (!base) {
                    serverDot.classList.add('server-status-none');
                    serverDot.title = '서버 미설정 (서버 모니터에서 URL 입력)';
                    return;
                }
                const url = base.replace(/\/$/, '') + '/api/status';
                fetch(url).then(r => r.ok ? r.json() : Promise.reject()).then(() => {
                    serverDot.classList.add('server-status-ok');
                    serverDot.title = '서버 연결됨 (클릭: 서버 모니터)';
                }).catch(() => {
                    serverDot.classList.add('server-status-offline');
                    serverDot.title = '서버 연결 실패 (클릭: 서버 모니터)';
                });
            }
            updateServerDot();
            const serverPollInterval = setInterval(updateServerDot, 60000);
            serverDot.addEventListener('click', () => { if (tools.some(t => t.id === 'servermonitor')) switchPage('servermonitor'); });
            serverDot.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); serverDot.click(); } });
        }
    }

    function toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        if (!sidebar) return;
        const collapsed = sidebar.classList.toggle('collapsed');
        try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? '1' : '0'); } catch (_) {}
        const btn = document.getElementById('sidebarToggle');
        if (btn) {
            btn.setAttribute('aria-label', collapsed ? '사이드바 펼치기' : '사이드바 접기');
            btn.title = collapsed ? '사이드바 펼치기' : '사이드바 접기';
        }
    }

    /* ===== Landing Page Builder ===== */

    function buildLanding() {
        const landing = document.createElement('div');
        landing.className = 'landing-page';
        landing.id = 'page-home';

        const hero = document.createElement('div');
        hero.className = 'landing-hero';
        hero.innerHTML = `
            <p class="landing-subtitle">KarmoLab</p>
            <h1 class="landing-title">KarmoLab</h1>
            <p class="landing-tagline">삶을 섞고 술을 바꿀 시간</p>
        `;
        landing.appendChild(hero);

        const cta = document.createElement('div');
        cta.className = 'landing-cta';
        cta.innerHTML = `
            <div class="landing-cta-grid">
                <button type="button" class="landing-cta-card" onclick="Toolbox.switchPage('favorites')">
                    <div class="landing-cta-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg></div>
                    <div class="landing-cta-card-title">즐겨찾기</div>
                    <div class="landing-cta-card-desc">자주 쓰는 도구를 모아봐요</div>
                </button>
                <button type="button" class="landing-cta-card" onclick="Toolbox.switchPage('docs')">
                    <div class="landing-cta-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg></div>
                    <div class="landing-cta-card-title">문서</div>
                    <div class="landing-cta-card-desc">API 레퍼런스 & 가이드</div>
                </button>
            </div>
            <p class="landing-cta-hint">왼쪽 사이드바에서 도구를 선택하세요</p>
        `;
        landing.appendChild(cta);

        return landing;
    }

    /* ===== Navigation ===== */

    function pageIdFromHash() {
        const h = location.hash ? location.hash.slice(1) : '';
        return h || 'home';
    }

    function switchPage(pageId, opts = {}) {
        const { pushHistory = true } = opts;
        const base = location.pathname + (location.search || '');
        const urlWithHash = base + '#' + pageId;
        if (pushHistory) {
            history.pushState({ pageId }, '', urlWithHash);
        }
        const landing = document.getElementById('page-home');
        const allPages = document.querySelectorAll('.tool-page');
        const allNav = document.querySelectorAll('.nav-item');
        const sidebarHome = document.getElementById('sidebarHome');
        const breadcrumb = document.getElementById('breadcrumb');

        const toolForPage = tools.find(t => t.id === pageId);
        if (toolForPage && toolForPage._deferred) {
            kickLazyLoad(pageId);
        }

        allPages.forEach(p => p.classList.remove('active'));
        allNav.forEach(n => n.classList.remove('active'));
        if (sidebarHome) sidebarHome.classList.remove('active');
        if (landing) landing.classList.remove('active');

        if (pageId === 'home') {
            if (landing) landing.classList.add('active');
            if (sidebarHome) sidebarHome.classList.add('active');
            document.querySelectorAll('[data-page="home"]').forEach(n => n.classList.add('active'));
            document.getElementById('pageTitle').textContent = 'KarmoLab';
            if (breadcrumb) breadcrumb.innerHTML = '';
            try { localStorage.setItem(LAST_PAGE_KEY, 'home'); } catch (_) {}
            if (typeof Mdd !== 'undefined') {
                Mdd.linePreset('home_hub');
            }
            return;
        }

        const page = document.getElementById('page-' + pageId);
        if (page) {
            page.classList.add('active');
            try { localStorage.setItem(LAST_PAGE_KEY, pageId); } catch (_) {}
        }
        document.querySelectorAll(`[data-page="${pageId}"]`).forEach(n => n.classList.add('active'));

        const userBtn = document.getElementById('userPageBtn');
        if (userBtn) userBtn.classList.toggle('active', pageId === 'user');

        const tool = tools.find(t => t.id === pageId);
        if (tool) {
            document.getElementById('pageTitle').textContent = tool.title;
            if (breadcrumb && tool.category) {
                const cat = CATEGORIES.find(c => c.id === tool.category);
                breadcrumb.innerHTML = `
                    <button class="breadcrumb-link" onclick="Toolbox.switchPage('home')">KarmoLab</button>
                    <span class="breadcrumb-sep">/</span>
                    <span class="breadcrumb-current">${cat ? cat.label : ''}</span>
                `;
            } else if (breadcrumb) {
                breadcrumb.innerHTML = `<button class="breadcrumb-link" onclick="Toolbox.switchPage('home')">KarmoLab</button>`;
            }
        }
    }

    function switchTab(btn, tabId) {
        if (typeof btn === 'string') {
            tabId = btn;
            btn = document.querySelector(`[data-tab-id="${tabId}"]`);
            if (!btn) return;
        }
        const tabRow = btn.closest('.tab-row');
        const page = btn.closest('.tool-page');
        tabRow.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        page.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('panel-' + tabId)?.classList.add('active');
    }

    /* ===== Page Builder ===== */

    function buildToolPage(tool) {
        const div = document.createElement('div');
        div.className = 'tool-page';
        if (tool.layout) div.classList.add('layout-' + tool.layout);
        div.id = 'page-' + tool.id;

        if (tool.noHero !== true) {
            const hero = document.createElement('div');
            hero.className = 'tool-page-hero';
            hero.innerHTML =
                `<h1 class="tool-page-hero-title">${tool.title}</h1>` +
                (tool.desc ? `<p class="tool-page-hero-desc">${tool.desc}</p>` : '');
            div.appendChild(hero);
        }

        if (tool.tabs.length > 1) {
            const tabRow = document.createElement('div');
            tabRow.className = 'tab-row';
            tool.tabs.forEach((tab, i) => {
                const btn = document.createElement('button');
                btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
                btn.dataset.tabId = tab.id;
                btn.textContent = tab.label;
                btn.onclick = function () { switchTab(this, tab.id); };
                tabRow.appendChild(btn);
            });
            div.appendChild(tabRow);
        }

        tool.tabs.forEach((tab, i) => {
            const panel = document.createElement('div');
            panel.className = 'tab-panel' + (i === 0 ? ' active' : '');
            panel.id = 'panel-' + tab.id;
            tab.build(panel);
            div.appendChild(panel);
        });

        return div;
    }

    /* ===== Shared Helpers ===== */

    function showToast(msg, type = 'success', detail) {
        const t = document.getElementById('statusToast');
        if (!t) return;
        const hasDetail = detail !== undefined && detail !== null && detail !== '';
        const detailText = typeof detail === 'string' ? detail : (detail && detail.message) ? detail.message + (detail.stack ? '\n' + detail.stack : '') : '';
        if (hasDetail && detailText) {
            t.className = 'status-toast visible has-detail ' + type;
            const fullText = msg + '\n\n' + detailText;
            t.innerHTML = '<span class="status-toast-msg">' + escapeHtml(msg) + '</span><button type="button" class="status-toast-copy" title="복사">📋</button>';
            t.onclick = null;
            const copyBtn = t.querySelector('.status-toast-copy');
            if (copyBtn) {
                copyBtn.onclick = function (ev) {
                    ev.stopPropagation();
                    if (navigator.clipboard?.writeText) {
                        navigator.clipboard.writeText(fullText).then(() => showToast('클립보드에 복사됨'));
                    } else {
                        const ta = document.createElement('textarea');
                        ta.value = fullText;
                        document.body.appendChild(ta);
                        ta.select();
                        document.execCommand('copy');
                        document.body.removeChild(ta);
                        showToast('클립보드에 복사됨');
                    }
                };
            }
            t.style.pointerEvents = 'auto';
            clearTimeout(t._toastHide);
            t._toastHide = setTimeout(() => {
                t.classList.remove('visible');
                t.onclick = null;
                t.style.pointerEvents = '';
            }, 5000);
            mirrorToastToDesktop(msg, type, detailText);
        } else {
            t.textContent = msg;
            t.className = 'status-toast visible ' + type;
            t.onclick = null;
            t.style.pointerEvents = '';
            clearTimeout(t._toastHide);
            t._toastHide = setTimeout(() => t.classList.remove('visible'), 2500);
            mirrorToastToDesktop(msg, type, '');
        }
    }

    function escapeHtml(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function formatTimestamp(ts) {
        const d = new Date(ts);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function showLightbox(imageUrl) {
        let overlay = document.querySelector('.tb-lightbox-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'tb-lightbox-overlay';
            overlay.onclick = () => overlay.remove();
            document.body.appendChild(overlay);
        }
        overlay.innerHTML = `<img src="${escapeHtml(imageUrl)}" alt="확대 이미지">`;
    }

    function displayResult(prefix, title, content, timeTaken, isError = false) {
        const box = document.getElementById(prefix + 'Result');
        const label = document.getElementById(prefix + 'ResultLabel');
        const area = document.getElementById(prefix + 'ResultContent');
        if (label) label.textContent = title + (timeTaken ? ` · ${timeTaken.toFixed(2)}s` : '');
        if (label) label.className = 'result-label ' + (isError ? 'error' : 'success');
        if (area) area.textContent = content;
        if (box) box.classList.add('visible');
    }

    function copyResult(contentId) {
        const text = document.getElementById(contentId).textContent;
        if (navigator.clipboard?.writeText) {
            navigator.clipboard.writeText(text).then(() => showToast('클립보드에 복사됨'));
        } else {
            const ta = document.createElement('textarea');
            ta.value = text; document.body.appendChild(ta); ta.select();
            document.execCommand('copy'); document.body.removeChild(ta);
            showToast('클립보드에 복사됨');
        }
    }

    function toggleCollapsible(trigger) {
        trigger.classList.toggle('open');
        trigger.nextElementSibling.classList.toggle('open');
    }

    function field(container, { tag = 'textarea', id, label, placeholder, type, topRight, mono }) {
        const g = document.createElement('div');
        g.className = 'field-group';
        if (topRight) {
            const row = document.createElement('div');
            row.className = 'field-row';
            row.style.marginBottom = '6px';
            const lbl = document.createElement('label');
            lbl.className = 'field-label'; lbl.style.marginBottom = '0';
            lbl.htmlFor = id; lbl.textContent = label;
            row.appendChild(lbl); row.appendChild(topRight);
            g.appendChild(row);
        } else {
            const lbl = document.createElement('label');
            lbl.className = 'field-label'; lbl.htmlFor = id; lbl.textContent = label;
            g.appendChild(lbl);
        }
        const el = document.createElement(tag);
        el.id = id; el.placeholder = placeholder || '';
        if (type) el.type = type;
        if (mono) el.className = 'mono-input';
        g.appendChild(el);
        container.appendChild(g);
        return el;
    }

    function resultBox(container, prefix) {
        const box = document.createElement('div');
        box.className = 'result-box'; box.id = prefix + 'Result';
        box.innerHTML = `<div class="result-header"><span class="result-label" id="${prefix}ResultLabel">결과</span><button class="btn-ghost" onclick="Toolbox.copyResult('${prefix}ResultContent')">복사</button></div><pre class="result-content" id="${prefix}ResultContent"></pre>`;
        container.appendChild(box);
    }

    function button(container, { text, onclick, style }) {
        const btn = document.createElement('button');
        btn.className = 'btn btn-primary'; btn.textContent = text; btn.onclick = onclick;
        if (style) btn.setAttribute('style', style);
        container.appendChild(btn);
    }

    function select(container, { id, label, options, onChange }) {
        const g = document.createElement('div');
        g.className = 'field-group';
        const lbl = document.createElement('label');
        lbl.className = 'field-label'; lbl.htmlFor = id; lbl.textContent = label;
        g.appendChild(lbl);
        const sel = document.createElement('select');
        sel.id = id;
        options.forEach(o => {
            const opt = document.createElement('option');
            opt.value = o.value; opt.textContent = o.label;
            sel.appendChild(opt);
        });
        if (onChange) sel.onchange = onChange;
        g.appendChild(sel);
        container.appendChild(g);
        return sel;
    }

    /* ===== 테마 (라이트/다크) ===== */
    const THEME_KEY = 'toolbox_theme';

    function getTheme() { return localStorage.getItem(THEME_KEY) || 'dark'; }

    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(THEME_KEY, theme);
    }

    function toggleTheme() {
        const next = getTheme() === 'dark' ? 'light' : 'dark';
        setTheme(next);
    }

    /* ===== 배경 테마 (mesh/gradient) ===== */
    const BG_THEME_KEY = 'toolbox_bg_theme';
    const BG_THEMES = [
        { id: 'blue-magenta', label: '블루 매젠타' },
        { id: 'mesh-dots', label: '메쉬 도트' },
        { id: 'aurora', label: '오로라' },
        { id: 'subtle', label: '은은한' },
        { id: 'minimal', label: '미니멀' },
    ];

    function getBgTheme() {
        const saved = localStorage.getItem(BG_THEME_KEY);
        if (saved && BG_THEMES.some(t => t.id === saved)) return saved;
        return 'blue-magenta';
    }

    function setBgTheme(bgId) {
        document.documentElement.setAttribute('data-bg', bgId);
        localStorage.setItem(BG_THEME_KEY, bgId);
    }

    function getBgThemes() { return [...BG_THEMES]; }

    /* ===== Prism 코드 테마 ===== */
    const PRISM_THEME_KEY = 'toolbox_prism_theme';
    const PRISM_BASE = '/apps/karmolab/js/vendor/prism/themes-cdn';
    const PRISM_EXT = '/apps/karmolab/js/vendor/prism/themes-ext';

    const PRISM_THEMES = [
        { id: 'tomorrow', label: 'Tomorrow Night', url: `${PRISM_BASE}/prism-tomorrow.min.css` },
        { id: 'dracula', label: 'Dracula', url: `${PRISM_EXT}/prism-dracula.min.css` },
        { id: 'one-dark', label: 'One Dark', url: `${PRISM_EXT}/prism-one-dark.min.css` },
        { id: 'nord', label: 'Nord', url: `${PRISM_EXT}/prism-nord.min.css` },
        { id: 'material-dark', label: 'Material Dark', url: `${PRISM_EXT}/prism-material-dark.min.css` },
        { id: 'vsc-dark-plus', label: 'VS Dark+', url: `${PRISM_EXT}/prism-vsc-dark-plus.min.css` },
        { id: 'okaidia', label: 'Okaidia', url: `${PRISM_BASE}/prism-okaidia.min.css` },
        { id: 'twilight', label: 'Twilight', url: `${PRISM_BASE}/prism-twilight.min.css` },
        { id: 'prism', label: 'Default (라이트)', url: `${PRISM_BASE}/prism.min.css` },
        { id: 'ghcolors', label: 'GitHub', url: `${PRISM_EXT}/prism-ghcolors.min.css` },
        { id: 'one-light', label: 'One Light', url: `${PRISM_EXT}/prism-one-light.min.css` },
        { id: 'material-light', label: 'Material Light', url: `${PRISM_EXT}/prism-material-light.min.css` },
        { id: 'coy', label: 'Coy', url: `${PRISM_BASE}/prism-coy.min.css` },
    ];

    function getPrismTheme() {
        const saved = localStorage.getItem(PRISM_THEME_KEY);
        if (saved && PRISM_THEMES.some(t => t.id === saved)) return saved;
        return getTheme() === 'light' ? 'ghcolors' : 'tomorrow';
    }

    function setPrismTheme(themeId, silent) {
        const t = PRISM_THEMES.find(x => x.id === themeId);
        if (!t) return;
        localStorage.setItem(PRISM_THEME_KEY, themeId);
        document.getElementById('prism-theme-inject')?.remove();
        const oldLink = document.getElementById('prism-css');
        const url = t.url + '?v=' + Date.now();
        const newLink = document.createElement('link');
        newLink.rel = 'stylesheet';
        newLink.href = url;
        newLink.onload = () => {
            if (oldLink) oldLink.remove();
            newLink.id = 'prism-css';
            if (!silent) showToast('코드 테마: ' + t.label);
        };
        newLink.onerror = () => {
            newLink.remove();
            if (oldLink) oldLink.href = t.url;
            showToast('코드 테마 로드 실패', 'error');
        };
        (oldLink ? oldLink.parentNode : document.head).appendChild(newLink);
    }

    function initTheme() {
        setTheme(getTheme());
        setBgTheme(getBgTheme());
        const btn = document.getElementById('themeToggle');
        if (btn) btn.onclick = toggleTheme;
        setPrismTheme(getPrismTheme(), true);
    }

    /* ===== 설정 저장소 ===== */
    const PREFS_KEY = 'toolbox_widget_prefs';

    function getPrefs() {
        try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; }
        catch (_) { return {}; }
    }

    function setPref(key, value) {
        const prefs = getPrefs();
        prefs[key] = value;
        localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    }

    function getPref(key, fallback) {
        const v = getPrefs()[key];
        return v !== undefined ? v : fallback;
    }

    /** 사이트 공용 "내 서버" 주소. servermonitor_base → ytdl_api_base 순으로 fallback */
    function getServerBase() {
        const base = getPref('servermonitor_base', '') || getPref('ytdl_api_base', '');
        return typeof base === 'string' ? base.trim() : '';
    }

    /* ===== 사용량 추적 ===== */
    const USAGE_KEY = 'toolbox_usage_stats';

    function getUsageStats() {
        try { return JSON.parse(localStorage.getItem(USAGE_KEY)) || {}; }
        catch (_) { return {}; }
    }

    function saveUsageStats(stats) {
        localStorage.setItem(USAGE_KEY, JSON.stringify(stats));
    }

    const DAILY_WARN_THRESHOLDS = { chat: 50, image: 20, tokens: 500000 };

    function recordUsage(type, tokens) {
        const stats = getUsageStats();
        const today = new Date().toISOString().slice(0, 10);
        if (!stats[today]) stats[today] = { chatCount: 0, chatTokens: 0, imageCount: 0, imageTokens: 0 };
        let totalChat = 0, totalImage = 0;
        Object.values(stats).forEach(s => { totalChat += s.chatCount || 0; totalImage += s.imageCount || 0; });
        if (type === 'chat') {
            stats[today].chatCount++;
            stats[today].chatTokens += (tokens || 0);
            if (totalChat === 0) completeAchievement('first_chat', { title: '첫 대화' });
        } else if (type === 'image') {
            stats[today].imageCount++;
            stats[today].imageTokens += (tokens || 0);
            if (totalImage === 0) completeAchievement('first_image', { title: '첫 이미지 생성' });
        }
        saveUsageStats(stats);

        const d = stats[today];
        const totalDailyTokens = (d.chatTokens || 0) + (d.imageTokens || 0);
        if (type === 'chat' && d.chatCount === DAILY_WARN_THRESHOLDS.chat) {
            showToast(`오늘 채팅 ${DAILY_WARN_THRESHOLDS.chat}회 도달. API 사용량에 유의하세요.`, 'error');
        } else if (type === 'image' && d.imageCount === DAILY_WARN_THRESHOLDS.image) {
            showToast(`오늘 이미지 ${DAILY_WARN_THRESHOLDS.image}회 도달. API 사용량에 유의하세요.`, 'error');
        } else if (totalDailyTokens >= DAILY_WARN_THRESHOLDS.tokens && (totalDailyTokens - (tokens || 0)) < DAILY_WARN_THRESHOLDS.tokens) {
            showToast(`오늘 총 토큰 ${(DAILY_WARN_THRESHOLDS.tokens / 1000).toFixed(0)}K 도달. API 사용량에 유의하세요.`, 'error');
        }
    }

    /* ===== 유저 데이터 (도전과제, 뱃지, 진행도) ===== */
    const USER_DATA_KEY = 'toolbox_user_data';
    const ACHIEVEMENT_REGISTRY = {};
    const BADGE_REGISTRY = {};

    function getUserData() {
        try {
            const raw = localStorage.getItem(USER_DATA_KEY);
            if (raw) {
                const data = JSON.parse(raw);
                if (!data.streaks || typeof data.streaks !== 'object') data.streaks = {};
                return data;
            }
        } catch (_) {}
        return { achievements: [], badges: [], progress: {}, streaks: {} };
    }

    function getStreaks() {
        const data = getUserData();
        return (data.streaks && typeof data.streaks === 'object') ? data.streaks : {};
    }

    function saveUserData(data) {
        try { localStorage.setItem(USER_DATA_KEY, JSON.stringify(data)); } catch (_) {}
    }

    function getProgress(key) {
        const data = getUserData();
        return (data.progress && data.progress[key]) || 0;
    }

    function setProgress(key, value) {
        const data = getUserData();
        if (!data.progress) data.progress = {};
        data.progress[key] = value;
        saveUserData(data);
        return value;
    }

    function incrementProgress(key, amount = 1) {
        return setProgress(key, getProgress(key) + amount);
    }

    function completeAchievement(id, meta = {}) {
        const data = getUserData();
        if (!data.achievements) data.achievements = [];
        if (data.achievements.includes(id)) return false;
        data.achievements.push(id);
        saveUserData(data);
        const title = meta.title || ACHIEVEMENT_REGISTRY[id]?.title || id;
        showToast('도전과제 달성: ' + title, 'success');
        return true;
    }

    function unlockBadge(id, meta = {}) {
        const data = getUserData();
        if (!data.badges) data.badges = [];
        if (data.badges.includes(id)) return false;
        data.badges.push(id);
        saveUserData(data);
        const title = meta.title || BADGE_REGISTRY[id]?.title || id;
        showToast('뱃지 획득: ' + title, 'success');
        return true;
    }

    function registerAchievement(id, def) { ACHIEVEMENT_REGISTRY[id] = def; }
    function registerBadge(id, def) { BADGE_REGISTRY[id] = def; }
    function hasAchievement(id) { return (getUserData().achievements || []).includes(id); }
    function hasBadge(id) { return (getUserData().badges || []).includes(id); }

    function getTools() { return [...tools]; }

    return {
        register, registerDeferred, init, initTheme, switchPage, switchTab, toggleSidebar, getTools,
        isDesktopApp,
        kickLazyLoad, getLazyWidgetPublicMeta,
        showToast, displayResult, copyResult, toggleCollapsible,
        field, resultBox, button, select,
        escapeHtml, formatTimestamp, showLightbox,
        recordUsage, getUsageStats,
        getPref, setPref, getServerBase,
        getTheme, setTheme, toggleTheme,
        getBgTheme, setBgTheme, getBgThemes,
        getPrismTheme, setPrismTheme, getPrismThemes: () => [...PRISM_THEMES],
        getUserData, getStreaks, getProgress, setProgress, incrementProgress,
        completeAchievement, unlockBadge, hasAchievement, hasBadge,
        registerAchievement, registerBadge,
        getAchievementRegistry: () => ({ ...ACHIEVEMENT_REGISTRY }),
        getBadgeRegistry: () => ({ ...BADGE_REGISTRY }),
        getToolMeta, CATEGORIES,
    };
})();

/* ===== Bootstrap ===== */
/* widgets-loader.js가 위젯 로드 후 init 호출 */
