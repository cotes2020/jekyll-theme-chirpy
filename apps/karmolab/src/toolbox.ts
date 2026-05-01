/**
 * KarmoLab — 도구 레지스트리 기반 모듈 시스템
 *
 * ┌─ 아키텍처 ─────────────────────────────────────────────────┐
 * │                                                            │
 * │  index.html ─→ toolbox.js (코어)                           │
 * │                  ├─ 랜딩 페이지 (히어로 + 즐겨찾기 CTA)       │
 * │                  ├─ 상단 메뉴 (카테고리별 드롭다운)          │
 * │                  ├─ 검색, breadcrumb, 테마, 사용량 추적      │
 * │                  └─ 도전과제/뱃지/진행도 시스템              │
 * │               ─→ mdd.js (마스코트)                          │
 * │                  ├─ 이미지 마스코트 (12 감정)               │
 * │                  ├─ 말풍선, 바운스 인터랙션                  │
 * │                  └─ 호감도/스토리 진행 시스템                │
 * │               ─→ gemini.js (AI API)                        │
 * │               ─→ widgets/*.js (개별 도구)                   │
 * │                                                            │
 * │  카테고리:  tool (도구)  /  play (놀이)  /  lab (실험실·개발중)  /  desktop (데스크톱 앱 전용)  /  null (기타)  │
 * └────────────────────────────────────────────────────────────┘
 *
 * 새 도구 추가 방법:
 * 1. widgets/ 폴더에 새 JS 파일 생성
 * 2. widgets-manifest.js(boot) + widgets-lazy-meta.js(지연 메타 단일 출처)
 * 3. Toolbox.register({ id, title, icon, category, desc, hidden?, tabs }) 호출
 *    - icon: SVG path 문자열 (viewBox 0 0 24 24 기준)
 *    - category: 'tool' | 'play' | 'lab' | 'desktop' | null  ('desktop'은 Tauri 앱에서만 메뉴·페이지에 표시)
 *    - desc: 한 줄 설명 (검색·즐겨찾기용)
 *    - hidden: true면 메뉴에 비표시 (user 등)
 *    - tabs: [{ id, label, build(container) }]
 *    - tabLayout: (선택) `'sidebar'` — 탭이 많을 때 왼쪽 세로 목록 + 오른쪽 패널 (문서 위젯 등)
 *
 * 마스코트 연동:
 *   Mdd.setMood('happy')   — 감정 변경
 *   Mdd.say('메시지')       — 말풍선 표시
 *   Mdd.linePreset('success', { msg?, mood?, duration? }) — 티메토 대사 프리셋 (`mdd.js`의 LINE_PRESETS)
 *   Mdd.bounce()           — 바운스 애니메이션
 *   Mdd.addAffection(n)    — 호감도 증가 (스토리 해금 트리거)
 */
// @ts-nocheck — core shell; narrow types incrementally
const Toolbox = (() => {
    const tools = [];

    /* ===== 카테고리 & 메타데이터 ===== */

    const CATEGORIES = [
        { id: 'tool', label: '도구', icon: '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94L6.73 20.15a2.1 2.1 0 0 1-3-3l6.72-6.72a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>' },
        { id: 'play', label: '놀이', icon: '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 12h4m-2-2v4"/><circle cx="15" cy="11" r="1"/><circle cx="18" cy="13" r="1"/>' },
        { id: 'lab', label: '실험실 · 개발중', icon: '<path d="M9 3h6v5l4 4v7a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-7l4-4V3z"/><path d="M9 3h6"/>' },
        { id: 'desktop', label: '데스크톱 앱', icon: '<rect x="2" y="4" width="20" height="14" rx="2" ry="2" fill="none" stroke="currentColor" stroke-width="2"/><line x1="8" y1="20" x2="16" y2="20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>' },
    ];

    /** 위젯별 메타데이터 (category, desc, hidden 등) — 각 위젯 register에서 정의 */
    function getToolMeta(id) {
        const t = tools.find(x => x.id === id);
        return t ? { category: t.category, desc: t.desc, hidden: t.hidden } : null;
    }
    const LAST_PAGE_KEY = 'toolbox_last_page';
    const NAV_LAYOUT_KEY = 'toolbox_nav_layout';
    const SIDEBAR_GROUP_KEY = 'toolbox_sidebar_groups';

    function getNavLayout() {
        const v = localStorage.getItem(NAV_LAYOUT_KEY);
        return (v === 'sidebar' || v === 'header') ? v : 'header';
    }

    function setNavLayout(layout) {
        document.documentElement.setAttribute('data-nav', layout);
        try { localStorage.setItem(NAV_LAYOUT_KEY, layout); } catch (_) {}
    }

    function getSidebarGroupState() {
        try {
            const raw = localStorage.getItem(SIDEBAR_GROUP_KEY);
            if (raw) return JSON.parse(raw);
        } catch (_) {}
        return { tool: true, play: false, lab: false, misc: true };
    }

    function setSidebarGroupState(state) {
        try { localStorage.setItem(SIDEBAR_GROUP_KEY, JSON.stringify(state)); } catch (_) {}
    }

    let megaMenuCloseTimer = null;

    function clearMegaMenuTimer() {
        if (megaMenuCloseTimer) {
            clearTimeout(megaMenuCloseTimer);
            megaMenuCloseTimer = null;
        }
    }

    function scheduleMegaMenuClose() {
        clearMegaMenuTimer();
        megaMenuCloseTimer = setTimeout(() => {
            megaMenuCloseTimer = null;
            closeAllHeaderNav();
        }, 220);
    }

    function closeAllHeaderNav() {
        clearMegaMenuTimer();
        document.querySelectorAll('.header-nav-group.is-open').forEach((wrap) => {
            wrap.classList.remove('is-open');
            const tr = wrap.querySelector('.header-nav-trigger');
            if (tr) tr.setAttribute('aria-expanded', 'false');
            const p = wrap.querySelector('.header-nav-panel');
            if (p) p.hidden = true;
        });
    }

    function closeAllHeaderNavExcept(except) {
        document.querySelectorAll('.header-nav-group.is-open').forEach((w) => {
            if (w === except) return;
            w.classList.remove('is-open');
            const tr = w.querySelector('.header-nav-trigger');
            if (tr) tr.setAttribute('aria-expanded', 'false');
            const p = w.querySelector('.header-nav-panel');
            if (p) p.hidden = true;
        });
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

    /** 레지스트리·초기화용 — 스크립트는 첫 방문 시 loadDeferredWidget에서 로드 */
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

    function getWorldScriptBase() {
        const b = typeof window !== 'undefined' && window.KARMOLAB_WORLD_SCRIPT_BASE;
        if (b) return b;
        try {
            const origin = location.origin || '';
            return origin + '/apps/karmolab/world/';
        } catch (_) {
            return '/apps/karmolab/world/';
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
        const worldBase = getWorldScriptBase();
        const p = (async () => {
            let waitIdx = (window.KARMOLAB_WIDGET_LOADER_WAIT || []).length;
            for (let i = 0; i < paths.length; i++) {
                const rawPath = paths[i];
                const isWorld = typeof rawPath === 'string' && rawPath.startsWith('world/');
                const url = (isWorld ? worldBase : base) + (isWorld ? rawPath.slice('world/'.length) : rawPath) + '.js';
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

    const UPDATE_DISMISS_KEY = 'karmolab-update-dismissed-version';

    function setupUpdateBannerListener() {
        if (typeof window === 'undefined' || !window.__KARMOLAB_DESKTOP__) return;
        const listenFn = window.__TAURI__?.event?.listen;
        if (typeof listenFn !== 'function') return;
        listenFn('karmolab://update-available', (e) => {
            const payload = (e?.payload || {}) as { current?: string; new?: string };
            if (!payload.new) return;
            // 사용자가 이미 닫은 버전이면 다시 띄우지 않는다 (수동으로 트레이 메뉴 사용 가능).
            try {
                if (localStorage.getItem(UPDATE_DISMISS_KEY) === payload.new) return;
            } catch (_) { /* localStorage 차단 환경 무시 */ }
            showUpdateBanner(payload.current || '?', payload.new);
        }).catch(() => {});
    }

    function showUpdateBanner(current, newVer) {
        if (document.querySelector('.karmolab-update-banner')) return;
        const banner = document.createElement('div');
        banner.className = 'karmolab-update-banner';

        const body = document.createElement('div');
        body.className = 'karmolab-update-banner-body';

        const msg = document.createElement('div');
        msg.className = 'karmolab-update-banner-msg';
        msg.innerHTML = `새 버전: <code>${escapeHtml(current)}</code> → <code>${escapeHtml(newVer)}</code>`;

        const notesA = document.createElement('a');
        notesA.className = 'karmolab-update-banner-notes';
        notesA.href = `https://github.com/mascari4615/mascari4615.github.io/releases/tag/karmolab-v${encodeURIComponent(newVer)}`;
        notesA.target = '_blank';
        notesA.rel = 'noopener noreferrer';
        notesA.textContent = '변경사항 보기';

        const progress = document.createElement('progress');
        progress.className = 'karmolab-update-banner-progress';
        progress.value = 0;
        progress.max = 1;
        progress.hidden = true;

        body.appendChild(msg);
        body.appendChild(notesA);
        body.appendChild(progress);

        const installBtn = document.createElement('button');
        installBtn.type = 'button';
        installBtn.className = 'karmolab-update-banner-install';
        installBtn.textContent = '지금 설치';

        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'karmolab-update-banner-close';
        closeBtn.setAttribute('aria-label', '닫기');
        closeBtn.textContent = '×';

        banner.appendChild(body);
        banner.appendChild(installBtn);
        banner.appendChild(closeBtn);
        document.body.appendChild(banner);

        const formatBytes = (n: number): string => {
            if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' MB';
            if (n >= 1024) return (n / 1024).toFixed(0) + ' KB';
            return n + ' B';
        };

        const listenFn = window.__TAURI__?.event?.listen;
        let unlistenProgress: (() => void) | null = null;
        let unlistenFinish: (() => void) | null = null;

        const stopListeners = () => {
            try { unlistenProgress?.(); } catch (_) { /* ignore */ }
            try { unlistenFinish?.(); } catch (_) { /* ignore */ }
            unlistenProgress = null;
            unlistenFinish = null;
        };

        closeBtn.addEventListener('click', () => {
            try { localStorage.setItem(UPDATE_DISMISS_KEY, newVer); } catch (_) { /* ignore */ }
            stopListeners();
            banner.remove();
        });

        installBtn.addEventListener('click', () => {
            const invoke = window.__TAURI__?.core?.invoke;
            if (typeof invoke !== 'function') {
                msg.textContent = '설치 불가: Tauri invoke를 찾지 못했습니다.';
                return;
            }

            installBtn.disabled = true;
            installBtn.textContent = '준비 중…';
            progress.hidden = false;

            if (typeof listenFn === 'function') {
                listenFn('karmolab://update-progress', (e) => {
                    const p = (e?.payload || {}) as { downloaded?: number; total?: number };
                    if (typeof p.total === 'number' && p.total > 0 && typeof p.downloaded === 'number') {
                        progress.value = Math.min(p.downloaded, p.total);
                        progress.max = p.total;
                        installBtn.textContent = `${formatBytes(p.downloaded)} / ${formatBytes(p.total)}`;
                    } else if (typeof p.downloaded === 'number') {
                        progress.removeAttribute('value'); // indeterminate
                        installBtn.textContent = `${formatBytes(p.downloaded)} 받는 중`;
                    }
                }).then((un) => { unlistenProgress = un; }).catch(() => {});

                listenFn('karmolab://update-download-finished', () => {
                    progress.removeAttribute('value');
                    installBtn.textContent = '설치 중…';
                }).then((un) => { unlistenFinish = un; }).catch(() => {});
            }

            invoke('desktop_install_pending_update', {})
                .then((res) => {
                    stopListeners();
                    progress.hidden = true;
                    msg.textContent = typeof res === 'string' ? res : '설치 완료.';
                    installBtn.disabled = false;
                    installBtn.textContent = '재시작';
                    installBtn.classList.add('karmolab-update-banner-restart');
                    installBtn.onclick = () => {
                        installBtn.disabled = true;
                        installBtn.textContent = '재시작 중…';
                        void invoke('desktop_restart_app', {}).catch(() => {
                            installBtn.disabled = false;
                            installBtn.textContent = '재시작';
                        });
                    };
                })
                .catch((err) => {
                    stopListeners();
                    progress.hidden = true;
                    const errMsg = err instanceof Error ? err.message : String(err);
                    msg.textContent = `실패: ${errMsg}`;
                    installBtn.disabled = false;
                    installBtn.textContent = '다시 시도';
                });
        });
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
            const ver = window.__KARMOLAB_VERSION__;
            span.textContent = ver ? `앱 v${ver}` : '앱';
            span.title = ver
              ? `KarmoLab 데스크톱 앱 v${ver}`
              : 'Tauri 데스크톱 앱에서 실행 중입니다. 웹에서는 이 배지가 보이지 않습니다.';
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

    /** decorations:false 윈도우의 헤더 컨트롤(min/max/close)을 활성화. 데스크톱 외에는 noop. */
    function installWindowControls() {
        if (!isDesktopApp()) return;
        const controls = document.getElementById('windowControls');
        if (!controls) return;

        const tauriWin = window.__TAURI__?.window;
        const getCurrentWindow = tauriWin?.getCurrentWindow;
        if (typeof getCurrentWindow !== 'function') {
            console.warn('[Toolbox] Tauri window API 미주입 — 윈도우 컨트롤 비활성');
            return;
        }
        const win = getCurrentWindow();

        controls.style.display = 'flex';
        controls.removeAttribute('aria-hidden');

        document.getElementById('wcMinimize')?.addEventListener('click', () => {
            win.minimize().catch((e) => console.warn('minimize 실패', e));
        });
        document.getElementById('wcMaximize')?.addEventListener('click', () => {
            win.toggleMaximize().catch((e) => console.warn('toggleMaximize 실패', e));
        });
        document.getElementById('wcClose')?.addEventListener('click', () => {
            win.close().catch((e) => console.warn('close 실패', e));
        });

        async function syncMaximized() {
            try {
                const m = await win.isMaximized();
                controls!.setAttribute('data-maximized', m ? 'true' : 'false');
            } catch { /* ignore */ }
        }
        void syncMaximized();
        win.onResized?.(() => { void syncMaximized(); }).catch(() => {});
    }

    /** 데스크톱 전용(category desktop) 도구는 일반 브라우저에서 메뉴·페이지에 넣지 않음 */
    function isDesktopOnlyTool(tool) {
        return tool && tool.category === 'desktop';
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
        const headerNav = document.getElementById('header-nav');
        const mobileNav = document.getElementById('mobile-nav');
        const toolPages = document.getElementById('tool-pages');
        const hiddenSet = new Set(tools.filter(t => t.hidden).map(t => t.id));

        function addNavItem(container, tool) {
            const a = document.createElement('a');
            a.className = 'nav-item';
            a.href = '#';
            a.dataset.page = tool.id;
            a.title = tool.title;
            a.innerHTML = `<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${tool.icon}</svg><span class="nav-item-text">${tool.title}</span>`;
            a.onclick = (e) => {
                e.preventDefault();
                closeAllHeaderNav();
                switchPage(tool.id);
            };
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

        function buildHeaderNavGroup(label, catTools, navParent) {
            if (!catTools.length) return;

            const canHover = typeof window !== 'undefined' && window.matchMedia('(hover: hover)').matches;

            const wrap = document.createElement('div');
            wrap.className = 'header-nav-group';
            const trigger = document.createElement('button');
            trigger.type = 'button';
            trigger.className = 'header-nav-trigger';
            trigger.setAttribute('aria-expanded', 'false');
            trigger.setAttribute('aria-haspopup', 'true');
            const labelSpan = document.createElement('span');
            labelSpan.className = 'header-nav-trigger-label';
            labelSpan.textContent = label;

            const panel = document.createElement('div');
            panel.className = 'header-nav-panel header-nav-panel--mega';
            panel.hidden = true;
            const inner = document.createElement('div');
            inner.className = 'header-nav-panel-inner';
            catTools.forEach(tool => addNavItem(inner, tool));
            panel.appendChild(inner);

            trigger.appendChild(labelSpan);

            function openThis() {
                clearMegaMenuTimer();
                closeAllHeaderNavExcept(wrap);
                wrap.classList.add('is-open');
                panel.hidden = false;
                trigger.setAttribute('aria-expanded', 'true');
            }

            function toggleClick(e) {
                e.stopPropagation();
                const wasOpen = wrap.classList.contains('is-open');
                if (wasOpen) {
                    wrap.classList.remove('is-open');
                    panel.hidden = true;
                    trigger.setAttribute('aria-expanded', 'false');
                } else {
                    openThis();
                }
            }

            if (canHover) {
                wrap.addEventListener('mouseenter', openThis);
                wrap.addEventListener('mouseleave', scheduleMegaMenuClose);
            } else {
                trigger.addEventListener('click', toggleClick);
            }

            wrap.appendChild(trigger);
            wrap.appendChild(panel);
            navParent.appendChild(wrap);
        }

        if (headerNav) {
            const headerNavScroll = document.createElement('div');
            headerNavScroll.className = 'header-nav-scroll';
            headerNav.appendChild(headerNavScroll);

            CATEGORIES.forEach(cat => {
                const catTools = tools
                    .filter(t => !hiddenSet.has(t.id) && t.category === cat.id && (cat.id !== 'desktop' || isDesktopApp()))
                    .sort((a, b) => (a.title || '').localeCompare(b.title || '', 'ko-KR'));
                buildHeaderNavGroup(cat.label, catTools, headerNavScroll);
            });

            const uncategorized = tools
                .filter(t => !hiddenSet.has(t.id) && !t.category)
                .sort((a, b) => (a.title || '').localeCompare(b.title || '', 'ko-KR'));
            if (uncategorized.length) {
                buildHeaderNavGroup('기타', uncategorized, headerNavScroll);
            }

            document.addEventListener('click', (e) => {
                if (!e.target.closest('.header-nav')) closeAllHeaderNav();
            });
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') closeAllHeaderNav();
            });
        }

        // Build sidebar nav groups
        const sidebarNavEl = document.getElementById('sidebar-nav');
        if (sidebarNavEl) {
            function buildSidebarGroup(catId, label, catTools) {
                if (!catTools.length) return;
                const isOpen = getSidebarGroupState()[catId] !== undefined
                    ? getSidebarGroupState()[catId]
                    : (catId === 'tool');
                const wrap = document.createElement('div');
                wrap.className = 'sidebar-group';
                const trigger = document.createElement('button');
                trigger.type = 'button';
                trigger.className = 'sidebar-group-trigger' + (isOpen ? ' open' : '');
                trigger.setAttribute('aria-expanded', String(isOpen));
                trigger.innerHTML = '<span class="chevron" aria-hidden="true"></span>'
                    + '<span class="sidebar-group-label">' + label + '</span>';
                const body = document.createElement('div');
                body.className = 'sidebar-group-body' + (isOpen ? ' open' : '');
                catTools.forEach(tool => addNavItem(body, tool));
                trigger.onclick = () => {
                    const open = body.classList.toggle('open');
                    trigger.classList.toggle('open', open);
                    trigger.setAttribute('aria-expanded', String(open));
                    setSidebarGroupState({ ...getSidebarGroupState(), [catId]: open });
                };
                wrap.appendChild(trigger);
                wrap.appendChild(body);
                sidebarNavEl.appendChild(wrap);
            }

            CATEGORIES.forEach(cat => {
                const catTools = tools
                    .filter(t => !hiddenSet.has(t.id) && t.category === cat.id && (cat.id !== 'desktop' || isDesktopApp()))
                    .sort((a, b) => (a.title || '').localeCompare(b.title || '', 'ko-KR'));
                buildSidebarGroup(cat.id, cat.label, catTools);
            });

            const sidebarUncategorized = tools
                .filter(t => !hiddenSet.has(t.id) && !t.category)
                .sort((a, b) => (a.title || '').localeCompare(b.title || '', 'ko-KR'));
            buildSidebarGroup('misc', '기타', sidebarUncategorized);
        }

        // Build landing page
        toolPages.appendChild(buildLanding());

        // Build tool pages (가나다순)
        const sortedTools = [...tools].sort((a, b) => (a.title || '').localeCompare(b.title || '', 'ko-KR'));
        sortedTools.forEach(tool => {
            if (!hiddenSet.has(tool.id) && (!isDesktopOnlyTool(tool) || isDesktopApp())) addMobileNavItem(tool);
            if (!isDesktopOnlyTool(tool) || isDesktopApp()) toolPages.appendChild(buildToolPage(tool));
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
        const isValidPage = (id) => {
            if (id === 'home' || id === 'user') return true;
            const t = tools.find(x => x.id === id);
            if (!t) return false;
            if (isDesktopOnlyTool(t) && !isDesktopApp()) return false;
            return true;
        };
        const initialPage = (hashPage && isValidPage(hashPage))
            ? hashPage
            : (lastPage && isValidPage(lastPage) ? lastPage : 'home');

        switchPage(initialPage, { pushHistory: false });
        history.replaceState({ pageId: initialPage }, '', location.pathname + (location.search || '') + '#' + initialPage);

        window.addEventListener('popstate', () => {
            const pageId = pageIdFromHash();
            if (isValidPage(pageId)) switchPage(pageId, { pushHistory: false });
        });

        injectDesktopBadge();
        setupUpdateBannerListener();
        installWindowControls();
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
            <p class="landing-cta-hint">상단 메뉴에서 카테고리를 열고 도구를 선택하세요</p>
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
        closeAllHeaderNav();
        let { pushHistory = true } = opts;
        const base = location.pathname + (location.search || '');
        const denied = tools.find(t => t.id === pageId);
        if (denied && isDesktopOnlyTool(denied) && !isDesktopApp()) {
            history.replaceState({ pageId: 'home' }, '', base + '#home');
            pageId = 'home';
            pushHistory = false;
        }
        const urlWithHash = base + '#' + pageId;
        if (pushHistory) {
            history.pushState({ pageId }, '', urlWithHash);
        }
        const landing = document.getElementById('page-home');
        const allPages = document.querySelectorAll('.tool-page');
        const allNav = document.querySelectorAll('.nav-item');
        const headerHomeBtn = document.getElementById('headerHomeBtn');
        const breadcrumb = document.getElementById('breadcrumb');

        const toolForPage = tools.find(t => t.id === pageId);
        if (toolForPage && toolForPage._deferred) {
            kickLazyLoad(pageId);
        }

        allPages.forEach(p => p.classList.remove('active'));
        allNav.forEach(n => n.classList.remove('active'));
        if (headerHomeBtn) headerHomeBtn.classList.remove('active');
        if (landing) landing.classList.remove('active');

        if (pageId === 'home') {
            if (landing) landing.classList.add('active');
            if (headerHomeBtn) headerHomeBtn.classList.add('active');
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
        tabRow.querySelectorAll('.tab-btn').forEach((b) => {
            b.classList.remove('active');
            b.setAttribute('aria-selected', 'false');
        });
        page.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        btn.setAttribute('aria-selected', 'true');
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

        let panelsHost: HTMLElement = div;

        if (tool.tabs.length > 1) {
            const tabRow = document.createElement('div');
            tabRow.className = 'tab-row';
            if (tool.tabLayout === 'sidebar') {
                tabRow.classList.add('tab-row--sidebar');
                tabRow.setAttribute('role', 'tablist');
                tabRow.setAttribute('aria-orientation', 'vertical');
            }
            tool.tabs.forEach((tab, i) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
                btn.dataset.tabId = tab.id;
                btn.textContent = tab.label;
                btn.setAttribute('role', 'tab');
                btn.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
                btn.onclick = function () { switchTab(this, tab.id); };
                tabRow.appendChild(btn);
            });

            if (tool.tabLayout === 'sidebar') {
                const wrap = document.createElement('div');
                wrap.className = 'tool-tab-sidebar-layout';
                wrap.appendChild(tabRow);
                const col = document.createElement('div');
                col.className = 'tab-panels-column';
                wrap.appendChild(col);
                div.appendChild(wrap);
                panelsHost = col;
            } else {
                div.appendChild(tabRow);
            }
        }

        tool.tabs.forEach((tab, i) => {
            const panel = document.createElement('div');
            panel.className = 'tab-panel' + (i === 0 ? ' active' : '');
            panel.id = 'panel-' + tab.id;
            panel.setAttribute('role', 'tabpanel');
            tab.build(panel);
            panelsHost.appendChild(panel);
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
        { id: 'observatory', label: '관측실' },
        { id: 'blue-magenta', label: '블루 매젠타' },
        { id: 'mesh-dots', label: '메쉬 도트' },
        { id: 'aurora', label: '오로라' },
        { id: 'subtle', label: '은은한' },
        { id: 'minimal', label: '미니멀' },
    ];

    function getBgTheme() {
        const saved = localStorage.getItem(BG_THEME_KEY);
        if (saved && BG_THEMES.some(t => t.id === saved)) return saved;
        return 'observatory';
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
        setNavLayout(getNavLayout());
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
        register, registerDeferred, init, initTheme, switchPage, switchTab, getTools,
        isDesktopApp,
        kickLazyLoad, getLazyWidgetPublicMeta,
        showToast, displayResult, copyResult, toggleCollapsible,
        field, resultBox, button, select,
        escapeHtml, formatTimestamp, showLightbox,
        recordUsage, getUsageStats,
        getPref, setPref,
        getNavLayout, setNavLayout,
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
