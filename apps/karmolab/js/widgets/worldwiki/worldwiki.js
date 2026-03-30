/**
 * World Wiki — KarmoWorld 위키 뷰어
 * - 좌측: 항목 목록(캐릭터 위주 시작)
 * - 우측: Markdown 문서 로드/렌더
 */
(function () {
    Mdd.injectCSS('worldwiki', `
        .ww-wrap { display:grid; grid-template-columns: 280px 1fr; gap:16px; height:100%; }
        .ww-left { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg); padding:12px; overflow:auto; }
        .ww-right { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg); padding:16px; overflow:auto; }
        .ww-title { font-size:var(--font-size-md); font-weight:800; margin:0 0 10px; }
        .ww-search { width:100%; padding:8px 10px; border-radius:10px; border:1px solid var(--border); background:var(--bg-tertiary); color:var(--text-primary); font-size:var(--font-size-xs); }
        .ww-list { margin-top:10px; display:flex; flex-direction:column; gap:6px; }
        .ww-item { width:100%; text-align:left; padding:10px 10px; border-radius:12px; border:1px solid var(--border); background:var(--bg-tertiary); color:var(--text-primary); cursor:pointer; font-size:var(--font-size-xs); }
        .ww-item:hover { border-color:var(--accent); }
        .ww-item.active { border-color:var(--accent); box-shadow:0 0 0 2px var(--accent-subtle); }
        .ww-md h1 { font-size:22px; margin:0 0 14px; padding-bottom:12px; border-bottom:1px solid var(--border); }
        .ww-md h2 { font-size:16px; margin:22px 0 10px; color:var(--accent); }
        .ww-md p, .ww-md li { color:var(--text-secondary); line-height:1.8; font-size:14px; }
        .ww-md code { font-family:var(--font-mono); font-size:var(--font-size-xs); background:var(--bg-tertiary); padding:2px 6px; border-radius:4px; color:var(--accent); }
        .ww-md pre { border:1px solid var(--border); border-radius:var(--radius-md); overflow:auto; }
        .ww-md pre code { display:block; padding:14px; }
        .ww-hint { color:var(--text-tertiary); font-size:var(--font-size-xs); margin:8px 0 0; }
    `);

    const WORLD_BASE = (function () {
        const script = document.currentScript;
        if (script && script.src) {
            const url = new URL(script.src);
            return url.origin + url.pathname.replace(/\/js\/widgets\/.*/, '/world/');
        }
        return './world/';
    })();

    function wikiUrlForCharacter(slug) {
        return WORLD_BASE + 'wiki/entities/characters/' + slug + '.md';
    }

    async function loadWikiText(url) {
        const r = await fetch(url);
        if (!r.ok) throw new Error('문서 로드 실패');
        return await r.text();
    }

    function renderWikiMarkdown(container, md) {
        const wrap = document.createElement('div');
        wrap.className = 'ww-md';
        if (window.ChatbotMarkdown?.renderMarkdown) {
            wrap.innerHTML = window.ChatbotMarkdown.renderMarkdown(md);
        } else if (typeof marked !== 'undefined') {
            marked.setOptions({ breaks: true, gfm: true });
            wrap.innerHTML = marked.parse(md);
        } else {
            wrap.innerHTML = '<p style="color:var(--error)">마크다운 렌더러가 없습니다.</p>';
        }
        container.innerHTML = '';
        container.appendChild(wrap);
        if (typeof Prism !== 'undefined') {
            wrap.querySelectorAll('pre code').forEach(el => Prism.highlightElement(el));
        }
    }

    function getCharacterIndex() {
        const chars = window.KarmoWorld?.entities?.characters;
        if (!chars) return [];
        return Object.values(chars)
            .filter(x => x && x.slug && x.nameKo)
            .map(x => ({ slug: x.slug, title: `${x.nameKo}${x.nameEn ? ` (${x.nameEn})` : ''}`, oneLine: x.oneLine || '' }))
            .sort((a, b) => a.title.localeCompare(b.title, 'ko'));
    }

    function build(container) {
        container.innerHTML = `
            <div class="ww-wrap">
                <aside class="ww-left">
                    <p class="ww-title">세계관 위키</p>
                    <input class="ww-search" id="wwSearch" placeholder="검색 (이름/한 줄)..." />
                    <div class="ww-list" id="wwList"></div>
                    <p class="ww-hint">초기 버전: 캐릭터 문서만 표시합니다.</p>
                </aside>
                <section class="ww-right">
                    <div id="wwDoc" class="ww-md" style="color:var(--text-secondary)">왼쪽에서 항목을 선택하세요.</div>
                </section>
            </div>
        `;

        const listEl = container.querySelector('#wwList');
        const searchEl = container.querySelector('#wwSearch');
        const docEl = container.querySelector('#wwDoc');
        if (!listEl || !searchEl || !docEl) return;

        const all = getCharacterIndex();
        let activeSlug = '';

        async function open(slug) {
            activeSlug = slug;
            listEl.querySelectorAll('.ww-item').forEach(b => b.classList.toggle('active', b.dataset.slug === slug));
            docEl.textContent = '문서 불러오는 중...';
            try {
                const md = await loadWikiText(wikiUrlForCharacter(slug));
                renderWikiMarkdown(docEl, md);
            } catch (e) {
                docEl.innerHTML = '<p style="color:var(--error)">문서를 불러오지 못했습니다.</p>';
            }
        }

        function renderList(q) {
            const query = (q || '').trim().toLowerCase();
            const items = query
                ? all.filter(x => (x.title + ' ' + x.oneLine).toLowerCase().includes(query))
                : all;
            listEl.innerHTML = items.map(x => (
                `<button type="button" class="ww-item" data-slug="${Toolbox.escapeHtml(x.slug)}">` +
                `<div style="font-weight:700">${Toolbox.escapeHtml(x.title)}</div>` +
                (x.oneLine ? `<div style="margin-top:2px;color:var(--text-tertiary);font-size:12px;">${Toolbox.escapeHtml(x.oneLine)}</div>` : '') +
                `</button>`
            )).join('');
            listEl.querySelectorAll('.ww-item').forEach(btn => {
                btn.addEventListener('click', () => open(btn.dataset.slug));
                if (btn.dataset.slug === activeSlug) btn.classList.add('active');
            });
        }

        renderList('');
        searchEl.addEventListener('input', () => renderList(searchEl.value));
        if (all[0]) void open(all[0].slug);
    }

    Toolbox.register({
        ...Toolbox.getLazyWidgetPublicMeta('worldwiki'),
        tabs: [{ id: 'worldwiki-main', label: '위키', build }]
    });
})();

