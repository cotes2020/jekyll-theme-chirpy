/**
 * World Wiki — KarmoWorld 위키 뷰어
 * - 좌측: 항목 목록(캐릭터 위주 시작)
 * - 우측: Markdown 문서 로드/렌더
 */
(function (): void {
    const esc = (s: string): string => (Toolbox.escapeHtml ? Toolbox.escapeHtml(s) : s);

    Mdd.injectCSS('worldwiki', `
        .ww-wrap { display:grid; grid-template-columns: 280px minmax(0, 1fr) 260px; gap:16px; height:100%; }
        .ww-left, .ww-mid, .ww-toc { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg); overflow:auto; }
        .ww-left { padding:12px; }
        .ww-mid { padding:18px 20px; }
        .ww-toc { padding:12px; }

        .ww-title { font-size:var(--font-size-md); font-weight:800; margin:0 0 10px; }
        .ww-search { width:100%; padding:8px 10px; border-radius:10px; border:1px solid var(--border); background:var(--bg-tertiary); color:var(--text-primary); font-size:var(--font-size-xs); }
        .ww-list { margin-top:10px; display:flex; flex-direction:column; gap:6px; }
        .ww-item { width:100%; text-align:left; padding:10px 10px; border-radius:12px; border:1px solid var(--border); background:var(--bg-tertiary); color:var(--text-primary); cursor:pointer; font-size:var(--font-size-xs); }
        .ww-item:hover { border-color:var(--accent); }
        .ww-item.active { border-color:var(--accent); box-shadow:0 0 0 2px var(--accent-subtle); }

        .ww-doc { max-width:860px; margin:0 auto; }
        .ww-doc .ww-md { font-size:14px; line-height:1.85; }
        .ww-md h1 { font-size:26px; font-weight:900; letter-spacing:-0.03em; margin:0 0 14px; padding-bottom:12px; border-bottom:1px solid var(--border); }
        .ww-md h2 { font-size:18px; font-weight:800; margin:30px 0 10px; color:var(--accent); letter-spacing:-0.02em; }
        .ww-md h3 { font-size:15px; font-weight:800; margin:22px 0 8px; }
        .ww-md p { margin:0 0 12px; color:var(--text-secondary); }
        .ww-md ul, .ww-md ol { margin:0 0 14px; padding-left:22px; color:var(--text-secondary); }
        .ww-md li { margin:4px 0; }
        .ww-md a { color:var(--accent); text-decoration:none; }
        .ww-md a:hover { text-decoration:underline; }
        .ww-md hr { border:none; border-top:1px solid var(--border); margin:22px 0; }
        .ww-md blockquote { margin:14px 0; padding:10px 14px; border-left:3px solid var(--accent); background:var(--accent-subtle); border-radius:0 var(--radius-sm) var(--radius-sm) 0; color:var(--text-secondary); }

        .ww-heading { position:relative; scroll-margin-top:16px; }
        .ww-heading:hover .ww-anchor { opacity:1; }
        .ww-anchor { opacity:0; position:absolute; left:-22px; top:50%; transform:translateY(-50%); font-size:14px; color:var(--text-tertiary); cursor:pointer; user-select:none; }
        .ww-anchor:focus { opacity:1; outline:2px solid var(--accent-subtle); outline-offset:2px; border-radius:6px; }

        .ww-callout { border:1px solid var(--border); border-radius:14px; padding:12px 14px; margin:14px 0; background:var(--bg-tertiary); }
        .ww-callout-title { font-weight:900; margin:0 0 6px; font-size:13px; letter-spacing:-0.01em; }
        .ww-callout-body { color:var(--text-secondary); }
        .ww-callout-tip .ww-callout-title { color:var(--accent); }
        .ww-callout-note .ww-callout-title { color:var(--text-primary); }
        .ww-callout-warning .ww-callout-title { color:var(--warn, var(--error)); }

        .ww-toc-title { font-size:var(--font-size-xs); font-weight:900; color:var(--text-secondary); margin:2px 0 10px; }
        .ww-toc-list { display:flex; flex-direction:column; gap:6px; }
        .ww-toc-a { font-size:12px; color:var(--text-tertiary); text-decoration:none; line-height:1.45; padding:6px 8px; border-radius:10px; border:1px solid transparent; }
        .ww-toc-a:hover { color:var(--text-secondary); border-color:var(--border); background:var(--bg-tertiary); }
        .ww-toc-a.active { color:var(--text-primary); border-color:var(--accent); box-shadow:0 0 0 2px var(--accent-subtle); }
        .ww-toc-l2 { padding-left:18px; }
        .ww-toc-l3 { padding-left:28px; }

        .ww-hint { color:var(--text-tertiary); font-size:var(--font-size-xs); margin:8px 0 0; }
    `);

    const WORLD_BASE = (function (): string {
        const script = document.currentScript;
        if (script && 'src' in script && script.src) {
            const url = new URL(script.src);
            return url.origin + url.pathname.replace(/\/js\/widgets\/.*/, '/world/');
        }
        return './world/';
    })();

    function wikiUrlForCharacter(slug: string): string {
        return WORLD_BASE + 'wiki/entities/characters/' + slug + '.md';
    }

    async function loadWikiText(url: string): Promise<string> {
        const r = await fetch(url);
        if (!r.ok) throw new Error('문서 로드 실패');
        return await r.text();
    }

    function stripYamlFrontMatter(md: string): string {
        if (typeof md !== 'string') return '';
        // YAML frontmatter: starts with --- on first line and ends with --- (or ...).
        if (!md.startsWith('---\n') && !md.startsWith('---\r\n')) return md;
        const lines = md.split(/\r?\n/);
        if (lines[0].trim() !== '---') return md;
        for (let i = 1; i < lines.length; i++) {
            const t = lines[i].trim();
            if (t === '---' || t === '...') {
                return lines.slice(i + 1).join('\n').replace(/^\n+/, '');
            }
        }
        return md;
    }

    function slugify(s: string): string {
        return String(s || '')
            .trim()
            .toLowerCase()
            .replace(/[\s]+/g, '-')
            .replace(/[^\w\-가-힣]+/g, '')
            .replace(/\-+/g, '-')
            .replace(/^\-+|\-+$/g, '');
    }

    function ensureUniqueId(base: string, used: Set<string>): string {
        let id = base || 'section';
        if (!used.has(id)) { used.add(id); return id; }
        let i = 2;
        while (used.has(`${id}-${i}`)) i++;
        const out = `${id}-${i}`;
        used.add(out);
        return out;
    }

    function applyCallouts(root: HTMLElement): void {
        root.querySelectorAll('blockquote').forEach((bq) => {
            const p = bq.querySelector(':scope > p');
            if (!p) return;
            const raw = (p.textContent || '').trim();
            const m = raw.match(/^\[!(tip|note|warning)\]\s*(.*)$/i);
            if (!m || !m[1]) return;
            const kind = m[1].toLowerCase();
            const title = m[2] || (kind === 'tip' ? '팁' : kind === 'warning' ? '주의' : '노트');
            // remove marker from first paragraph
            p.textContent = (p.textContent || '').replace(/^\[!(tip|note|warning)\]\s*/i, '');

            const box = document.createElement('div');
            box.className = `ww-callout ww-callout-${kind}`;
            const h = document.createElement('div');
            h.className = 'ww-callout-title';
            h.textContent = title;
            const body = document.createElement('div');
            body.className = 'ww-callout-body';
            while (bq.firstChild) body.appendChild(bq.firstChild);
            box.appendChild(h);
            box.appendChild(body);
            bq.replaceWith(box);
        });
    }

    function applyAnchors(root: HTMLElement, tocEl: HTMLElement | null): Array<{ id: string; text: string; level: number }> {
        const used = new Set<string>();
        const headings = Array.from(root.querySelectorAll('h1, h2, h3'));
        const toc: Array<{ id: string; text: string; level: number }> = [];

        headings.forEach((h) => {
            const el = h as HTMLElement;
            const level = h.tagName === 'H1' ? 1 : h.tagName === 'H2' ? 2 : 3;
            const text = (h.textContent || '').trim();
            if (!text) return;
            const id = ensureUniqueId(slugify(text), used);
            el.id = el.id || id;
            el.classList.add('ww-heading');

            const a = document.createElement('span');
            a.className = 'ww-anchor';
            a.tabIndex = 0;
            a.setAttribute('role', 'button');
            a.setAttribute('aria-label', '링크 복사');
            a.textContent = '#';
            const copy = async () => {
                const url = location.origin + location.pathname + '#' + el.id;
                try {
                    await navigator.clipboard.writeText(url);
                    Toolbox.showToast?.('링크 복사됨', undefined, undefined);
                } catch (_) {
                    location.hash = el.id;
                    Toolbox.showToast?.('링크를 복사하지 못했습니다.', 'error', undefined);
                }
            };
            a.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); location.hash = el.id; void copy(); });
            a.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); location.hash = el.id; void copy(); } });
            el.prepend(a);

            toc.push({ id: el.id, text, level });
        });

        if (tocEl) {
            tocEl.innerHTML = `
                <div class="ww-toc-title">이 문서 목차</div>
                <nav class="ww-toc-list">
                    ${toc.map(x => `<a class="ww-toc-a ww-toc-l${x.level}" href="#${esc(x.id)}">${esc(x.text)}</a>`).join('')}
                </nav>
            `;
        }

        return toc;
    }

    function wireTocActive(tocWrap: HTMLElement, docWrap: HTMLElement): void {
        const links = Array.from(tocWrap.querySelectorAll('.ww-toc-a'));
        if (!links.length) return;

        const headings = Array.from(docWrap.querySelectorAll('h1, h2, h3')).filter((h) => h.id);
        const obs = new IntersectionObserver((entries) => {
            const visible = entries.filter(e => e.isIntersecting).sort((a, b) => (a.boundingClientRect.top - b.boundingClientRect.top));
            if (!visible.length) return;
            const id = visible[0].target.id;
            links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${id}`));
        }, { root: docWrap, threshold: [0.2, 0.5, 0.8] });
        headings.forEach(h => obs.observe(h));

        links.forEach(a => a.addEventListener('click', () => {
            const id = a.getAttribute('href')?.slice(1);
            const target = id ? docWrap.querySelector(`#${CSS.escape(id)}`) : null;
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }));
    }

    function renderWikiMarkdown(container: HTMLElement, md: string, tocContainer: HTMLElement | null): void {
        const wrap = document.createElement('div');
        wrap.className = 'ww-md';
        md = stripYamlFrontMatter(md);
        if (typeof marked !== 'undefined') {
            marked.setOptions({ breaks: true, gfm: true });
            wrap.innerHTML = marked.parse(md);
        } else {
            wrap.innerHTML = '<p style="color:var(--error)">marked.js가 없어 위키를 렌더링할 수 없습니다.</p>';
        }
        container.innerHTML = '';
        container.appendChild(wrap);
        applyCallouts(wrap);
        applyAnchors(wrap, tocContainer);
    }

    type CharRow = { slug: string; title: string; oneLine: string };

    function getCharacterIndex(): CharRow[] {
        const chars = window.KarmoWorld?.entities?.characters;
        if (!chars) return [];
        return Object.values(chars)
            .filter((x): x is Record<string, unknown> => 
                x != null && typeof x === 'object' && 'slug' in x && 'nameKo' in x
            )
            .map((x) => ({
                slug: String(x.slug),
                title: `${String(x.nameKo)}${x.nameEn != null ? ` (${String(x.nameEn)})` : ''}`,
                oneLine: String(x.oneLine || '')
            }))
            .sort((a, b) => a.title.localeCompare(b.title, 'ko'));
    }

    function build(container: HTMLElement): void {
        container.innerHTML = `
            <div class="ww-wrap">
                <aside class="ww-left">
                    <p class="ww-title">세계관 위키</p>
                    <input class="ww-search" id="wwSearch" placeholder="검색 (이름/한 줄)..." />
                    <div class="ww-list" id="wwList"></div>
                    <p class="ww-hint">초기 버전: 캐릭터 문서만 표시합니다.</p>
                </aside>
                <section class="ww-mid">
                    <div class="ww-doc">
                        <div id="wwDoc" class="ww-md" style="color:var(--text-secondary)">왼쪽에서 항목을 선택하세요.</div>
                    </div>
                </section>
                <aside class="ww-toc">
                    <div id="wwToc" class="ww-toc-list"></div>
                </aside>
            </div>
        `;

        const listEl = container.querySelector('#wwList') as HTMLElement | null;
        const searchEl = container.querySelector('#wwSearch') as HTMLInputElement | null;
        const docEl = container.querySelector('#wwDoc') as HTMLElement | null;
        const tocEl = container.querySelector('#wwToc') as HTMLElement | null;
        if (!listEl || !searchEl || !docEl || !tocEl) return;

        const listRoot = listEl;
        const docRoot = docEl;
        const tocRoot = tocEl;

        const all = getCharacterIndex();
        let activeSlug = '';

        async function open(slug: string): Promise<void> {
            activeSlug = slug;
            listRoot.querySelectorAll('.ww-item').forEach((b) => b.classList.toggle('active', (b as HTMLElement).dataset.slug === slug));
            docRoot.textContent = '문서 불러오는 중...';
            tocRoot.innerHTML = '';
            try {
                const md = await loadWikiText(wikiUrlForCharacter(slug));
                renderWikiMarkdown(docRoot, md, tocRoot);
                const mid = container.querySelector('.ww-mid');
                if (mid instanceof HTMLElement) wireTocActive(tocRoot, mid);
            } catch {
                docRoot.innerHTML = '<p style="color:var(--error)">문서를 불러오지 못했습니다.</p>';
            }
        }

        function renderList(q: string): void {
            const query = (q || '').trim().toLowerCase();
            const items = query
                ? all.filter(x => (x.title + ' ' + x.oneLine).toLowerCase().includes(query))
                : all;
            listRoot.innerHTML = items.map(x => (
                `<button type="button" class="ww-item" data-slug="${esc(x.slug)}">` +
                `<div style="font-weight:700">${esc(x.title)}</div>` +
                (x.oneLine ? `<div style="margin-top:2px;color:var(--text-tertiary);font-size:12px;">${esc(x.oneLine)}</div>` : '') +
                `</button>`
            )).join('');
            listRoot.querySelectorAll('.ww-item').forEach((btn) => {
                const el = btn as HTMLElement;
                el.addEventListener('click', () => void open(el.dataset.slug || ''));
                if (el.dataset.slug === activeSlug) el.classList.add('active');
            });
        }

        renderList('');
        searchEl.addEventListener('input', () => renderList(searchEl.value));
        if (all[0]) void open(all[0].slug);
    }

    Toolbox.register({
        ...(Toolbox.getLazyWidgetPublicMeta?.('worldwiki') ?? {}),
        tabs: [{ id: 'worldwiki-main', label: '위키', build }]
    });
})();

