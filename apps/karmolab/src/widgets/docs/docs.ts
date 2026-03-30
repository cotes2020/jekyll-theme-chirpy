// @ts-nocheck
/**
 * Docs — 소개, 로드맵·기획, 가이드
 *
 * marked.js로 마크다운 렌더링, Prism.js로 코드 하이라이팅.
 * 탭: 소개 | 로드맵 & 기획 | 가이드
 */
(function () {

    Mdd.injectCSS('docs', `
        .docs-body { font-size:14px; line-height:1.8; color:var(--text-primary); max-width:800px; }
        .docs-body h1 { font-size:24px; font-weight:800; letter-spacing:-0.03em; margin:0 0 16px; padding-bottom:12px; border-bottom:2px solid var(--border); }
        .docs-body h2 { font-size:18px; font-weight:700; letter-spacing:-0.02em; margin:32px 0 12px; color:var(--accent); }
        .docs-body h3 { font-size:15px; font-weight:600; margin:24px 0 8px; }
        .docs-body p { margin:0 0 12px; color:var(--text-secondary); }
        .docs-body ul, .docs-body ol { margin:0 0 12px; padding-left:24px; color:var(--text-secondary); }
        .docs-body li { margin-bottom:4px; }
        .docs-body code { font-family:var(--font-mono); font-size:var(--font-size-xs); background:var(--bg-tertiary); padding:2px 6px; border-radius:4px; color:var(--accent); }
        .docs-body pre { margin:0 0 16px; border-radius:var(--radius-md); overflow-x:auto; border:1px solid var(--border); }
        .docs-body pre code { display:block; padding:16px; background:var(--bg-tertiary); color:var(--text-primary); border-radius:var(--radius-md); font-size:var(--font-size-xs); line-height:1.6; }
        .docs-body table { width:100%; border-collapse:collapse; margin:0 0 16px; font-size:var(--font-size-sm); }
        .docs-body th { text-align:left; padding:8px 12px; background:var(--bg-tertiary); border:1px solid var(--border); font-weight:600; color:var(--text-primary); font-size:var(--font-size-xs); text-transform:uppercase; letter-spacing:0.06em; }
        .docs-body td { padding:8px 12px; border:1px solid var(--border); color:var(--text-secondary); }
        .docs-body blockquote { margin:0 0 16px; padding:12px 16px; border-left:3px solid var(--accent); background:var(--accent-subtle); border-radius:0 var(--radius-sm) var(--radius-sm) 0; color:var(--text-secondary); }
        .docs-body blockquote p { margin:0; }
        .docs-body hr { border:none; border-top:1px solid var(--border); margin:24px 0; }
        .docs-body a { color:var(--accent); text-decoration:none; }
        .docs-body a:hover { text-decoration:underline; }
        .docs-body strong { color:var(--text-primary); }
        .docs-toc { position:sticky; top:0; background:var(--bg-secondary); padding:12px 0; margin-bottom:16px; z-index:2; }
        .docs-toc-list { display:flex; flex-wrap:wrap; gap:6px; }
        .docs-toc-item { font-size:var(--font-size-xs); padding:4px 12px; border-radius:100px; background:var(--bg-tertiary); color:var(--text-secondary); cursor:pointer; border:1px solid var(--border); transition:all 0.15s; text-decoration:none; }
        .docs-toc-item:hover { color:var(--text-primary); border-color:var(--accent); }
    `);

    /* ===== 문서 경로 (스크립트 위치 기준) ===== */
    const DOCS_BASE = (function () {
        const script = document.currentScript;
        if (script && script.src) {
            const url = new URL(script.src);
            return url.origin + url.pathname.replace(/\/[^/]+$/, '/');
        }
        return './js/widgets/docs/';
    })();

    function getDocUrl(filename) {
        return DOCS_BASE + filename;
    }

    function loadDoc(filename) {
        return fetch(getDocUrl(filename)).then(function (r) {
            if (!r.ok) throw new Error('문서 로드 실패: ' + filename);
            return r.text();
        });
    }

    /* ===== 렌더러 ===== */
    function renderMarkdown(container, md) {
        const body = document.createElement('div');
        body.className = 'docs-body';

        if (typeof marked !== 'undefined') {
            marked.setOptions({ breaks: true, gfm: true });
            body.innerHTML = marked.parse(md);
        } else {
            body.innerHTML = '<p style="color:var(--error)">marked.js 로드 실패. 새로고침해주세요.</p>';
            return;
        }

        container.innerHTML = '';
        container.appendChild(body);

        body.querySelectorAll('pre code').forEach(block => {
            const text = block.textContent;
            const lang = block.className.match(/language-(\w+)/)?.[1] || 'javascript';
            block.className = 'language-' + lang;
            if (typeof Prism !== 'undefined') {
                Prism.highlightElement(block);
            }
        });
    }

    /* ===== 위젯 등록 ===== */
    Toolbox.register({
        id: 'docs',
        title: '문서',
        category: null,  // 기타
        desc: 'KarmoLab 소개, 로드맵·기획, 가이드',
        layout: 'wide',
        icon: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
        tabs: [
            { id: 'docs-intro', label: '소개', build: function (c) { Mdd.linePreset('tool_run', { msg: '문서 페이지예요!' }); c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>'; loadDoc('intro.md').then(function (md) { renderMarkdown(c, md); }).catch(function () { renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*'); }); } },
            { id: 'docs-roadmap', label: '로드맵', build: function (c) { Mdd.linePreset('daily_start', { msg: '로드맵이랑 기획이에요~' }); c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>'; loadDoc('roadmap.md').then(function (md) { renderMarkdown(c, md); }).catch(function () { renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*'); }); } },
            { id: 'docs-guide', label: '가이드', build: function (c) { Mdd.linePreset('tool_run', { msg: '사용법을 알려줄게요~' }); c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>'; loadDoc('guide.md').then(function (md) { renderMarkdown(c, md); }).catch(function () { renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*'); }); } },
        ]
    });
})();
