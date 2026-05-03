/**
 * Docs — 소개, 로드맵·기획, 가이드, 프로젝트 통합 명령
 *
 * marked.js로 마크다운 렌더링, Prism.js로 코드 하이라이팅, ```mermaid 는 Mermaid 렌더.
 */
(function (): void {
  /** 동일 출처(Tracking Prevention 회피). CDN 금지 — Jekyll 블로그 `assets/lib/mermaid/mermaid.min.js` 와 공유 */
  function getMermaidScriptUrl(): string {
    const w = window as unknown as { KARMOLAB_WIDGET_SCRIPT_BASE?: string };
    if (w.KARMOLAB_WIDGET_SCRIPT_BASE) {
      try {
        return new URL('../../../../assets/lib/mermaid/mermaid.min.js', w.KARMOLAB_WIDGET_SCRIPT_BASE).href;
      } catch {
        /* noop */
      }
    }
    const cur = document.currentScript as HTMLScriptElement | null;
    if (cur?.src) {
      try {
        return new URL('../../../../../assets/lib/mermaid/mermaid.min.js', cur.src).href;
      } catch {
        /* noop */
      }
    }
    return (typeof location !== 'undefined' ? location.origin : '') + '/assets/lib/mermaid/mermaid.min.js';
  }

  type MermaidApi = {
    initialize: (config: Record<string, unknown>) => void;
    run: (opts?: { nodes?: Iterable<Element> | null; suppressErrors?: boolean }) => Promise<unknown>;
    render?: (
      id: string,
      text: string
    ) => Promise<{ svg: string; bindFunctions?: (element: Element) => void }>;
  };

  let mermaidScriptPromise: Promise<void> | null = null;
  let markedMermaidRegistered = false;
  /** 마지막으로 적용한 Mermaid 설정(불필요한 재초기화 방지) */
  let mermaidInitKey: string | null = null;

  /** UMD/ESM 번들 모두: default 또는 루트에 API 가 올 수 있음 */
  function getMermaidApi(): MermaidApi | null {
    const w = window as unknown as { mermaid?: MermaidApi & { default?: MermaidApi } };
    const root = w.mermaid;
    if (!root) return null;
    if (typeof root.initialize === 'function' && typeof root.run === 'function') return root;
    const d = root.default;
    if (d && typeof d.initialize === 'function' && typeof d.run === 'function') return d;
    return null;
  }

  /** `run()`이 조용히 실패하는 환경 대비: `render`로 SVG를 대상 요소에 직접 넣음 */
  async function paintMermaidElements(mm: MermaidApi, elements: Element[]): Promise<void> {
    const render = mm.render;
    if (typeof render === 'function') {
      let n = 0;
      for (const el of elements) {
        if (!(el instanceof HTMLElement)) continue;
        const text = (el.textContent || '').replace(/\uFEFF/g, '').trim();
        if (!text) continue;
        const id = 'kl-mmd-' + Date.now() + '-' + ++n;
        const { svg, bindFunctions } = await render.call(mm, id, text);
        el.innerHTML = svg;
        bindFunctions?.(el);
      }
      return;
    }
    await mm.run({ nodes: elements, suppressErrors: false });
  }

  function loadMermaidScript(): Promise<void> {
    if (getMermaidApi()) return Promise.resolve();
    if (mermaidScriptPromise) return mermaidScriptPromise;
    mermaidScriptPromise = new Promise(function (resolve, reject) {
      const s = document.createElement('script');
      s.src = getMermaidScriptUrl();
      s.async = true;
      s.onload = function () {
        resolve();
      };
      s.onerror = function () {
        mermaidScriptPromise = null;
        reject(new Error('Mermaid 스크립트 로드 실패'));
      };
      document.head.appendChild(s);
    });
    return mermaidScriptPromise;
  }

  /** marked 기본 코드 블록(HTML 이스케이프)을 거치지 않고 ```mermaid 원문을 div 에 넣음 */
  function registerMarkedMermaid(): void {
    if (markedMermaidRegistered) return;
    markedMermaidRegistered = true;
    const mk = typeof marked !== 'undefined' ? (marked as { use?: (opts: unknown) => void }) : null;
    if (!mk || typeof mk.use !== 'function') return;
    mk.use({
      extensions: [
        {
          name: 'mermaid',
          level: 'block',
          start(src: string) {
            const m = /^```mermaid[ \t]*(?:\r?\n|$)/m.exec(src);
            return m ? m.index : undefined;
          },
          tokenizer(src: string) {
            const r = /^```mermaid[ \t]*\r?\n([\s\S]*?)\r?\n```/;
            const m = r.exec(src);
            if (!m) return undefined;
            return {
              type: 'mermaid',
              raw: m[0],
              text: m[1].replace(/\r\n/g, '\n').trim(),
            };
          },
          renderer(token: { text: string }) {
            const safe = token.text.replace(/&/g, '&amp;').replace(/</g, '&lt;');
            return '<div class="mermaid">' + safe + '</div>\n';
          },
        },
      ],
    });
  }

  /** 펜스가 marked 확장을 타지 않았을 때(구버전 등) 대비 */
  function replaceMermaidCodeBlocksFallback(body: HTMLElement): number {
    let n = 0;
    body.querySelectorAll('pre > code.language-mermaid, pre > code[class*="language-mermaid"]').forEach(function (code) {
      const pre = code.parentElement;
      if (!pre) return;
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = (code.textContent || '').replace(/\r\n/g, '\n').trim();
      pre.replaceWith(div);
      n++;
    });
    return n;
  }
  Mdd.injectCSS(
    'docs',
    `
        /* 본문 + 목차 (세계관 위키 위젯과 같은 패턴: 슬러그 앵커, 측면 nav, IntersectionObserver) */
        .docs-md-layout {
          display:grid;
          grid-template-columns:minmax(0,1fr) 220px;
          grid-template-areas:'main toc';
          gap:16px;
          align-items:start;
        }
        .docs-md-layout--no-toc { grid-template-columns:1fr; grid-template-areas:'main'; }
        .docs-md-layout--no-toc .docs-md-toc { display:none; }
        .docs-md-main { grid-area:main; min-width:0; }
        .docs-md-toc {
          grid-area:toc;
          position:sticky;
          top:12px;
          align-self:start;
          max-height:min(72vh, 640px);
          overflow:auto;
          padding:12px;
          background:var(--bg-secondary);
          border:1px solid var(--border);
          border-radius:var(--radius-lg);
        }
        .docs-toc-title { font-size:var(--font-size-xs); font-weight:900; color:var(--text-secondary); margin:2px 0 10px; }
        .docs-toc-listnav { display:flex; flex-direction:column; gap:6px; }
        .docs-toc-a {
          font-size:12px;
          color:var(--text-tertiary);
          text-decoration:none;
          line-height:1.45;
          padding:6px 8px;
          border-radius:10px;
          border:1px solid transparent;
        }
        .docs-toc-a:hover { color:var(--text-secondary); border-color:var(--border); background:var(--bg-tertiary); }
        .docs-toc-a.active { color:var(--text-primary); border-color:var(--accent); box-shadow:0 0 0 2px var(--accent-subtle); }
        .docs-toc-l2 { padding-left:18px; }
        .docs-toc-l3 { padding-left:28px; }
        .docs-heading { position:relative; scroll-margin-top:16px; }
        .docs-heading:hover .docs-anchor { opacity:1; }
        .docs-anchor {
          opacity:0;
          position:absolute;
          left:-22px;
          top:50%;
          transform:translateY(-50%);
          font-size:14px;
          color:var(--text-tertiary);
          cursor:pointer;
          user-select:none;
        }
        .docs-anchor:focus { opacity:1; outline:2px solid var(--accent-subtle); outline-offset:2px; border-radius:6px; }
        @media (max-width:920px) {
          .docs-md-layout:not(.docs-md-layout--no-toc) {
            grid-template-columns:1fr;
            grid-template-areas:'toc' 'main';
          }
          .docs-md-toc { max-height:min(38vh, 280px); top:0; z-index:3; }
        }
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
        .docs-body .mermaid { margin:0 0 16px; padding:12px; border-radius:var(--radius-md); border:1px solid var(--border); background:var(--bg-tertiary); overflow-x:auto; text-align:center; }
        .docs-body .mermaid svg { max-width:100%; height:auto; }
    `
  );

  const DOCS_BASE = (function (): string {
    const w = window as unknown as { KARMOLAB_WIDGET_SCRIPT_BASE?: string };
    if (w.KARMOLAB_WIDGET_SCRIPT_BASE) {
      return w.KARMOLAB_WIDGET_SCRIPT_BASE + 'docs/';
    }
    const script = document.currentScript;
    if (script && 'src' in script && script.src) {
      const url = new URL(script.src);
      return url.origin + url.pathname.replace(/\/[^/]+$/, '/');
    }
    return (typeof location !== 'undefined' ? location.origin : '') + '/apps/karmolab/js/widgets/docs/';
  })();

  function getDocUrl(filename: string): string {
    return DOCS_BASE + filename;
  }

  function loadDoc(filename: string): Promise<string> {
    return fetch(getDocUrl(filename)).then(function (r: Response) {
      if (!r.ok) throw new Error('문서 로드 실패: ' + filename);
      return r.text();
    });
  }

  /**
   * GitHub `raw.githubusercontent.com` 등 — 레포 루트 기준 상대 경로 Markdown.
   * 기본: 이 사이트 레포 `master`. 포크·다른 브랜치는 `window.KARMOLAB_DOCS_RAW_BASE`로 덮어쓰기
   * (끝에 `/` 포함한 전체 prefix, 예: https://raw.githubusercontent.com/you/repo/main/)
   */
  function getDocsRepoRawBase(): string {
    const w = window as unknown as { KARMOLAB_DOCS_RAW_BASE?: string };
    const custom = (w.KARMOLAB_DOCS_RAW_BASE ?? '').trim();
    if (custom) {
      return custom.replace(/\/?$/, '/');
    }
    return 'https://raw.githubusercontent.com/mascari4615/mascari4615.github.io/master/';
  }

  function normalizeRepoDocPath(path: string): string {
    return path
      .trim()
      .replace(/^\/+/, '')
      .replace(/\/{2,}/g, '/');
  }

  function loadDocFromRepo(repoRelativePath: string): Promise<string> {
    const url = getDocsRepoRawBase() + normalizeRepoDocPath(repoRelativePath);
    return fetch(url).then(function (r: Response) {
      if (!r.ok) throw new Error('레포 문서 로드 실패: ' + repoRelativePath + ' (' + r.status + ')');
      return r.text();
    });
  }

  /** worldwiki 위젯과 동일한 슬러그·앵커·목차 패턴 */
  function docsEsc(s: string): string {
    return typeof Toolbox !== 'undefined' && Toolbox.escapeHtml ? Toolbox.escapeHtml(s) : s;
  }

  function docsSlugify(s: string): string {
    return String(s || '')
      .trim()
      .toLowerCase()
      .replace(/[\s]+/g, '-')
      .replace(/[^\w\-가-힣]+/g, '')
      .replace(/\-+/g, '-')
      .replace(/^\-+|\-+$/g, '');
  }

  function docsEnsureUniqueId(base: string, used: Set<string>): string {
    let id = base || 'section';
    if (!used.has(id)) {
      used.add(id);
      return id;
    }
    let i = 2;
    while (used.has(`${id}-${i}`)) i++;
    const out = `${id}-${i}`;
    used.add(out);
    return out;
  }

  function findDocsScrollRoot(from: HTMLElement): Element | null {
    let el: HTMLElement | null = from.parentElement;
    for (let i = 0; i < 16 && el; i++) {
      const st = window.getComputedStyle(el);
      const oy = st.overflowY;
      if ((oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 4) {
        return el;
      }
      el = el.parentElement;
    }
    return null;
  }

  function applyDocsAnchors(
    root: HTMLElement,
    tocEl: HTMLElement | null
  ): Array<{ id: string; text: string; level: number }> {
    const used = new Set<string>();
    const headings = Array.from(root.querySelectorAll('h1, h2, h3'));
    const toc: Array<{ id: string; text: string; level: number }> = [];

    headings.forEach(function (h) {
      const el = h as HTMLElement;
      const level = el.tagName === 'H1' ? 1 : el.tagName === 'H2' ? 2 : 3;
      const text = (el.textContent || '').trim();
      if (!text) return;
      const id = docsEnsureUniqueId(docsSlugify(text), used);
      el.id = el.id || id;
      el.classList.add('docs-heading');

      const a = document.createElement('span');
      a.className = 'docs-anchor';
      a.tabIndex = 0;
      a.setAttribute('role', 'button');
      a.setAttribute('aria-label', '링크 복사');
      a.textContent = '#';
      const copy = async function () {
        const url = location.origin + location.pathname + location.search + '#' + el.id;
        try {
          await navigator.clipboard.writeText(url);
          Toolbox.showToast?.('링크 복사됨', undefined, undefined);
        } catch {
          location.hash = el.id;
          Toolbox.showToast?.('링크를 복사하지 못했습니다.', 'error', undefined);
        }
      };
      a.addEventListener('click', function (e: MouseEvent) {
        e.preventDefault();
        e.stopPropagation();
        location.hash = el.id;
        void copy();
      });
      a.addEventListener('keydown', function (e: KeyboardEvent) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          location.hash = el.id;
          void copy();
        }
      });
      el.prepend(a);

      toc.push({ id: el.id, text, level });
    });

    if (tocEl && toc.length >= 2) {
      tocEl.innerHTML =
        '<div class="docs-toc-title">이 문서 목차</div>' +
        '<nav class="docs-toc-listnav" aria-label="목차">' +
        toc
          .map(function (x) {
            return (
              '<a class="docs-toc-a docs-toc-l' +
              x.level +
              '" href="#' +
              docsEsc(x.id) +
              '">' +
              docsEsc(x.text) +
              '</a>'
            );
          })
          .join('') +
        '</nav>';
    } else if (tocEl) {
      tocEl.innerHTML = '';
    }

    return toc;
  }

  function wireDocsTocActive(tocWrap: HTMLElement, docWrap: HTMLElement, scrollRoot: Element | null): void {
    const links = Array.from(tocWrap.querySelectorAll('.docs-toc-a'));
    if (!links.length) return;

    const headings = Array.from(docWrap.querySelectorAll('h1, h2, h3')).filter(function (h) {
      return !!h.id;
    });
    const ioRoot = scrollRoot ?? null;
    const obs = new IntersectionObserver(
      function (entries) {
        const visible = entries
          .filter(function (e) {
            return e.isIntersecting;
          })
          .sort(function (a, b) {
            return a.boundingClientRect.top - b.boundingClientRect.top;
          });
        if (!visible.length) return;
        const id = visible[0].target.id;
        links.forEach(function (a) {
          a.classList.toggle('active', a.getAttribute('href') === '#' + id);
        });
      },
      { root: ioRoot, threshold: [0.12, 0.35, 0.55] }
    );
    headings.forEach(function (h) {
      obs.observe(h);
    });

    links.forEach(function (a) {
      a.addEventListener('click', function () {
        const href = a.getAttribute('href');
        const id = href && href.startsWith('#') ? href.slice(1) : '';
        let target: HTMLElement | null = null;
        try {
          target = id ? (docWrap.querySelector('#' + CSS.escape(id)) as HTMLElement | null) : null;
        } catch {
          target = null;
        }
        if (!target) return;
        const root = scrollRoot;
        if (root instanceof HTMLElement) {
          const top = target.getBoundingClientRect().top - root.getBoundingClientRect().top + root.scrollTop - 10;
          root.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
        } else {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  }

  /** 레포 루트 기준 Markdown 경로를 GitHub raw로 불러와 본문 위에 출처 블록을 붙여 렌더 */
  function renderRepoMarkdownInContainer(container: HTMLElement, repoRelativePath: string): void {
    container.innerHTML =
      '<p class="docs-body" style="color:var(--text-secondary)">GitHub에서 문서 불러오는 중...</p>';
    loadDocFromRepo(repoRelativePath)
      .then(function (md: string) {
        const banner =
          '> **원본:** `' +
          repoRelativePath +
          '` — GitHub **raw** (`master` 기본). `window.KARMOLAB_DOCS_RAW_BASE` 에 끝이 `/`인 URL을 넣으면 다른 브랜치·포크를 볼 수 있어요.\n\n---\n\n';
        renderMarkdown(container, banner + md);
      })
      .catch(function () {
        renderMarkdown(
          container,
          '*문서를 불러오지 못했어요. 네트워크·브랜치(`master`)·`window.KARMOLAB_DOCS_RAW_BASE` 를 확인한 뒤 탭을 다시 열어 주세요.*',
        );
      });
  }

  function renderMarkdown(container: HTMLElement, md: string): void {
    const body = document.createElement('div');
    body.className = 'docs-body';
    md = md.replace(/^\uFEFF/, '');

    if (typeof marked === 'undefined') {
      container.innerHTML = '<p class="docs-body" style="color:var(--error)">marked.js 로드 실패. 새로고침해주세요.</p>';
      return;
    }

    registerMarkedMermaid();
    marked.setOptions({ breaks: true, gfm: true });
    body.innerHTML = marked.parse(md);

    const layout = document.createElement('div');
    layout.className = 'docs-md-layout';

    const main = document.createElement('div');
    main.className = 'docs-md-main';
    main.appendChild(body);

    const aside = document.createElement('aside');
    aside.className = 'docs-md-toc';
    aside.setAttribute('aria-label', '문서 목차');
    const tocNav = document.createElement('div');
    tocNav.className = 'docs-toc-nav-host';
    aside.appendChild(tocNav);

    layout.appendChild(main);
    layout.appendChild(aside);

    container.innerHTML = '';
    container.appendChild(layout);

    replaceMermaidCodeBlocksFallback(body);

    body.querySelectorAll('pre code').forEach((block: Element) => {
      const lang = block.className.match(/language-(\w+)/)?.[1] || 'javascript';
      block.className = 'language-' + lang;
      if (typeof Prism !== 'undefined') {
        Prism.highlightElement(block);
      }
    });

    const tocMeta = applyDocsAnchors(body, tocNav);
    if (tocMeta.length < 2) {
      layout.classList.add('docs-md-layout--no-toc');
    } else {
      wireDocsTocActive(tocNav, body, findDocsScrollRoot(layout));
    }

    const mermaidEls = body.querySelectorAll('.mermaid');
    if (mermaidEls.length > 0) {
      void (async function () {
        try {
          await loadMermaidScript();
          const mm = getMermaidApi();
          if (!mm) {
            console.error('[docs mermaid] window.mermaid API 없음');
            return;
          }
          const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
          const theme = isDark ? 'dark' : 'default';
          const initKey = theme + '|loose';
          if (mermaidInitKey !== initKey) {
            mermaidInitKey = initKey;
            mm.initialize({
              startOnLoad: false,
              theme,
              securityLevel: 'loose',
            });
          }
          await paintMermaidElements(mm, Array.from(mermaidEls));
        } catch (e) {
          console.error('[docs mermaid]', e);
          body.insertAdjacentHTML(
            'beforeend',
            '<p class="docs-body" style="color:var(--error)">다이어그램(Mermaid)을 그리지 못했습니다. 콘솔(F12)에 오류가 있는지, CDN·문법을 확인해 주세요.</p>',
          );
        }
      })();
    }
  }

  // ── TASK-KL-015-B: 통합 문서 위젯 ─────────────────────────────────────────────────────────
  // docs.ts 가 자체 사이드바 (그룹 헤더 + 동적 항목) + 본문 + TOC 그림. Toolbox tabs 단일.
  // 「프로젝트 문서」 그룹 = 외부 md/GitHub raw (현 hardcode).
  // 「캐릭터/시스템/개념/lore」 그룹 = `world/wiki/manifest.json` 동적 walk (sub-A 의 sync 결과).
  type ExternalSource =
    | { kind: 'local'; path: string }
    | { kind: 'github'; path: string };

  interface ExternalDoc {
    id: string;
    label: string;
    source: ExternalSource;
    mddPreset?: string;
    mddMsg?: string;
  }

  const EXTERNAL_DOCS: ExternalDoc[] = [
    { id: 'docs-intro', label: '소개', source: { kind: 'local', path: 'intro.md' }, mddPreset: 'tool_run', mddMsg: '문서 페이지예요!' },
    { id: 'docs-roadmap', label: '로드맵', source: { kind: 'local', path: 'roadmap.md' }, mddPreset: 'daily_start', mddMsg: '로드맵이랑 기획이에요~' },
    { id: 'docs-guide', label: '가이드', source: { kind: 'local', path: 'guide.md' }, mddPreset: 'tool_run', mddMsg: '사용법을 알려줄게요~' },
    { id: 'docs-karmolab-ai', label: 'KarmoLabAI', source: { kind: 'local', path: 'karmolab-ai.md' }, mddPreset: 'tool_run', mddMsg: 'karmolab-ai 패키지 쓰는 법이에요.' },
    { id: 'docs-discord-yawnbot', label: 'Discord·욘봇', source: { kind: 'local', path: 'discord-yawnbot.md' }, mddPreset: 'tool_run', mddMsg: '욘 봇 음성·DAVE·기능 요약·TODO 한곳이에요.' },
    { id: 'docs-discord-bots-readme', label: 'discord-bots · README', source: { kind: 'github', path: 'apps/discord-bots/README.md' }, mddPreset: 'tool_run', mddMsg: 'discord-bots 워크스페이스 README (GitHub).' },
    { id: 'docs-tauri-readme', label: 'Tauri · README', source: { kind: 'github', path: 'apps/karmolab-tauri/README.md' }, mddPreset: 'tool_run', mddMsg: '데스크톱 앱 폴더 README (GitHub).' },
    { id: 'docs-project-commands', label: '프로젝트 명령', source: { kind: 'local', path: 'project-commands-guide.md' }, mddPreset: 'tool_run', mddMsg: '블로그·KarmoLab·앱 전체 명령을 모아 뒀어요. 복사해서 쓰기 좋게!' },
    { id: 'docs-local-dev', label: '데스크톱·로컬', source: { kind: 'local', path: 'local-dev-runner.md' }, mddPreset: 'tool_run', mddMsg: 'Tauri 앱에서만 쓰는 로컬 데브 러너 안내예요.' },
    { id: 'docs-servermonitor-deploy-log-design', label: '로컬 · deploy 로그', source: { kind: 'local', path: 'servermonitor-deploy-log-stream.md' }, mddPreset: 'tool_run', mddMsg: '서버 모니터 deploy·npm i 로그 스트림 — 이벤트·커맨드는 본문 참고.' },
  ];

  interface EntityManifestItem { slug: string; title: string; oneLine: string; tags: string[]; }
  interface DocsManifest {
    characters: EntityManifestItem[];
    systems: EntityManifestItem[];
    concepts: EntityManifestItem[];
    lore: EntityManifestItem[];
  }

  const ENTITY_GROUPS: Array<{ key: keyof DocsManifest; label: string; dirName: string }> = [
    { key: 'characters', label: '캐릭터 위키', dirName: 'characters' },
    { key: 'systems', label: '시스템 / 흐름', dirName: 'systems' },
    { key: 'concepts', label: '개념', dirName: 'concepts' },
    { key: 'lore', label: '세계관 lore', dirName: 'lore' },
  ];

  /** DOCS_BASE = .../js/widgets/docs/. wiki 산출물은 .../world/wiki/. */
  function wikiBaseUrl(): string {
    return DOCS_BASE.replace(/\/js\/widgets\/docs\/?$/, '/world/wiki/');
  }

  function loadManifest(): Promise<DocsManifest> {
    return fetch(wikiBaseUrl() + 'manifest.json').then(function (r: Response) {
      if (!r.ok) throw new Error('manifest 로드 실패: ' + r.status);
      return r.json();
    });
  }

  function loadEntityMd(dirName: string, slug: string): Promise<string> {
    const url = wikiBaseUrl() + 'entities/' + dirName + '/' + slug + '.md';
    return fetch(url).then(function (r: Response) {
      if (!r.ok) throw new Error('entity md 실패: ' + slug);
      return r.text();
    });
  }

  function injectShellStyles(): void {
    if (document.getElementById('docs-shell-styles')) return;
    const style = document.createElement('style');
    style.id = 'docs-shell-styles';
    style.textContent = `
      .docs-shell { display: grid; grid-template-columns: 220px 1fr; gap: 16px; min-height: 400px; }
      .docs-shell-side { border-right: 1px solid var(--border); padding-right: 12px; max-height: 80vh; overflow-y: auto; }
      .docs-shell-group { margin-bottom: 14px; }
      .docs-shell-group-label { font-size: 10px; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 6px; padding: 0 4px; font-weight: 600; }
      .docs-shell-list { list-style: none; padding: 0; margin: 0; }
      .docs-shell-list li { margin: 0; }
      .docs-shell-btn { width: 100%; text-align: left; background: transparent; border: 0; padding: 5px 8px; cursor: pointer; font-size: 12.5px; color: var(--text-secondary); border-radius: var(--radius-sm); font-family: inherit; }
      .docs-shell-btn:hover { background: var(--bg-tertiary); color: var(--text-primary); }
      .docs-shell-btn--active { background: var(--accent-subtle); color: var(--accent); font-weight: 600; }
      .docs-shell-content { min-width: 0; }
    `;
    document.head.appendChild(style);
  }

  function buildDocsShell(panel: HTMLElement): void {
    injectShellStyles();
    panel.innerHTML =
      '<div class="docs-shell">'
      + '<aside class="docs-shell-side" data-docs-side></aside>'
      + '<section class="docs-shell-content" data-docs-content></section>'
      + '</div>';
    const sideEl = panel.querySelector<HTMLElement>('[data-docs-side]')!;
    const contentEl = panel.querySelector<HTMLElement>('[data-docs-content]')!;

    let manifest: DocsManifest = { characters: [], systems: [], concepts: [], lore: [] };
    let activeKey: string | null = null;

    function escAttr(s: string): string {
      return docsEsc(s).replace(/"/g, '&quot;');
    }

    function renderSide(): void {
      const html: string[] = [];
      // 외부 문서 그룹 (현재 10 항목 hardcode — 외부 README/가이드 등)
      html.push('<div class="docs-shell-group">');
      html.push('<h3 class="docs-shell-group-label">프로젝트 문서</h3>');
      html.push('<ul class="docs-shell-list">');
      for (const item of EXTERNAL_DOCS) {
        html.push('<li><button type="button" class="docs-shell-btn" data-key="external:' + escAttr(item.id) + '">' + docsEsc(item.label) + '</button></li>');
      }
      html.push('</ul></div>');
      // entity 그룹 (manifest 의 동적 walk — 비면 그룹 자체 숨김)
      for (const group of ENTITY_GROUPS) {
        const items = manifest[group.key];
        if (!items || items.length === 0) continue;
        html.push('<div class="docs-shell-group">');
        html.push('<h3 class="docs-shell-group-label">' + docsEsc(group.label) + '</h3>');
        html.push('<ul class="docs-shell-list">');
        for (const item of items) {
          const titleAttr = escAttr(item.oneLine || '');
          html.push('<li><button type="button" class="docs-shell-btn" data-key="entity:' + escAttr(group.dirName) + ':' + escAttr(item.slug) + '" title="' + titleAttr + '">' + docsEsc(item.title) + '</button></li>');
        }
        html.push('</ul></div>');
      }
      sideEl.innerHTML = html.join('');
      sideEl.querySelectorAll<HTMLButtonElement>('button[data-key]').forEach(function (btn) {
        btn.onclick = function () {
          const key = btn.getAttribute('data-key') || '';
          selectKey(key, btn);
        };
      });
      // 첫 항목 자동 선택 (initial)
      if (activeKey == null) {
        const first = sideEl.querySelector<HTMLButtonElement>('button[data-key]');
        if (first) first.click();
      }
    }

    function selectKey(key: string, btn: HTMLButtonElement): void {
      activeKey = key;
      sideEl.querySelectorAll<HTMLButtonElement>('button[data-key]').forEach(function (b) {
        b.classList.toggle('docs-shell-btn--active', b === btn);
      });
      contentEl.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">불러오는 중...</p>';
      void loadAndRender(key, contentEl);
    }

    async function loadAndRender(key: string, target: HTMLElement): Promise<void> {
      const sep = key.indexOf(':');
      const kind = sep < 0 ? key : key.slice(0, sep);
      const rest = sep < 0 ? '' : key.slice(sep + 1);
      if (kind === 'external') {
        const ext = EXTERNAL_DOCS.find(function (e) { return e.id === rest; });
        if (!ext) {
          renderMarkdown(target, '*항목 없음*');
          return;
        }
        if (ext.mddPreset && ext.mddMsg && typeof Mdd !== 'undefined' && Mdd.linePreset) {
          Mdd.linePreset(ext.mddPreset, { msg: ext.mddMsg });
        }
        if (ext.source.kind === 'local') {
          try {
            const md = await loadDoc(ext.source.path);
            renderMarkdown(target, md);
          } catch (_) {
            renderMarkdown(target, '*문서를 불러오지 못했어요. 새로고침해 주세요.*');
          }
        } else {
          renderRepoMarkdownInContainer(target, ext.source.path);
        }
        return;
      }
      if (kind === 'entity') {
        const colon = rest.indexOf(':');
        const dirName = colon < 0 ? rest : rest.slice(0, colon);
        const slug = colon < 0 ? '' : rest.slice(colon + 1);
        const group = ENTITY_GROUPS.find(function (g) { return g.dirName === dirName; });
        const item = group ? manifest[group.key].find(function (e) { return e.slug === slug; }) : undefined;
        try {
          const md = await loadEntityMd(dirName, slug);
          const title = item?.title || slug;
          const oneLine = item?.oneLine || '';
          const banner = oneLine
            ? '> **' + docsEsc(title) + '** — ' + docsEsc(oneLine) + '\n\n---\n\n'
            : '> **' + docsEsc(title) + '**\n\n---\n\n';
          renderMarkdown(target, banner + md);
        } catch (err) {
          renderMarkdown(target, '*entity 로드 실패: ' + docsEsc(String((err as Error)?.message || err)) + '*');
        }
        return;
      }
      renderMarkdown(target, '*알 수 없는 항목 키: ' + docsEsc(key) + '*');
    }

    // 진입 — manifest fetch (실패해도 외부 문서 그룹은 표시)
    loadManifest()
      .then(function (m) { manifest = m; renderSide(); })
      .catch(function (err) {
        console.warn('[docs] manifest 로드 실패 (entity 그룹 없이 진행):', err);
        renderSide();
      });
  }

  Toolbox.register({
    id: 'docs',
    title: '문서',
    desc: 'KarmoLab 소개·로드맵·가이드 + 캐릭터·시스템 위키 — 사이드바 그룹 내비게이션, 본문 + 목차',
    layout: 'wide',
    icon: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    tabs: [
      {
        id: 'docs',
        label: '문서',
        build: buildDocsShell,
      },
    ],
  });

})();
