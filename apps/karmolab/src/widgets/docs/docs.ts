/**
 * Docs — 소개, 로드맵·기획, 가이드, 프로젝트 통합 명령
 *
 * marked.js로 마크다운 렌더링, Prism.js로 코드 하이라이팅, ```mermaid 는 Mermaid 렌더.
 */
(function (): void {
  /** 동일 출처(Tracking Prevention 회피). CDN 금지 — `js/vendor/mermaid.min.js` */
  function getMermaidScriptUrl(): string {
    const w = window as unknown as { KARMOLAB_WIDGET_SCRIPT_BASE?: string };
    if (w.KARMOLAB_WIDGET_SCRIPT_BASE) {
      try {
        return new URL('../vendor/mermaid.min.js', w.KARMOLAB_WIDGET_SCRIPT_BASE).href;
      } catch {
        /* noop */
      }
    }
    const cur = document.currentScript as HTMLScriptElement | null;
    if (cur?.src) {
      try {
        return new URL('../vendor/mermaid.min.js', cur.src).href;
      } catch {
        /* noop */
      }
    }
    return (typeof location !== 'undefined' ? location.origin : '') + '/apps/karmolab/js/vendor/mermaid.min.js';
  }

  type MermaidApi = {
    initialize: (config: Record<string, unknown>) => void;
    run: (opts?: { nodes?: Iterable<Element> | null }) => Promise<unknown>;
  };

  let mermaidScriptPromise: Promise<void> | null = null;
  let markedMermaidRegistered = false;

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

  function renderMarkdown(container: HTMLElement, md: string): void {
    const body = document.createElement('div');
    body.className = 'docs-body';

    if (typeof marked !== 'undefined') {
      registerMarkedMermaid();
      marked.setOptions({ breaks: true, gfm: true });
      body.innerHTML = marked.parse(md);
    } else {
      body.innerHTML = '<p style="color:var(--error)">marked.js 로드 실패. 새로고침해주세요.</p>';
      return;
    }

    container.innerHTML = '';
    container.appendChild(body);

    replaceMermaidCodeBlocksFallback(body);

    body.querySelectorAll('pre code').forEach((block: Element) => {
      const lang = block.className.match(/language-(\w+)/)?.[1] || 'javascript';
      block.className = 'language-' + lang;
      if (typeof Prism !== 'undefined') {
        Prism.highlightElement(block);
      }
    });

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
          mm.initialize({
            startOnLoad: false,
            theme: isDark ? 'dark' : 'default',
            securityLevel: 'loose',
          });
          await mm.run({ nodes: Array.from(mermaidEls) });
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

  Toolbox.register({
    id: 'docs',
    title: '문서',
    desc: 'KarmoLab 소개, 로드맵·기획, 가이드, Discord 봇 문서, 레포 명령',
    layout: 'wide',
    icon: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    tabs: [
      {
        id: 'docs-intro',
        label: '소개',
        build: function (c: HTMLElement): void {
          Mdd.linePreset('tool_run', { msg: '문서 페이지예요!' });
          c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>';
          loadDoc('intro.md')
            .then(function (md: string) {
              renderMarkdown(c, md);
            })
            .catch(function () {
              renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*');
            });
        }
      },
      {
        id: 'docs-roadmap',
        label: '로드맵',
        build: function (c: HTMLElement): void {
          Mdd.linePreset('daily_start', { msg: '로드맵이랑 기획이에요~' });
          c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>';
          loadDoc('roadmap.md')
            .then(function (md: string) {
              renderMarkdown(c, md);
            })
            .catch(function () {
              renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*');
            });
        }
      },
      {
        id: 'docs-guide',
        label: '가이드',
        build: function (c: HTMLElement): void {
          Mdd.linePreset('tool_run', { msg: '사용법을 알려줄게요~' });
          c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>';
          loadDoc('guide.md')
            .then(function (md: string) {
              renderMarkdown(c, md);
            })
            .catch(function () {
              renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*');
            });
        }
      },
      {
        id: 'docs-discord-voice',
        label: 'Discord·음성',
        build: function (c: HTMLElement): void {
          Mdd.linePreset('tool_run', { msg: '욘 봇 `/sound`·음성·DAVE 트러블슈팅 정리예요.' });
          c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>';
          loadDoc('discord-voice-dave-troubleshooting.md')
            .then(function (md: string) {
              renderMarkdown(c, md);
            })
            .catch(function () {
              renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*');
            });
        }
      },
      {
        id: 'docs-discord-improvements',
        label: 'Discord·개선',
        build: function (c: HTMLElement): void {
          Mdd.linePreset('tool_run', { msg: '욘 봇 개선 아이디어 백로그예요.' });
          c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>';
          loadDoc('discord-bot-improvements.md')
            .then(function (md: string) {
              renderMarkdown(c, md);
            })
            .catch(function () {
              renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*');
            });
        }
      },
      {
        id: 'docs-project-commands',
        label: '프로젝트 명령',
        build: function (c: HTMLElement): void {
          Mdd.linePreset('tool_run', { msg: '블로그·KarmoLab·앱 전체 명령을 모아 뒀어요. 복사해서 쓰기 좋게!' });
          c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>';
          loadDoc('project-commands-guide.md')
            .then(function (md: string) {
              renderMarkdown(c, md);
            })
            .catch(function () {
              renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*');
            });
        }
      },
      {
        id: 'docs-ai-commonization',
        label: 'AI 공통화',
        build: function (c: HTMLElement): void {
          Mdd.linePreset('tool_run', { msg: 'Gemini·봇·카카오 스크립트 공통화 기획이에요.' });
          c.innerHTML = '<p class="docs-body" style="color:var(--text-secondary)">문서 불러오는 중...</p>';
          loadDoc('ai-commonization-plan.md')
            .then(function (md: string) {
              renderMarkdown(c, md);
            })
            .catch(function () {
              renderMarkdown(c, '*문서를 불러오지 못했어요. 새로고침해 주세요.*');
            });
        }
      }
    ]
  });
})();
