/**
 * karmoddrine-flows — Tauri 데스크톱 전용 (category: 'desktop' 자동 게이팅).
 *
 * memo/flows/<slug>.md 의 흐름 문서를 사이드바 + 본문 패널로 렌더.
 * 마크다운 = marked CDN, Mermaid 코드블록 = mermaid CDN (둘 다 lazy).
 * Rust 명령: list_flow_docs / read_flow_doc.
 *
 * github.io 공개 사이트엔 표시 X (category: 'desktop' + invoke 부재 → placeholder).
 */
// @ts-nocheck — Toolbox / window.__TAURI__ 글로벌은 ambient 타입에 다 안 잡혀 있음.
(function (): void {
  interface FlowMeta {
    slug: string;
    title: string;
    summary: string | null;
    category: string | null;
    lastModifiedUnix: number;
    filePath: string;
  }
  interface FlowDoc {
    meta: FlowMeta;
    body: string;
  }
  interface FlowList {
    flows: FlowMeta[];
    generatedAtUnix: number;
    memoPath: string;
    errors: { filePath: string; reason: string }[];
  }

  function isKarmolabDesktop(): boolean {
    return typeof window !== 'undefined' && !!window.__KARMOLAB_DESKTOP__;
  }
  function getInvoke(): ((cmd: string, args?: any) => Promise<any>) | null {
    const i = window.__TAURI__?.core?.invoke;
    return typeof i === 'function' ? i : null;
  }

  async function fetchList(): Promise<FlowList | null> {
    const invoke = getInvoke();
    if (!invoke) return null;
    try { return await invoke('list_flow_docs'); } catch (e) { console.error('list_flow_docs 실패', e); return null; }
  }
  async function fetchDoc(slug: string): Promise<FlowDoc | null> {
    const invoke = getInvoke();
    if (!invoke) return null;
    try { return await invoke('read_flow_doc', { slug }); } catch (e) { console.error('read_flow_doc 실패', e); return null; }
  }

  // marked CDN (한 번만 로드)
  let markedPromise: Promise<any> | null = null;
  function loadMarked(): Promise<any> {
    if (markedPromise) return markedPromise;
    markedPromise = new Promise((resolve, reject) => {
      if ((window as any).marked) { resolve((window as any).marked); return; }
      const script = document.createElement('script');
      script.type = 'module';
      script.textContent = `
        import { marked } from 'https://cdn.jsdelivr.net/npm/marked@13/+esm';
        window.marked = marked;
        window.dispatchEvent(new Event('kf-marked-loaded'));
      `;
      window.addEventListener('kf-marked-loaded', () => resolve((window as any).marked), { once: true });
      setTimeout(() => reject(new Error('marked load timeout')), 15000);
      document.head.appendChild(script);
    });
    return markedPromise;
  }

  // Mermaid CDN — dashboard 와 동일 패턴 (이벤트명만 kf-)
  let mermaidPromise: Promise<any> | null = null;
  function loadMermaid(): Promise<any> {
    if (mermaidPromise) return mermaidPromise;
    mermaidPromise = new Promise((resolve, reject) => {
      if ((window as any).mermaid) { resolve((window as any).mermaid); return; }
      const script = document.createElement('script');
      script.type = 'module';
      script.textContent = `
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        window.mermaid = mermaid;
        window.dispatchEvent(new Event('kf-mermaid-loaded'));
      `;
      window.addEventListener('kf-mermaid-loaded', () => resolve((window as any).mermaid), { once: true });
      setTimeout(() => reject(new Error('mermaid load timeout')), 15000);
      document.head.appendChild(script);
    }).then((m: any) => {
      m.initialize({
        startOnLoad: false,
        theme: 'dark',
        themeVariables: {
          background: '#1a1a1a',
          primaryColor: '#2a2a2a',
          primaryTextColor: '#e8e8e8',
          primaryBorderColor: '#444',
          lineColor: '#888',
          edgeLabelBackground: '#222',
          fontSize: '12px',
        },
      });
      return m;
    });
    return mermaidPromise;
  }

  function esc(s: string): string {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function fmtDate(unix: number): string {
    if (!unix) return '?';
    return new Date(unix * 1000).toLocaleString('ko-KR');
  }

  // ── 폴링/상태 (TDZ 회피용 register 위 선언) ──────────────────────
  let currentSlug: string | null = null;

  Toolbox.register({
    ...Toolbox.getLazyWidgetPublicMeta('karmoddrine-flows'),
    tabs: [
      {
        id: 'app',
        label: 'flows',
        build(container: HTMLElement): void {
          injectStyles();
          renderShell(container);
          void load(container);
        },
      },
    ],
  });

  function injectStyles(): void {
    if (document.getElementById('kf-flows-styles')) return;
    const style = document.createElement('style');
    style.id = 'kf-flows-styles';
    style.textContent = `
      .kf-flows { color: var(--text-primary, #e8e8e8); padding: 16px; max-width: 1500px; margin: 0 auto; display: grid; grid-template-columns: 240px 1fr; gap: 16px; min-height: 400px; }
      .kf-flows .kf-side { border-right: 1px solid var(--border-color, #333); padding-right: 12px; }
      .kf-flows .kf-side h3 { font-size: 12px; color: var(--accent, #d4a849); font-weight: 500; margin: 0 0 8px; }
      .kf-flows .kf-side .kf-cat-group { margin-bottom: 14px; }
      .kf-flows .kf-side .kf-cat-label { font-size: 10px; color: var(--text-tertiary, #888); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
      .kf-flows .kf-side ul { list-style: none; padding: 0; margin: 0; }
      .kf-flows .kf-side li { padding: 4px 6px; margin-bottom: 2px; cursor: pointer; font-size: 12px; border-radius: 2px; }
      .kf-flows .kf-side li:hover { background: rgba(255,255,255,0.05); }
      .kf-flows .kf-side li.kf-active { background: var(--accent, #d4a849); color: #000; }
      .kf-flows .kf-meta { font-size: 11px; color: var(--text-tertiary, #888); margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px dashed var(--border-color, #333); }
      .kf-flows .kf-body { line-height: 1.55; font-size: 13.5px; }
      .kf-flows .kf-body h1, .kf-flows .kf-body h2 { color: var(--accent, #d4a849); border-bottom: 1px solid var(--border-color, #333); padding-bottom: 4px; }
      .kf-flows .kf-body h3 { color: var(--accent, #d4a849); }
      .kf-flows .kf-body code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 2px; font-size: 11.5px; font-family: var(--font-mono, "JetBrains Mono", monospace); }
      .kf-flows .kf-body pre { background: rgba(255,255,255,0.03); padding: 10px; overflow-x: auto; }
      .kf-flows .kf-body pre code { background: transparent; padding: 0; }
      .kf-flows .kf-body table { border-collapse: collapse; }
      .kf-flows .kf-body th, .kf-flows .kf-body td { border: 1px solid var(--border-color, #333); padding: 4px 8px; font-size: 12px; }
      .kf-flows .kf-body .kf-mermaid { background: rgba(255,255,255,0.02); padding: 14px; overflow-x: auto; margin: 12px 0; }
      .kf-flows .kf-body .kf-mermaid svg { max-width: 100%; height: auto; }
      .kf-flows .kf-empty { color: var(--text-tertiary, #888); font-style: italic; font-size: 12px; padding: 20px; text-align: center; }
      .kf-flows .kf-disabled { padding: 28px; text-align: center; color: var(--text-tertiary, #888); }
      .kf-flows .kf-error { color: #c08040; font-size: 11px; margin-top: 8px; }
    `;
    document.head.appendChild(style);
  }

  function renderShell(container: HTMLElement): void {
    if (!isKarmolabDesktop()) {
      container.innerHTML = `<div class="kf-flows"><div class="kf-disabled">karmoddrine-flows 는 Tauri 데스크톱 앱 전용입니다.</div></div>`;
      return;
    }
    container.innerHTML = `
      <div class="kf-flows">
        <aside class="kf-side"><h3>흐름 문서</h3><div data-kf="side">로딩 중…</div></aside>
        <main>
          <div class="kf-meta" data-kf="meta"></div>
          <div class="kf-body" data-kf="body"><div class="kf-empty">왼쪽에서 흐름을 선택하세요.</div></div>
        </main>
      </div>
    `;
  }

  async function load(container: HTMLElement): Promise<void> {
    if (!isKarmolabDesktop()) return;
    const list = await fetchList();
    if (!list) {
      const side = container.querySelector('[data-kf="side"]') as HTMLElement | null;
      if (side) side.innerHTML = `<div class="kf-empty">로드 실패 (Tauri 환경 확인)</div>`;
      return;
    }
    renderSide(container, list);
  }

  function renderSide(container: HTMLElement, list: FlowList): void {
    const side = container.querySelector('[data-kf="side"]') as HTMLElement | null;
    if (!side) return;
    if (list.flows.length === 0) {
      side.innerHTML = `<div class="kf-empty">memo/flows/ 가 비어있어요.<br>첫 .md 추가하면 보입니다.</div>`;
      return;
    }
    const groups = new Map<string, FlowMeta[]>();
    for (const flow of list.flows) {
      const cat = flow.category || '기타';
      if (!groups.has(cat)) groups.set(cat, []);
      groups.get(cat)!.push(flow);
    }
    const html = [...groups.entries()].map(([cat, flows]) => `
      <div class="kf-cat-group">
        <div class="kf-cat-label">${esc(cat)}</div>
        <ul>${flows.map(f => `
          <li data-slug="${esc(f.slug)}">${esc(f.title)}</li>
        `).join('')}</ul>
      </div>
    `).join('');
    side.innerHTML = html;
    side.querySelectorAll('li[data-slug]').forEach(li => {
      li.addEventListener('click', () => {
        const slug = li.getAttribute('data-slug');
        if (!slug) return;
        side.querySelectorAll('li[data-slug]').forEach(other => other.classList.remove('kf-active'));
        li.classList.add('kf-active');
        currentSlug = slug;
        void renderDoc(container, slug);
      });
    });
    if (list.flows.length > 0 && currentSlug == null) {
      const first = side.querySelector('li[data-slug]') as HTMLElement | null;
      if (first) first.click();
    }
  }

  async function renderDoc(container: HTMLElement, slug: string): Promise<void> {
    const meta = container.querySelector('[data-kf="meta"]') as HTMLElement | null;
    const body = container.querySelector('[data-kf="body"]') as HTMLElement | null;
    if (!meta || !body) return;
    body.innerHTML = `<div class="kf-empty">로딩 중…</div>`;
    const doc = await fetchDoc(slug);
    if (!doc) {
      body.innerHTML = `<div class="kf-empty">로드 실패</div>`;
      return;
    }
    meta.innerHTML = `<strong>${esc(doc.meta.title)}</strong> · <code>${esc(doc.meta.filePath)}</code> · 수정: ${esc(fmtDate(doc.meta.lastModifiedUnix))}${doc.meta.summary ? ` · ${esc(doc.meta.summary)}` : ''}`;

    const debug = {
      buildMarker: 'v5',
      markedLoaded: false,
      bodyLen: doc.body.length,
      bodyFenceCount: (doc.body.match(/^```/gm) || []).length,
      bodyMermaidFences: (doc.body.match(/^```mermaid\b/gm) || []).length,
      placeholders: 0,
      bodyHead: doc.body.slice(0, 80).replace(/\n/g, '↵'),
      htmlLen: 0,
      htmlTail: '',
      mermaidLoaded: false,
      rendered: 0,
      errors: [] as string[],
    };
    try {
      let marked;
      try {
        marked = await loadMarked();
        debug.markedLoaded = true;
      } catch (e) {
        debug.errors.push(`marked load: ${(e as Error)?.message ?? String(e)}`);
        body.innerHTML = `<div class="kf-error">marked 로드 실패: ${esc((e as Error)?.message ?? String(e))}</div>`;
        prependDebug(body, debug);
        return;
      }
      // mermaid fence 를 marked 가 처리하지 못함 (어떤 marked 버전에선 `<pre><code>` 로 안 만듦)
      // → 직접 추출 후 placeholder 로 대체. 그 후 marked 로 나머지만 처리. 마지막에 placeholder 에 svg.
      const placeholders = new Map<string, string>();
      let phIdx = 0;
      const preprocessed = doc.body.replace(
        /^```mermaid[ \t]*\n([\s\S]*?)\n```\s*$/gm,
        (_match, def: string) => {
          const id = `__kf_mmd_${phIdx++}__`;
          placeholders.set(id, def);
          return `\n\n<div data-kf-mermaid="${id}"></div>\n\n`;
        }
      );
      debug.placeholders = placeholders.size;

      const html: string = marked.parse(preprocessed) as string;
      debug.htmlLen = typeof html === 'string' ? html.length : 0;
      debug.htmlTail = (typeof html === 'string' ? html : '').slice(-100).replace(/\n/g, '↵');
      body.innerHTML = html;

      if (placeholders.size > 0) {
        let mermaid;
        try {
          mermaid = await loadMermaid();
          debug.mermaidLoaded = true;
        } catch (e) {
          debug.errors.push(`mermaid load: ${(e as Error)?.message ?? String(e)}`);
        }
        if (mermaid) {
          const mmdDivs = Array.from(body.querySelectorAll('[data-kf-mermaid]'));
          let i = 0;
          for (const div of mmdDivs) {
            const id = div.getAttribute('data-kf-mermaid') || '';
            const def = placeholders.get(id);
            if (!def) {
              debug.errors.push(`placeholder ${id}: def 없음`);
              continue;
            }
            const renderId = `kf-mmd-${slug}-${i++}-${Date.now()}`;
            try {
              const { svg } = await mermaid.render(renderId, def);
              const wrap = document.createElement('div');
              wrap.className = 'kf-mermaid';
              wrap.innerHTML = svg;
              (div as HTMLElement).replaceWith(wrap);
              debug.rendered++;
            } catch (e) {
              debug.errors.push(`render ${i - 1}: ${(e as Error)?.message ?? String(e)}`);
              const errDiv = document.createElement('div');
              errDiv.className = 'kf-error';
              errDiv.textContent = `Mermaid 렌더 실패: ${(e as Error)?.message ?? String(e)}`;
              (div as HTMLElement).replaceWith(errDiv);
            }
          }
        }
      }
      prependDebug(body, debug);
    } catch (e) {
      body.innerHTML = `<div class="kf-error">렌더 실패: ${esc((e as Error)?.message ?? String(e))}</div>`;
      prependDebug(body, debug);
    }
  }

  function prependDebug(body: HTMLElement, d: { buildMarker: string; markedLoaded: boolean; bodyLen: number; bodyFenceCount: number; bodyMermaidFences: number; placeholders: number; bodyHead: string; htmlLen: number; htmlTail: string; mermaidLoaded: boolean; rendered: number; errors: string[] }): void {
    const div = document.createElement('div');
    div.style.cssText = 'background:#222;color:#9cf;padding:6px 8px;font-size:11px;font-family:monospace;margin-bottom:8px;border-left:2px solid #4af;line-height:1.4;white-space:pre-wrap;word-break:break-all;';
    const lines = [
      `[debug ${d.buildMarker}] marked:${d.markedLoaded ? 'OK' : 'X'} bodyLen:${d.bodyLen} bodyFences:${d.bodyFenceCount} mmdFences:${d.bodyMermaidFences} placeholders:${d.placeholders} htmlLen:${d.htmlLen} mermaid:${d.mermaidLoaded ? 'OK' : 'X'} rendered:${d.rendered}`,
      `bodyHead: ${d.bodyHead}`,
      `htmlTail: ${d.htmlTail}`,
    ];
    if (d.errors.length > 0) lines.push(`errors: ${d.errors.join(' | ')}`);
    div.textContent = lines.join('\n');
    body.prepend(div);
  }
})();
