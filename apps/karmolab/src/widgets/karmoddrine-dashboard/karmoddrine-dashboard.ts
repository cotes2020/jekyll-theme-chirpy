/**
 * karmoddrine-dashboard — Tauri 데스크톱 전용 (category: 'desktop' 자동 게이팅).
 *
 * karmoddrine umbrella 의 활성 세션 / 최근 commit / 도구 인벤토리 / 룰 단일 출처를
 * 카드 + 표 + Mermaid 그래프로 시각화. 데이터는 Rust 명령 `get_karmoddrine_state`
 * 가 ~/repos/karmoddrine/ 로컬 파일 + 3 레포 git log 에서 수집해 반환. 10초 폴링.
 *
 * github.io 공개 사이트엔 표시 X (category: 'desktop' + invoke 결과 없음 → placeholder).
 */
// @ts-nocheck — Toolbox/Mdd/window.__TAURI__ 글로벌은 ambient 타입에 다 안 잡혀 있음.
(function (): void {
  // ── 타입 (Rust struct camelCase) ─────────────────────────────────
  interface BoardRow { start: string; topic: string; targets: string; status: string; }
  interface BoardData { raw: string; rows: BoardRow[]; }
  interface CommitInfo { hash: string; date: string; subject: string; }
  interface RuleRow { category: string; canonical: string; cite: string; }
  interface ToolsData { commands: string[]; hooks: string[]; settingsHooks: Record<string, string>; }
  interface KarmoddrineState {
    generatedAtUnix: number;
    home: string | null;
    umbrella: string | null;
    board: BoardData | null;
    commits: Record<string, CommitInfo[]>;
    rules: RuleRow[];
    tools: ToolsData;
  }

  const POLL_INTERVAL_MS = 10_000;
  const REPOS = ['memo', 'Mascari4615.github.io', 'WitchMendokusai'];

  function isKarmolabDesktop(): boolean {
    return typeof window !== 'undefined' && !!window.__KARMOLAB_DESKTOP__;
  }

  async function fetchState(): Promise<KarmoddrineState | null> {
    const invoke = window.__TAURI__?.core?.invoke;
    if (typeof invoke !== 'function') return null;
    try {
      return await invoke('get_karmoddrine_state');
    } catch (e) {
      console.error('get_karmoddrine_state 실패', e);
      return null;
    }
  }

  // ── Mermaid CDN (한 번만 로드) ──────────────────────────────────
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
        window.dispatchEvent(new Event('kd-mermaid-loaded'));
      `;
      window.addEventListener('kd-mermaid-loaded', () => resolve((window as any).mermaid), { once: true });
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

  // ── 헬퍼 ─────────────────────────────────────────────────────────
  function esc(s: string): string {
    return Toolbox.escapeHtml ? Toolbox.escapeHtml(s) : s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function inline(s: string): string {
    return esc(s)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.+?)`/g, '<code>$1</code>');
  }
  function statusClass(status: string): string {
    if (/-?done$|done$/.test(status)) return 'kd-pill kd-pill--done';
    if (/committing|pending-?commit|진입|대기/.test(status)) return 'kd-pill kd-pill--active';
    if (/verify|pending/.test(status)) return 'kd-pill kd-pill--warn';
    return 'kd-pill kd-pill--other';
  }
  function escMermaid(s: string): string {
    return s.replace(/[<>"`]/g, '').replace(/\n/g, ' ').replace(/\|/g, '/');
  }
  function hash6(s: string): string {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    return Math.abs(h).toString(36).slice(0, 6);
  }

  // ── Toolbox 등록 ─────────────────────────────────────────────────
  Toolbox.register({
    ...Toolbox.getLazyWidgetPublicMeta('karmoddrine-dashboard'),
    tabs: [
      {
        id: 'app',
        label: 'karmoddrine',
        build(container: HTMLElement): void {
          injectStyles();
          renderShell(container);
          startPolling(container);
        },
      },
    ],
  });

  // ── 스타일 주입 ──────────────────────────────────────────────────
  function injectStyles(): void {
    if (document.getElementById('kd-dashboard-styles')) return;
    const style = document.createElement('style');
    style.id = 'kd-dashboard-styles';
    style.textContent = `
      .kd-dashboard { color: var(--text-primary, #e8e8e8); padding: 16px; max-width: 1500px; margin: 0 auto; }
      .kd-dashboard .kd-meta { color: var(--text-tertiary, #888); font-size: 11px; margin-bottom: 16px; }
      .kd-dashboard .kd-section { margin-bottom: 28px; }
      .kd-dashboard h2 { color: var(--accent, #d4a849); border-bottom: 1px solid var(--border-color, #333); padding-bottom: 4px; margin: 0 0 12px; font-size: 14px; font-weight: 500; }
      .kd-dashboard h3 { font-size: 12px; color: var(--accent, #d4a849); font-weight: 500; margin: 0 0 6px; }
      .kd-dashboard .kd-empty { color: var(--text-tertiary, #888); font-style: italic; font-size: 12px; }
      .kd-dashboard code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 2px; font-size: 11px; font-family: var(--font-mono, "JetBrains Mono", monospace); }

      .kd-pill { display: inline-block; padding: 2px 8px; border-radius: 2px; font-size: 10px; font-weight: 500; white-space: nowrap; }
      .kd-pill--done { background: #4a7c4a; color: #fff; }
      .kd-pill--active { background: #c08040; color: #fff; }
      .kd-pill--warn { background: #c0a040; color: #000; }
      .kd-pill--other { background: #555; color: #fff; }

      .kd-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }
      .kd-card { background: rgba(255,255,255,0.03); border-left: 2px solid var(--accent, #d4a849); padding: 10px 12px; }
      .kd-card .kd-card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 6px; }
      .kd-card .kd-card-start { font-size: 11px; color: var(--text-tertiary, #888); white-space: nowrap; }
      .kd-card .kd-card-topic { font-size: 12px; line-height: 1.45; }
      .kd-card .kd-card-targets { font-size: 11px; color: var(--text-tertiary, #aaa); margin-top: 6px; word-break: break-all; }

      .kd-3col { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; }
      .kd-commit-list { list-style: none; padding: 0; margin: 0; font-size: 11px; }
      .kd-commit-list li { padding: 3px 0; border-bottom: 1px dashed var(--border-color, #333); display: grid; grid-template-columns: 60px 78px 1fr; gap: 8px; align-items: baseline; }
      .kd-commit-list .kd-hash { color: var(--accent, #d4a849); font-family: var(--font-mono, monospace); }
      .kd-commit-list .kd-date { color: var(--text-tertiary, #888); font-family: var(--font-mono, monospace); }

      .kd-tool-list { margin: 0; padding-left: 16px; font-size: 11px; }
      .kd-tool-list li { margin: 2px 0; }

      .kd-table { border-collapse: collapse; width: 100%; font-size: 11.5px; }
      .kd-table th, .kd-table td { text-align: left; padding: 5px 8px; vertical-align: top; border-bottom: 1px solid var(--border-color, #333); }
      .kd-table th { color: var(--text-tertiary, #888); font-weight: 500; }
      .kd-table td.kd-cat { font-weight: 500; color: var(--accent, #d4a849); }

      .kd-graph { background: rgba(255,255,255,0.02); padding: 14px; min-height: 100px; overflow-x: auto; }
      .kd-graph svg { max-width: 100%; height: auto; }
      .kd-graph pre { font-size: 10.5px; color: #888; overflow: auto; }

      .kd-disabled { padding: 28px; text-align: center; color: var(--text-tertiary, #888); }
    `;
    document.head.appendChild(style);
  }

  // ── 셸 ──────────────────────────────────────────────────────────
  function renderShell(container: HTMLElement): void {
    if (!isKarmolabDesktop()) {
      container.innerHTML = `<div class="kd-disabled">karmoddrine dashboard 는 Tauri 데스크톱 앱 전용입니다.</div>`;
      return;
    }
    container.innerHTML = `
      <div class="kd-dashboard">
        <div class="kd-meta" data-kd="meta">로딩 중…</div>
        <section class="kd-section"><h2>활성 세션</h2><div data-kd="board"></div></section>
        <section class="kd-section"><h2>최근 commit (3 레포)</h2><div data-kd="commits" class="kd-3col"></div></section>
        <section class="kd-section"><h2>도구 인벤토리</h2><div data-kd="tools" class="kd-cards"></div></section>
        <section class="kd-section"><h2>룰 단일 출처</h2><div data-kd="rules"></div></section>
        <section class="kd-section"><h2>파일 소유권 그래프 — 충돌 = 두 세션이 같은 파일 잡음</h2><div data-kd="ownership" class="kd-graph"></div></section>
        <section class="kd-section"><h2>룰 단일 출처 네트워크</h2><div data-kd="rules-graph" class="kd-graph"></div></section>
      </div>
    `;
  }

  // ── 폴링 ────────────────────────────────────────────────────────
  let pollTimer: number | null = null;

  function startPolling(container: HTMLElement): void {
    if (!isKarmolabDesktop()) return;
    void refresh(container);
    if (pollTimer != null) window.clearInterval(pollTimer);
    pollTimer = window.setInterval(() => {
      if (!container.isConnected) {
        if (pollTimer != null) { window.clearInterval(pollTimer); pollTimer = null; }
        return;
      }
      void refresh(container);
    }, POLL_INTERVAL_MS);
  }

  async function refresh(container: HTMLElement): Promise<void> {
    const state = await fetchState();
    if (!state) {
      const meta = container.querySelector('[data-kd="meta"]') as HTMLElement | null;
      if (meta) meta.textContent = '데이터 가져오기 실패 (Tauri 앱 안에서 실행 중인지 확인)';
      return;
    }
    renderMeta(container, state);
    renderBoard(container, state);
    renderCommits(container, state);
    renderTools(container, state);
    renderRules(container, state);
    void renderOwnership(container, state);
    void renderRulesGraph(container, state);
  }

  // ── 섹션 렌더 ──────────────────────────────────────────────────
  function renderMeta(container: HTMLElement, state: KarmoddrineState): void {
    const meta = container.querySelector('[data-kd="meta"]') as HTMLElement | null;
    if (!meta) return;
    const generated = state.generatedAtUnix > 0 ? new Date(state.generatedAtUnix * 1000).toLocaleString('ko-KR') : '?';
    meta.innerHTML = `생성: ${esc(generated)} · umbrella: <code>${esc(state.umbrella ?? '?')}</code> · 폴링: ${POLL_INTERVAL_MS / 1000}s`;
  }

  function renderBoard(container: HTMLElement, state: KarmoddrineState): void {
    const root = container.querySelector('[data-kd="board"]') as HTMLElement | null;
    if (!root) return;
    if (!state.board || state.board.rows.length === 0) {
      root.innerHTML = `<p class="kd-empty">(보드 비어있음)</p>`;
      return;
    }
    root.innerHTML = `<div class="kd-cards">${state.board.rows.map(r => `
      <div class="kd-card">
        <div class="kd-card-head">
          <span class="kd-card-start">${esc(r.start)}</span>
          <span class="${statusClass(r.status)}">${esc(r.status)}</span>
        </div>
        <div class="kd-card-topic">${inline(r.topic)}</div>
        <div class="kd-card-targets">${inline(r.targets)}</div>
      </div>
    `).join('')}</div>`;
  }

  function renderCommits(container: HTMLElement, state: KarmoddrineState): void {
    const root = container.querySelector('[data-kd="commits"]') as HTMLElement | null;
    if (!root) return;
    root.innerHTML = REPOS.map(repo => {
      const list = state.commits[repo] ?? [];
      if (list.length === 0) return `<div><h3>${esc(repo)}</h3><p class="kd-empty">(없음)</p></div>`;
      return `<div><h3>${esc(repo)}</h3><ul class="kd-commit-list">${list.map(c => `
        <li><span class="kd-hash">${esc(c.hash)}</span><span class="kd-date">${esc(c.date)}</span><span>${inline(c.subject)}</span></li>
      `).join('')}</ul></div>`;
    }).join('');
  }

  function renderTools(container: HTMLElement, state: KarmoddrineState): void {
    const root = container.querySelector('[data-kd="tools"]') as HTMLElement | null;
    if (!root) return;
    const cmd = state.tools.commands;
    const hk = state.tools.hooks;
    const sh = state.tools.settingsHooks;
    root.innerHTML = `
      <div class="kd-card">
        <h3>~/.claude/commands/ (슬래시 커맨드)</h3>
        ${cmd.length === 0 ? '<p class="kd-empty">(없음)</p>' : `<ul class="kd-tool-list">${cmd.map(f => `<li><code>/${esc(f.replace(/\.md$/, ''))}</code></li>`).join('')}</ul>`}
      </div>
      <div class="kd-card">
        <h3>~/.claude/hooks/ (hook 스크립트)</h3>
        ${hk.length === 0 ? '<p class="kd-empty">(없음)</p>' : `<ul class="kd-tool-list">${hk.map(f => `<li><code>${esc(f)}</code></li>`).join('')}</ul>`}
      </div>
      <div class="kd-card">
        <h3>settings.json hooks 등록</h3>
        ${Object.keys(sh).length === 0 ? '<p class="kd-empty">(없음)</p>' : `<ul class="kd-tool-list">${Object.entries(sh).map(([k, v]) => `<li><strong>${esc(k)}</strong>: <code>${esc(v)}</code></li>`).join('')}</ul>`}
      </div>
    `;
  }

  function renderRules(container: HTMLElement, state: KarmoddrineState): void {
    const root = container.querySelector('[data-kd="rules"]') as HTMLElement | null;
    if (!root) return;
    if (state.rules.length === 0) {
      root.innerHTML = `<p class="kd-empty">(룰 없음)</p>`;
      return;
    }
    root.innerHTML = `<table class="kd-table">
      <thead><tr><th>카테고리</th><th>Canonical</th><th>Cite/포인터</th></tr></thead>
      <tbody>${state.rules.map(r => `
        <tr><td class="kd-cat">${inline(r.category)}</td><td>${inline(r.canonical)}</td><td>${inline(r.cite)}</td></tr>
      `).join('')}</tbody>
    </table>`;
  }

  // ── 그래프: 파일 소유권 ─────────────────────────────────────────
  async function renderOwnership(container: HTMLElement, state: KarmoddrineState): Promise<void> {
    const root = container.querySelector('[data-kd="ownership"]') as HTMLElement | null;
    if (!root) return;
    if (!state.board || state.board.rows.length === 0) {
      root.innerHTML = `<p class="kd-empty">(세션 없음)</p>`;
      return;
    }
    interface Edge { sess: string; file: string; }
    const edges: Edge[] = [];
    const sessFileMap = new Map<string, Set<string>>();
    state.board.rows.forEach((r, idx) => {
      const sessId = `S${idx}`;
      extractFiles(r.targets).forEach(f => {
        edges.push({ sess: sessId, file: f });
        if (!sessFileMap.has(f)) sessFileMap.set(f, new Set());
        sessFileMap.get(f)!.add(sessId);
      });
    });
    if (edges.length === 0) {
      root.innerHTML = `<p class="kd-empty">(타겟 파일 없음)</p>`;
      return;
    }
    const sessNodes = state.board.rows.map((r, idx) => {
      const label = (r.topic.split(/[—:.\n]/)[0] ?? r.topic).slice(0, 28).trim();
      const cls = /-?done$|done$/.test(r.status) ? ':::done'
        : /committing|pending|진입|대기/.test(r.status) ? ':::active'
        : /verify/.test(r.status) ? ':::warn' : ':::other';
      return `S${idx}["${escMermaid(label)}<br/>(${escMermaid(r.status)})"]${cls}`;
    });
    const fileIdMap = new Map<string, string>();
    sessFileMap.forEach((_, f) => { fileIdMap.set(f, `F${hash6(f)}`); });
    const fileNodes: string[] = [];
    sessFileMap.forEach((sessSet, f) => {
      const id = fileIdMap.get(f)!;
      const conflict = sessSet.size > 1;
      const safe = escMermaid(f.length > 38 ? f.slice(0, 36) + '…' : f);
      fileNodes.push(`${id}["${safe}"]${conflict ? ':::conflict' : ''}`);
    });
    const links = edges.map(e => `${e.sess} --- ${fileIdMap.get(e.file)}`);
    const code = [
      'graph LR',
      ...sessNodes,
      ...fileNodes,
      ...links,
      'classDef done fill:#4a7c4a,stroke:#5a8c5a,color:#fff',
      'classDef active fill:#c08040,stroke:#d09050,color:#fff',
      'classDef warn fill:#c0a040,stroke:#d0b050,color:#000',
      'classDef other fill:#555,stroke:#666,color:#fff',
      'classDef conflict fill:#c04040,stroke:#d05050,color:#fff,stroke-width:2px',
    ].join('\n');
    await renderMermaid(root, 'kd-ownership', code);
  }

  // ── 그래프: 룰 단일 출처 네트워크 ───────────────────────────────
  async function renderRulesGraph(container: HTMLElement, state: KarmoddrineState): Promise<void> {
    const root = container.querySelector('[data-kd="rules-graph"]') as HTMLElement | null;
    if (!root) return;
    if (state.rules.length === 0) {
      root.innerHTML = `<p class="kd-empty">(룰 없음)</p>`;
      return;
    }
    const nodes = new Set<string>();
    const edges: string[] = [];
    state.rules.forEach((r, idx) => {
      const canonicalKey = (r.canonical.match(/^([KSWM])\b/)?.[1]) ?? `R${idx}`;
      const ruleLabel = escMermaid(r.category.replace(/\*\*/g, '').slice(0, 26));
      nodes.add(`R${idx}["${ruleLabel}"]`);
      nodes.add(`${canonicalKey}((${escMermaid(canonicalKey)}))`);
      edges.push(`${canonicalKey} -->|정본| R${idx}`);
      const citeText = r.cite.replace(/\*\*/g, '').trim();
      if (citeText && !/cite 없음|repo-specific|S repo-specific/.test(citeText)) {
        const citeShort = escMermaid(citeText.slice(0, 30));
        const citeId = `C${idx}`;
        nodes.add(`${citeId}["${citeShort}"]`);
        edges.push(`R${idx} -.->|cite| ${citeId}`);
      }
    });
    const code = ['graph LR', ...Array.from(nodes), ...edges].join('\n');
    await renderMermaid(root, 'kd-rules-net', code);
  }

  // ── Mermaid 렌더 ────────────────────────────────────────────────
  let mermaidCounter = 0;
  async function renderMermaid(root: HTMLElement, idPrefix: string, code: string): Promise<void> {
    try {
      const m = await loadMermaid();
      mermaidCounter++;
      const id = `${idPrefix}-${mermaidCounter}`;
      const { svg } = await m.render(id, code);
      root.innerHTML = svg;
    } catch (e) {
      console.error('mermaid 렌더 실패', e);
      root.innerHTML = `<p class="kd-empty">그래프 렌더 실패: ${esc(String(e))}</p><pre>${esc(code)}</pre>`;
    }
  }

  // ── 백틱 안 파일 추출 ───────────────────────────────────────────
  function extractFiles(targets: string): string[] {
    const set = new Set<string>();
    const re = /`([^`]+)`/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(targets)) !== null) {
      const inner = m[1].trim();
      // 너무 긴 토큰(중괄호 묶음)도 그대로. 단 빈 거 / 이상한 건 제외.
      if (inner.length > 0 && inner.length < 200) set.add(inner);
    }
    return Array.from(set);
  }
})();
