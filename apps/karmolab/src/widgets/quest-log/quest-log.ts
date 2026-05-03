/**
 * Quest Log — Tauri 데스크톱 전용 (category: 'desktop' 자동 게이팅).
 *
 * memo TASK 파일 트리. Rust 명령 `get_quest_tree` 가 6 도메인 walk
 * (wm, karmolab, yawnbot, life, hobby, learning) + frontmatter 파싱
 * + 본문 체크박스 추출 → JSON 산출.
 *
 * v1 = read-only. 폴링 2s. status 6단계 색상 (seed/ready/active/hold/done/sealed).
 * leaf 체크박스 표시 (토글은 v2). Phase E (notify watcher) 로 폴링 → 즉시 푸시 리팩터 예정.
 *
 * github.io 공개 사이트엔 안 표시 (category: 'desktop' + invoke 실패 시 안내).
 */
// @ts-nocheck — Toolbox/Mdd/window.__TAURI__ 글로벌은 ambient 타입에 미정의.
(function (): void {
  // ── 타입 (Rust struct camelCase) ─────────────────────────────────
  interface CheckItem {
    text: string;
    done: boolean;
    group: string | null;
  }
  interface TaskNode {
    id: string;
    status: string;
    priority: string;
    path: string[];
    parent: string | null;
    tags: string[];
    title: string;
    filePath: string;
    checks: CheckItem[];
  }
  interface TaskError {
    filePath: string;
    reason: string;
  }
  interface QuestTree {
    tasks: TaskNode[];
    generatedAtUnix: number;
    memoPath: string;
    errors: TaskError[];
  }

  const POLL_INTERVAL_MS = 2000;
  const DOMAIN_ORDER = ['wm', 'karmolab', 'yawnbot', 'life', 'hobby', 'learning'];
  const DOMAIN_LABEL: Record<string, string> = {
    wm: 'WitchMendokusai',
    karmolab: 'KarmoLab',
    yawnbot: 'YawnBot',
    life: 'LIFE — 인생',
    hobby: 'HOBBY — 취미',
    learning: 'LEARN — 학습',
  };

  // 이전 hardcoded 위젯의 localStorage 정본 코드 자기 소멸. memo 가 단일 정본.
  try {
    localStorage.removeItem('quest-log-state-v1');
  } catch {
    // ignore
  }

  function isKarmolabDesktop(): boolean {
    return typeof window !== 'undefined' && !!window.__KARMOLAB_DESKTOP__;
  }

  async function fetchTree(): Promise<QuestTree | null> {
    const invoke = window.__TAURI__?.core?.invoke;
    if (typeof invoke !== 'function') return null;
    try {
      return (await invoke('get_quest_tree')) as QuestTree;
    } catch (err) {
      console.error('get_quest_tree 실패', err);
      return null;
    }
  }

  // ── 헬퍼 ─────────────────────────────────────────────────────────
  function esc(value: string): string {
    return Toolbox.escapeHtml
      ? Toolbox.escapeHtml(value)
      : value
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
  }
  function statusClass(status: string): string {
    return `ql-pill ql-pill--${status}`;
  }
  const STATUS_LABEL: Record<string, string> = {
    seed: '🌱 seed',
    ready: '⏳ ready',
    active: '🔥 active',
    hold: '💤 hold',
    done: '✓ done',
    sealed: '🔒 sealed',
  };
  function statusLabel(status: string): string {
    return STATUS_LABEL[status] ?? status;
  }

  // ── 폴링 상태 — Toolbox.register build() 가 동기 startPolling 호출 → TDZ 회피 위해 미리 선언
  let pollTimer: number | null = null;

  // ── Toolbox 등록 ─────────────────────────────────────────────────
  Toolbox.register({
    ...Toolbox.getLazyWidgetPublicMeta('quest-log'),
    tabs: [
      {
        id: 'app',
        label: 'Quest Log',
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
    if (document.getElementById('ql-styles')) return;
    const style = document.createElement('style');
    style.id = 'ql-styles';
    style.textContent = `
      .kl-quest-log { color: var(--text-primary, #e8e8e8); padding: 16px; max-width: 1200px; margin: 0 auto; }
      .kl-quest-log .ql-meta { color: var(--text-tertiary, #888); font-size: 11px; margin-bottom: 12px; }
      .kl-quest-log .ql-empty { color: var(--text-tertiary, #888); font-style: italic; font-size: 12px; }
      .kl-quest-log .ql-disabled { padding: 28px; text-align: center; color: var(--text-tertiary, #888); }
      .kl-quest-log code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 2px; font-size: 11px; font-family: var(--font-mono, "JetBrains Mono", monospace); }

      .kl-quest-log h2 { color: var(--accent, #d4a849); border-bottom: 1px solid var(--border-color, #333); padding-bottom: 4px; margin: 18px 0 10px; font-size: 13px; font-weight: 500; }

      .ql-domain { margin-bottom: 24px; }
      .ql-task-list { list-style: none; padding-left: 0; margin: 0; }
      .ql-task { padding: 6px 0 6px 12px; border-left: 2px solid var(--border-color, #333); margin-bottom: 4px; }
      .ql-task--child { margin-left: 18px; border-left-color: rgba(255,255,255,0.06); }
      .ql-task-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
      .ql-task-title { font-size: 12px; }
      .ql-task-id { font-size: 10px; color: var(--text-tertiary, #888); font-family: var(--font-mono, "JetBrains Mono", monospace); }
      .ql-task-meta { font-size: 10px; color: var(--text-tertiary, #888); margin-top: 2px; }
      .ql-task-progress { font-size: 10px; color: var(--text-tertiary, #888); }

      .ql-pill { display: inline-block; padding: 1px 6px; border-radius: 2px; font-size: 9.5px; font-weight: 500; white-space: nowrap; }
      .ql-pill--seed { background: #4a5560; color: #ddd; }
      .ql-pill--ready { background: #5577a0; color: #fff; }
      .ql-pill--active { background: #c07040; color: #fff; }
      .ql-pill--hold { background: #6a6a4a; color: #fff; }
      .ql-pill--done { background: #4a7c4a; color: #fff; }
      .ql-pill--sealed { background: #404040; color: #aaa; }

      .ql-checks { list-style: none; padding-left: 16px; margin: 4px 0; font-size: 11px; }
      .ql-check { padding: 1px 0; }
      .ql-check--done { color: var(--text-tertiary, #888); text-decoration: line-through; }
      .ql-check-group { font-size: 10px; color: var(--accent, #d4a849); margin: 4px 0 2px; font-style: italic; }

      .ql-errors { background: rgba(192, 64, 64, 0.08); border-left: 2px solid #c04040; padding: 8px 12px; margin: 12px 0; font-size: 11px; }
      .ql-errors h3 { color: #e08080; margin: 0 0 4px; font-size: 12px; }
      .ql-errors ul { margin: 0; padding-left: 16px; }

      .ql-tags { font-size: 10px; color: var(--text-tertiary, #888); display: inline-flex; gap: 3px; flex-wrap: wrap; }
      .ql-tag { background: rgba(255,255,255,0.05); padding: 1px 4px; border-radius: 2px; }
    `;
    document.head.appendChild(style);
  }

  // ── 셸 ──────────────────────────────────────────────────────────
  function renderShell(container: HTMLElement): void {
    if (!isKarmolabDesktop()) {
      container.innerHTML = `<div class="kl-quest-log"><div class="ql-disabled">Quest Log 는 KarmoLab 데스크톱 앱 (Tauri) 에서만 동작합니다.<br/>memo TASK 파일을 런타임에 읽어 트리로 표시합니다.</div></div>`;
      return;
    }
    container.innerHTML = `
      <div class="kl-quest-log">
        <div class="ql-meta" data-ql="meta">로딩 중…</div>
        <div data-ql="errors"></div>
        <div data-ql="tree"></div>
      </div>
    `;
  }

  // ── 폴링 ────────────────────────────────────────────────────────
  function startPolling(container: HTMLElement): void {
    if (!isKarmolabDesktop()) return;
    void refresh(container);
    if (pollTimer != null) window.clearInterval(pollTimer);
    pollTimer = window.setInterval(() => {
      if (!container.isConnected) {
        if (pollTimer != null) {
          window.clearInterval(pollTimer);
          pollTimer = null;
        }
        return;
      }
      void refresh(container);
    }, POLL_INTERVAL_MS);
  }

  async function refresh(container: HTMLElement): Promise<void> {
    const tree = await fetchTree();
    const meta = container.querySelector('[data-ql="meta"]') as HTMLElement | null;
    if (!tree) {
      if (meta) meta.textContent = '데이터 가져오기 실패 (Tauri 앱 안에서 실행 중인지 확인)';
      return;
    }
    renderMeta(container, tree);
    renderErrors(container, tree);
    renderTree(container, tree);
  }

  // ── 섹션 렌더 ──────────────────────────────────────────────────
  function renderMeta(container: HTMLElement, tree: QuestTree): void {
    const meta = container.querySelector('[data-ql="meta"]') as HTMLElement | null;
    if (!meta) return;
    const generated =
      tree.generatedAtUnix > 0
        ? new Date(tree.generatedAtUnix * 1000).toLocaleTimeString('ko-KR')
        : '?';
    meta.innerHTML = `생성: ${esc(generated)} · TASK: ${tree.tasks.length} · 폴링: ${POLL_INTERVAL_MS / 1000}s · memo: <code>${esc(tree.memoPath)}</code>`;
  }

  function renderErrors(container: HTMLElement, tree: QuestTree): void {
    const root = container.querySelector('[data-ql="errors"]') as HTMLElement | null;
    if (!root) return;
    if (tree.errors.length === 0) {
      root.innerHTML = '';
      return;
    }
    root.innerHTML = `<div class="ql-errors">
      <h3>파싱 실패 (${tree.errors.length})</h3>
      <ul>${tree.errors
        .map(
          (errItem) =>
            `<li><code>${esc(errItem.filePath)}</code> — ${esc(errItem.reason)}</li>`,
        )
        .join('')}</ul>
    </div>`;
  }

  function renderTree(container: HTMLElement, tree: QuestTree): void {
    const root = container.querySelector('[data-ql="tree"]') as HTMLElement | null;
    if (!root) return;
    if (tree.tasks.length === 0) {
      root.innerHTML = `<p class="ql-empty">(TASK 없음 — memo/{wm,projects/karmolab,...}/tasks/ 에 TASK-*.md 파일이 있는지 확인)</p>`;
      return;
    }

    // 도메인(path[0]) 별 그룹화. path 비어있으면 '(unknown)'.
    const byDomain = new Map<string, TaskNode[]>();
    for (const task of tree.tasks) {
      const domain = task.path[0] ?? '(unknown)';
      if (!byDomain.has(domain)) byDomain.set(domain, []);
      byDomain.get(domain)!.push(task);
    }

    // 도메인 순서 — 정해진 순서 + 미지정 도메인 후미
    const sortedDomains = [
      ...DOMAIN_ORDER.filter((domain) => byDomain.has(domain)),
      ...Array.from(byDomain.keys()).filter((domain) => !DOMAIN_ORDER.includes(domain)),
    ];

    root.innerHTML = sortedDomains
      .map((domain) => renderDomain(domain, byDomain.get(domain)!))
      .join('');
  }

  function renderDomain(domain: string, tasks: TaskNode[]): string {
    // parent chain — child 그룹화. parent 가 같은 도메인 안에 존재해야 nest.
    const idSet = new Set(tasks.map((task) => task.id));
    const childrenByParent = new Map<string, TaskNode[]>();
    const rootTasks: TaskNode[] = [];
    for (const task of tasks) {
      if (task.parent && idSet.has(task.parent)) {
        if (!childrenByParent.has(task.parent)) childrenByParent.set(task.parent, []);
        childrenByParent.get(task.parent)!.push(task);
      } else {
        rootTasks.push(task);
      }
    }
    rootTasks.sort((a, b) => a.id.localeCompare(b.id));
    childrenByParent.forEach((arr) => arr.sort((a, b) => a.id.localeCompare(b.id)));

    const label = DOMAIN_LABEL[domain] ?? domain;
    return `<section class="ql-domain">
      <h2>${esc(label)} (${tasks.length})</h2>
      <ul class="ql-task-list">${rootTasks
        .map((task) => `<li class="ql-task">${renderTaskNode(task, childrenByParent)}</li>`)
        .join('')}</ul>
    </section>`;
  }

  function renderTaskNode(
    task: TaskNode,
    childrenByParent: Map<string, TaskNode[]>,
  ): string {
    const children = childrenByParent.get(task.id) ?? [];
    const checkedCount = task.checks.filter((c) => c.done).length;
    const progress =
      task.checks.length > 0 ? `${checkedCount}/${task.checks.length}` : '';

    const tagsHtml =
      task.tags.length > 0
        ? `<span class="ql-tags">${task.tags
            .map((tag) => `<span class="ql-tag">${esc(tag)}</span>`)
            .join('')}</span>`
        : '';

    const checksHtml =
      task.checks.length > 0
        ? `<ul class="ql-checks">${renderChecks(task.checks)}</ul>`
        : '';

    const childrenHtml =
      children.length > 0
        ? `<ul class="ql-task-list">${children
            .map(
              (child) =>
                `<li class="ql-task ql-task--child">${renderTaskNode(child, childrenByParent)}</li>`,
            )
            .join('')}</ul>`
        : '';

    const priorityMeta = task.priority !== 'normal' ? `priority=${esc(task.priority)} · ` : '';
    const parentMeta = task.parent ? ` · parent=<code>${esc(task.parent)}</code>` : '';
    const pathMeta = `path=[${task.path.map((p) => esc(p)).join(', ')}]`;

    return `
      <div class="ql-task-head">
        <span class="${statusClass(task.status)}">${esc(statusLabel(task.status))}</span>
        <span class="ql-task-title">${esc(task.title)}</span>
        <span class="ql-task-id">${esc(task.id)}</span>
        ${progress ? `<span class="ql-task-progress">${esc(progress)}</span>` : ''}
      </div>
      <div class="ql-task-meta">
        ${priorityMeta}${pathMeta}${parentMeta}
        ${tagsHtml}
      </div>
      ${checksHtml}
      ${childrenHtml}
    `;
  }

  function renderChecks(checks: CheckItem[]): string {
    let html = '';
    let lastGroup: string | null | undefined;
    for (const check of checks) {
      if (check.group !== lastGroup) {
        if (check.group) {
          html += `<li class="ql-check-group">${esc(check.group)}</li>`;
        }
        lastGroup = check.group;
      }
      const mark = check.done ? '✓' : '○';
      const cls = check.done ? 'ql-check ql-check--done' : 'ql-check';
      html += `<li class="${cls}">${mark} ${esc(check.text)}</li>`;
    }
    return html;
  }
})();
