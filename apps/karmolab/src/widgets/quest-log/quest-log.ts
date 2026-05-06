/**
 * Quest Log — 관측실(observatory) 미감의 프로젝트·인생 항목 트리. Tauri 데스크톱 전용 (category: 'desktop').
 *
 * **데이터 = memo 정본** (TASK-KL-009 Phase F): hardcoded `QUEST_DATA` 폐기,
 * Rust 명령 `get_quest_tree` (apps/karmolab-tauri/src-tauri/src/quest_index.rs) 가
 * memo 의 6 도메인 walk (wm/projects/karmolab/projects/yawnbot/life/hobby/learning)
 * + frontmatter 파싱 + 본문 체크박스 추출 → JSON. 위젯이 invoke 후 옛 트리 구조
 * (projects[].children[].leaf{checks} + sealed[]) 로 변환.
 *
 * status 매핑: memo (seed/ready/active/hold/done/sealed) → 옛 (seed/fire/sleep/sealed):
 *   seed/ready → seed · active → fire · hold → sleep · done/sealed → sealed.
 *   status='sealed' 인 TASK 만 sealed[] 로 분리.
 *
 * 시각/인터랙션 (옛 standalone `apps/karmolab/quest-log/` — 폐기됨, 시각만 위젯에 흡수):
 * - 진행도: leaf = checked/total, 부모 = 자식 평균
 * - 영속화: localStorage `quest-log-state-v1` (위젯 내부 working state — 폴링/재진입 시 memo 정본으로 갱신)
 * - CSS·DOM `.kl-quest-log` 스코프. drawer/sleep prompt fixed 모달은 위젯 컨테이너 자식.
 *
 * v1 = read-only (memo 정본 우선). v2 (TASK-KL-010): 위젯 토글 → memo write back. v3: 인라인 에디터.
 */
// @ts-nocheck — port of inline IIFE; types narrow incrementally
(function (): void {
  // ── DATA — memo 정본 view ─────────────────────────────────────────────
  // hardcoded QUEST_DATA 폐기. Rust 명령 `get_quest_tree` 가 6 도메인 walk
  // → 위젯이 invoke 후 옛 트리 구조 (projects/children/leaf + sealed[]) 로 변환.

  interface MemoCheckItem {
    text: string;
    done: boolean;
    group: string | null;
    lineNumber: number;
  }
  interface MemoTaskNode {
    id: string;
    status: string;
    priority: string;
    path: string[];
    parent: string | null;
    tags: string[];
    title: string;
    filePath: string;
    checks: MemoCheckItem[];
  }
  interface MemoTaskError {
    filePath: string;
    reason: string;
  }
  interface MemoQuestTree {
    tasks: MemoTaskNode[];
    generatedAtUnix: number;
    memoPath: string;
    errors: MemoTaskError[];
  }

  const DOMAIN_ORDER = ['wm', 'karmolab', 'yawnbot', 'life', 'hobby', 'learning'];
  const DOMAIN_LABEL: Record<string, string> = {
    wm: 'WitchMendokusai',
    karmolab: 'KarmoLab',
    yawnbot: 'YawnBot',
    life: '인생',
    hobby: '취미',
    learning: '학습',
  };
  const DOMAIN_ICON: Record<string, string> = {
    wm: '🔮',
    karmolab: '🧪',
    yawnbot: '🤖',
    life: '🏠',
    hobby: '🎨',
    learning: '📚',
  };
  const DOMAIN_SUBTITLE: Record<string, string> = {
    wm: '메인 프로젝트 · 주황머리 마녀와 인형들',
    karmolab: 'Tauri 데스크톱 + 웹 위젯 + AI',
    yawnbot: 'Discord 봇 · 캐릭터 호스트',
    life: '인생 일반 — 건강·금융·집·관계',
    hobby: '취미 — 음악·독서·게임·여행',
    learning: '학습 — 책·강의·언어·기술',
  };

  // 이전 위젯 (a344ee85) 의 자기 소멸 코드는 옛 인터랙션 살리려 제거 (localStorage 다시 사용).

  function isKarmolabDesktop(): boolean {
    return typeof window !== 'undefined' && !!window.__KARMOLAB_DESKTOP__;
  }

  async function fetchMemoTree(): Promise<MemoQuestTree | null> {
    const invoke = window.__TAURI__?.core?.invoke;
    if (typeof invoke !== 'function') return null;
    try {
      return (await invoke('get_quest_tree')) as MemoQuestTree;
    } catch (err) {
      console.error('get_quest_tree 실패', err);
      return null;
    }
  }

  function mapMemoStatus(status: string): string {
    if (status === 'seed' || status === 'ready') return 'seed';
    if (status === 'active') return 'fire';
    if (status === 'hold') return 'sleep';
    if (status === 'done' || status === 'sealed') return 'sealed';
    return 'seed';
  }

  /// 위젯 → memo status 역방향 매핑 (KL-018 status write-back).
  /// `ready` 는 위젯 표현 불가 → 위젯 'seed' 클릭은 memo 'seed' 로 통일 (lossy).
  function mapWidgetStatusToMemo(widgetStatus: string): string {
    if (widgetStatus === 'fire') return 'active';
    if (widgetStatus === 'sleep') return 'hold';
    if (widgetStatus === 'sealed') return 'done';
    return 'seed';
  }

  function taskNodeToLeaf(t: MemoTaskNode): any {
    return {
      id: t.id,
      title: t.title,
      status: mapMemoStatus(t.status),
      memoStatus: t.status, // KL-018 — write-back 시 expected_status 로 사용
      memoPriority: t.priority, // KL-021 — priority write-back expected
      filePath: t.filePath,
      checks: t.checks.map((c) => ({ t: c.text, done: c.done, lineNumber: c.lineNumber })),
    };
  }

  /// memo TaskNode 들 → 옛 위젯 데이터 (projects/children/leaf checks + sealed[]).
  /// status='sealed' 인 TASK 만 sealed[] 로 분리. 그 외는 트리 안.
  /// 도메인(path[0]) 별 그룹 + parent chain 카테고리 (parent 가 자식 가지면 children 노드로).
  function transformMemoToOld(tree: MemoQuestTree): any {
    const sealedTasks: MemoTaskNode[] = [];
    const liveTasks: MemoTaskNode[] = [];
    for (const t of tree.tasks) {
      if (t.status === 'sealed') sealedTasks.push(t);
      else liveTasks.push(t);
    }

    const byDomain = new Map<string, MemoTaskNode[]>();
    for (const t of liveTasks) {
      const domain = t.path[0] ?? 'unknown';
      if (!byDomain.has(domain)) byDomain.set(domain, []);
      byDomain.get(domain)!.push(t);
    }

    const sortedDomains = [
      ...DOMAIN_ORDER.filter((d) => byDomain.has(d)),
      ...Array.from(byDomain.keys()).filter((d) => !DOMAIN_ORDER.includes(d)),
    ];

    const projects = sortedDomains.map((domain) => {
      const tasks = byDomain.get(domain)!;
      const idSet = new Set(tasks.map((t) => t.id));
      const childrenByParent = new Map<string, MemoTaskNode[]>();
      const rootTasks: MemoTaskNode[] = [];
      for (const t of tasks) {
        if (t.parent && idSet.has(t.parent)) {
          if (!childrenByParent.has(t.parent)) childrenByParent.set(t.parent, []);
          childrenByParent.get(t.parent)!.push(t);
        } else {
          rootTasks.push(t);
        }
      }
      rootTasks.sort((a, b) => a.id.localeCompare(b.id));
      childrenByParent.forEach((arr) => arr.sort((a, b) => a.id.localeCompare(b.id)));

      const children = rootTasks.map((t) => {
        const subs = childrenByParent.get(t.id);
        if (subs && subs.length > 0) {
          // parent 가 sub 가지면 카테고리 노드. 자기 자신도 leaf 로 children 의 첫 항목으로.
          const allLeaves = [taskNodeToLeaf(t), ...subs.map((s) => taskNodeToLeaf(s))];
          return {
            id: t.id,
            title: t.title,
            note: `${t.id} · sub-phase ${subs.length}개`,
            children: allLeaves,
          };
        }
        return taskNodeToLeaf(t);
      });

      return {
        id: domain,
        title: DOMAIN_LABEL[domain] ?? domain,
        subtitle: DOMAIN_SUBTITLE[domain] ?? '',
        kind: domain === 'wm' ? 'main' : 'side',
        icon: DOMAIN_ICON[domain] ?? '📦',
        children,
      };
    });

    const sealed = sealedTasks.map((t) => ({
      id: t.id,
      title: t.title,
      project: DOMAIN_LABEL[t.path[0] ?? ''] ?? t.path[0] ?? '',
      note: t.filePath,
      sealedNote: t.tags.join(', '),
    }));

    return { projects, sealed };
  }

  // ── STYLES (injected once) ──────────────────────────────────────────────
  const STYLE_ID = 'kl-quest-log-styles';
  const CSS = `
.kl-quest-log {
  --bg: #0b0d12;
  --bg-2: #0f1218;
  --paper: #12151c;
  --paper-2: #171a22;
  --ink: #f2f2ee;
  --ink-2: #9a9a94;
  --ink-3: #55555a;
  --ink-4: #33363d;
  --line: #1f242d;
  --line-2: #2a3040;
  --line-3: #3d4557;
  --accent: #d4a849;
  --accent-2: #7fa6d4;
  --mag-wm: #e8d9a8;
  --mag-project: #9ec4a8;
  --mag-learn: #b7a3d6;
  --mag-life: #d8a4a0;
  --mag-career: #7fa6d4;

  position: relative;
  color: var(--ink);
  font-family: 'Noto Sans KR', system-ui, sans-serif;
  /* 외부 body backdrop(observatory) 통과 — 자체 background/그리드는 KarmoLab 안에서 중복이므로 제거 */
  background: transparent;
  /* layout-full 안에서 화면 전체를 채우고 자체 스크롤 */
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  /* contain: paint 금지 — drawer/sleep-prompt overlay가 fixed인데 contain이 fixed positioning containment block을 만들어 viewport 추적이 깨짐 */
}
.kl-quest-log *, .kl-quest-log *::before, .kl-quest-log *::after { box-sizing: border-box; margin: 0; padding: 0; }
.kl-quest-log .serif { font-family: 'Noto Serif KR', serif; }
.kl-quest-log .mono { font-family: 'JetBrains Mono', monospace; }
.kl-quest-log ::selection { background: var(--accent); color: var(--bg); }

.kl-quest-log .wrap { max-width: none; margin: 0; padding: 24px 28px 48px; position: relative; z-index: 1; }

/* ── HEADER ── */
.kl-quest-log header.hd {
  padding-bottom: 14px; border-bottom: 1px solid var(--line-2); margin-bottom: 22px;
}
.kl-quest-log header.hd h1 {
  margin: 0; font-family: 'Noto Serif KR', serif; font-weight: 900;
  font-size: clamp(28px, 3.6vw, 44px); line-height: 1; letter-spacing: -0.02em;
}
.kl-quest-log header.hd h1 em { font-style: italic; font-weight: 500; color: var(--ink-2); }

/* ── STATS ── */
.kl-quest-log .stats {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px;
  background: var(--line-2); border: 1px solid var(--line-2);
  margin-bottom: 22px;
}
.kl-quest-log .stat { background: var(--paper); padding: 12px 16px; display: flex; flex-direction: column; gap: 3px; }
.kl-quest-log .stat-toggle {
  border: none; text-align: left; color: inherit; cursor: pointer; font: inherit;
  transition: background 140ms;
}
.kl-quest-log .stat-toggle:hover { background: var(--paper-2); }
.kl-quest-log .stat-toggle.on { background: var(--accent); color: var(--bg); }
.kl-quest-log .stat-toggle.on .k,
.kl-quest-log .stat-toggle.on .v small { color: var(--bg); opacity: 0.85; }
.kl-quest-log .stat .k {
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-3);
}
.kl-quest-log .stat .v {
  font-family: 'Noto Serif KR', serif; font-weight: 700;
  font-size: 22px; line-height: 1; letter-spacing: -0.01em;
}
.kl-quest-log .stat .v small { font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 13.5px; color: var(--ink-2); margin-left: 3px; }
.kl-quest-log .stat.accent .v { color: var(--accent); }

/* ── chip (drawer status switcher) ── */
.kl-quest-log .chip {
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-2);
  padding: 5px 10px; border: 1px solid var(--line-2); background: transparent;
  cursor: pointer; transition: all 140ms; display: inline-flex; align-items: center; gap: 6px;
}
.kl-quest-log .chip:hover { border-color: var(--ink-2); color: var(--ink); }
.kl-quest-log .chip.on { background: var(--ink); color: var(--bg); border-color: var(--ink); }

/* ── COLUMNS ── */
.kl-quest-log .columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
.kl-quest-log .col {
  border: 1px solid var(--line-2); background: var(--paper);
  display: flex; flex-direction: column; position: relative;
}
.kl-quest-log .col::before {
  content: ''; position: absolute; top: -1px; left: -1px; width: 10px; height: 10px;
  border-top: 1px solid var(--ink); border-left: 1px solid var(--ink);
}
.kl-quest-log .col::after {
  content: ''; position: absolute; bottom: -1px; right: -1px; width: 10px; height: 10px;
  border-bottom: 1px solid var(--line-3); border-right: 1px solid var(--line-3);
}
.kl-quest-log .col-head {
  padding: 14px 16px 12px; border-bottom: 1px solid var(--line-2);
  display: grid; grid-template-columns: 1fr auto; gap: 6px; align-items: baseline;
}
.kl-quest-log .col-head h3 {
  margin: 0; font-family: 'Noto Serif KR', serif; font-weight: 700;
  font-size: 20px; letter-spacing: -0.01em; display: flex; align-items: baseline; gap: 8px;
}
.kl-quest-log .col-head h3 .idx {
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  font-weight: 400; color: var(--ink-3); letter-spacing: 0.18em;
}
.kl-quest-log .col-head .sub {
  font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-2);
  text-align: right;
}
.kl-quest-log .col-head .bar-line {
  grid-column: 1 / -1; height: 2px; background: var(--line); margin-top: 6px;
  position: relative; overflow: hidden;
}
.kl-quest-log .col-head .bar-line .fill {
  position: absolute; inset: 0 auto 0 0; background: var(--accent);
  transition: width 400ms ease;
}
.kl-quest-log .col-head .bar-meta {
  grid-column: 1 / -1;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--ink-3);
  display: flex; justify-content: space-between; margin-top: 4px;
}
.kl-quest-log .col-head .bar-meta b { color: var(--ink-2); font-weight: 400; }

/* ── SKY PATCH ── */
.kl-quest-log .sky {
  aspect-ratio: 16/8; border-bottom: 1px solid var(--line-2);
  position: relative; overflow: hidden;
  background: radial-gradient(ellipse at 30% 60%, #182033 0%, #0a0d14 70%);
}
.kl-quest-log .sky.photo { background: #0a0d14; }
.kl-quest-log .sky.photo img {
  width: 100%; height: 100%; object-fit: cover;
}
.kl-quest-log .sky.photo::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(to bottom, rgba(11,13,18,0.25) 0%, rgba(11,13,18,0.8) 100%);
}
.kl-quest-log .sky svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.kl-quest-log .sky .coord {
  position: absolute; left: 12px; top: 12px; z-index: 3;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  color: var(--ink-2); letter-spacing: 0.15em; line-height: 1.7;
}
.kl-quest-log .sky .coord .k { color: var(--ink-3); }
.kl-quest-log .sky .tag {
  position: absolute; right: 12px; bottom: 12px; z-index: 3;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  color: var(--ink-2); letter-spacing: 0.18em; text-transform: uppercase;
  padding: 3px 7px; background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.08);
}

/* ── LOG LIST ── */
.kl-quest-log .log { padding: 8px 16px 12px; flex: 1; display: flex; flex-direction: column; }
.kl-quest-log .obs {
  display: grid; grid-template-columns: 60px 1fr auto;
  gap: 12px; align-items: baseline;
  padding: 12px 0 11px; border-bottom: 1px dashed var(--line-2);
  cursor: pointer;
  transition: background 120ms, padding 120ms;
  position: relative;
}
.kl-quest-log .obs:last-child { border-bottom: none; }
.kl-quest-log .obs::before {
  content: ''; position: absolute; left: -16px; top: 50%; transform: translateY(-50%);
  width: 3px; height: 0; background: var(--accent); transition: height 160ms;
}
.kl-quest-log .obs:hover { background: var(--bg-2); }
.kl-quest-log .obs:hover::before { height: 60%; }
.kl-quest-log .obs.selected { background: var(--bg-2); margin: 0 -16px; padding-left: 16px; padding-right: 16px; }
.kl-quest-log .obs.selected::before { height: 70%; left: 0; }

.kl-quest-log .obs .time {
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  color: var(--ink-3); letter-spacing: 0.08em; line-height: 1.4;
}
.kl-quest-log .obs .time b { color: var(--ink-2); font-weight: 500; display: block; }
.kl-quest-log .obs .body .lane {
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-3);
  margin-bottom: 3px; display: flex; align-items: center; gap: 6px;
}
.kl-quest-log .obs .body .lane .sw { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.kl-quest-log .obs .body .t {
  font-family: 'Noto Serif KR', serif; font-weight: 500;
  font-size: 17.5px; letter-spacing: -0.01em; line-height: 1.3;
}
.kl-quest-log .obs .body .n {
  margin-top: 4px; font-size: 14px; color: var(--ink-2); line-height: 1.55;
}
.kl-quest-log .obs .body .mini-bar {
  margin-top: 6px; height: 2px; background: var(--line); width: 70%;
  position: relative; overflow: hidden;
}
.kl-quest-log .obs .body .mini-bar .f {
  position: absolute; inset: 0 auto 0 0; background: var(--ink-2);
}
.kl-quest-log .obs[data-status="done"] .body .mini-bar .f { background: var(--ink); width: 100% !important; }
.kl-quest-log .obs[data-status="in-progress"] .body .mini-bar .f { background: var(--accent); }
.kl-quest-log .obs .mag {
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--ink-3);
  padding: 3px 7px; border: 1px solid var(--line-2);
  white-space: nowrap; align-self: start;
}
.kl-quest-log .obs[data-status="in-progress"] .time b { color: var(--ink); }
.kl-quest-log .obs[data-status="in-progress"] .body .t::before {
  content: '✦'; color: var(--accent); margin-right: 6px; font-size: 15px;
  display: inline-block;
}
.kl-quest-log .obs[data-status="done"] .body .t { color: var(--ink-2); text-decoration: line-through; text-decoration-color: var(--ink-3); text-decoration-thickness: 1px; }
.kl-quest-log .obs[data-status="done"] .body .n { color: var(--ink-3); }
.kl-quest-log .obs[data-status="in-progress"] .mag { color: var(--bg); background: var(--accent); border-color: var(--accent); }
.kl-quest-log .obs[data-status="done"] .mag { background: var(--ink); color: var(--bg); border-color: var(--ink); }

.kl-quest-log .empty {
  padding: 32px 0; text-align: center; color: var(--ink-3);
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  letter-spacing: 0.22em; text-transform: uppercase;
}
.kl-quest-log .empty::before, .kl-quest-log .empty::after { content: '— '; opacity: 0.6; }
.kl-quest-log .empty::after { content: ' —'; }

/* ── DRAWER ── */
.kl-quest-log .drawer {
  position: fixed; inset: 0 0 0 auto; width: min(520px, 92vw);
  background: var(--paper); border-left: 1px solid var(--line-3);
  transform: translateX(100%); transition: transform 280ms cubic-bezier(0.22, 0.9, 0.32, 1);
  z-index: 100; overflow-y: auto;
  box-shadow: -40px 0 80px rgba(0,0,0,0.4);
}
.kl-quest-log .drawer.open { transform: translateX(0); }
.kl-quest-log .drawer-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.7);
  z-index: 99; opacity: 0; pointer-events: none; transition: opacity 220ms;
}
.kl-quest-log .drawer-backdrop.open { opacity: 1; pointer-events: auto; }

.kl-quest-log .drawer-head {
  padding: 18px 24px; border-bottom: 1px solid var(--line-2);
  display: flex; justify-content: space-between; align-items: center;
  position: sticky; top: 0; background: var(--paper); z-index: 2;
}
.kl-quest-log .drawer-head .crumb {
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-3);
}
.kl-quest-log .drawer-head .crumb b { color: var(--ink); }
.kl-quest-log .drawer-close {
  background: transparent; border: 1px solid var(--line-2); color: var(--ink-2);
  font-family: 'JetBrains Mono', monospace; font-size: 14px;
  width: 28px; height: 28px; cursor: pointer; display: flex; align-items: center; justify-content: center;
}
.kl-quest-log .drawer-close:hover { border-color: var(--ink); color: var(--ink); }

.kl-quest-log .drawer-body { padding: 28px 28px 40px; }
.kl-quest-log .drawer-body .lane-pill {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-2);
  padding: 5px 10px; border: 1px solid var(--line-2);
}
.kl-quest-log .drawer-body .lane-pill .sw { width: 6px; height: 6px; border-radius: 50%; }
.kl-quest-log .drawer-body h2 {
  margin: 16px 0 10px; font-family: 'Noto Serif KR', serif; font-weight: 700;
  font-size: 32px; line-height: 1.1; letter-spacing: -0.02em;
}
.kl-quest-log .drawer-body h2 em { font-style: italic; font-weight: 400; color: var(--ink-2); }
.kl-quest-log .drawer-body .lede {
  font-size: 16.5px; color: var(--ink-2); line-height: 1.65; max-width: 52ch;
}
.kl-quest-log .drawer-body .progress-wrap {
  margin-top: 24px; padding-top: 18px; border-top: 1px solid var(--line-2);
}
.kl-quest-log .drawer-body .progress-wrap .lbl {
  display: flex; justify-content: space-between; align-items: baseline;
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-3);
  margin-bottom: 8px;
}
.kl-quest-log .drawer-body .progress-wrap .lbl b {
  font-family: 'Noto Serif KR', serif; font-size: 20px; color: var(--ink); letter-spacing: -0.01em;
  font-weight: 700;
}
.kl-quest-log .drawer-body .progress-wrap .bar {
  height: 3px; background: var(--line); position: relative; overflow: hidden;
}
.kl-quest-log .drawer-body .progress-wrap .bar .f {
  position: absolute; inset: 0 auto 0 0; background: var(--accent);
}
.kl-quest-log .drawer-body .progress-wrap .ticks {
  display: flex; justify-content: space-between; margin-top: 4px;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  color: var(--ink-3); letter-spacing: 0.18em;
}

/* ── FEATURED + SUB-GRID ── */
.kl-quest-log .featured {
  border: 1px solid var(--line-2); background: var(--paper);
  margin-bottom: 20px; position: relative;
  display: grid; grid-template-columns: 1.1fr 1.5fr;
}
.kl-quest-log .featured::before {
  content: ''; position: absolute; top: -1px; left: -1px; width: 14px; height: 14px;
  border-top: 1px solid var(--accent); border-left: 1px solid var(--accent);
}
.kl-quest-log .featured::after {
  content: ''; position: absolute; bottom: -1px; right: -1px; width: 14px; height: 14px;
  border-bottom: 1px solid var(--line-3); border-right: 1px solid var(--line-3);
}
.kl-quest-log .featured .f-left { display: flex; flex-direction: column; border-right: 1px solid var(--line-2); }
.kl-quest-log .featured .f-sky { aspect-ratio: auto; flex: 1; min-height: 280px; border-bottom: 1px solid var(--line-2); position: relative; overflow: hidden; }
.kl-quest-log .featured .f-sky img {
  width: 100%; height: 100%; object-fit: cover;
}
.kl-quest-log .featured .f-sky::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(to bottom, rgba(11,13,18,0.2) 0%, rgba(11,13,18,0.8) 100%);
}
.kl-quest-log .featured .f-sky .coord {
  position: absolute; left: 14px; top: 14px; z-index: 3;
  font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
  color: var(--ink-2); letter-spacing: 0.15em; line-height: 1.8;
}
.kl-quest-log .featured .f-sky .coord .k { color: var(--ink-3); }
.kl-quest-log .featured .f-sky .tag {
  position: absolute; right: 14px; bottom: 14px; z-index: 3;
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  color: var(--ink); letter-spacing: 0.2em; text-transform: uppercase;
  padding: 4px 9px; background: rgba(0,0,0,0.55); border: 1px solid var(--line-3);
}
.kl-quest-log .featured .f-sky .overlay-title {
  position: absolute; left: 20px; right: 20px; bottom: 42px; z-index: 3;
  pointer-events: none;
}
.kl-quest-log .featured .f-sky .overlay-title .cst {
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  letter-spacing: 0.3em; color: var(--accent); text-transform: uppercase;
}
.kl-quest-log .featured .f-sky .overlay-title .name {
  font-family: 'Noto Serif KR', serif; font-style: italic; font-weight: 500;
  font-size: 30px; color: var(--ink); line-height: 1.05; margin-top: 4px;
  letter-spacing: -0.01em; text-shadow: 0 2px 20px rgba(0,0,0,0.8);
}
.kl-quest-log .featured .f-meta { padding: 16px 20px; display: flex; flex-direction: column; gap: 10px; }
.kl-quest-log .featured .f-meta .eye {
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  letter-spacing: 0.28em; color: var(--accent); text-transform: uppercase;
}
.kl-quest-log .featured .f-meta h2 {
  margin: 0; font-family: 'Noto Serif KR', serif; font-weight: 900;
  font-size: 40px; line-height: 0.98; letter-spacing: -0.02em;
}
.kl-quest-log .featured .f-meta h2 em { font-style: italic; font-weight: 500; color: var(--ink-2); display: block; font-size: 0.5em; margin-top: 6px; letter-spacing: 0.05em; }
.kl-quest-log .featured .f-meta .dek {
  font-size: 15px; color: var(--ink-2); line-height: 1.6; max-width: 46ch;
  border-top: 1px dashed var(--line-2); padding-top: 10px;
}
.kl-quest-log .featured .f-meta .mini-stats {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
  background: var(--line-2); border: 1px solid var(--line-2); margin-top: auto;
}
.kl-quest-log .featured .f-meta .mini-stats .s { background: var(--paper); padding: 10px 12px; }
.kl-quest-log .featured .f-meta .mini-stats .s .k {
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-3);
}
.kl-quest-log .featured .f-meta .mini-stats .s .v {
  font-family: 'Noto Serif KR', serif; font-weight: 700;
  font-size: 20px; line-height: 1.2; letter-spacing: -0.01em;
}
.kl-quest-log .featured .f-meta .mini-stats .s.accent .v { color: var(--accent); }
.kl-quest-log .featured .f-meta .bar-line {
  height: 2px; background: var(--line); position: relative; overflow: hidden; margin-top: 4px;
}
.kl-quest-log .featured .f-meta .bar-line .fill {
  position: absolute; inset: 0 auto 0 0; background: var(--accent);
}

.kl-quest-log .featured .f-right {
  padding: 16px 20px 18px;
  display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px;
  align-content: start;
}
.kl-quest-log .featured .f-right .log-head {
  grid-column: 1 / -1;
  display: flex; justify-content: space-between; align-items: baseline;
  padding-bottom: 8px; border-bottom: 1px solid var(--line-2); margin-bottom: 4px;
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink-3);
}
.kl-quest-log .featured .f-right .log-head b { color: var(--ink); font-weight: 500; }

.kl-quest-log .sub-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
@media (max-width: 1100px) { .kl-quest-log .sub-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 640px) {
  .kl-quest-log .sub-grid { grid-template-columns: 1fr; }
  .kl-quest-log .featured { grid-template-columns: 1fr; }
  .kl-quest-log .featured .f-left { border-right: none; border-bottom: 1px solid var(--line-2); }
  .kl-quest-log .featured .f-right { grid-template-columns: 1fr; }
}

/* ── status-coloured obs rows ── */
.kl-quest-log .obs[data-status="fire"] .body .t::before {
  content: '◉'; color: var(--accent); margin-right: 6px; font-size: 15px;
  display: inline-block;
}
.kl-quest-log .obs[data-status="fire"] .mag { color: var(--bg); background: var(--accent); border-color: var(--accent); }
.kl-quest-log .obs[data-status="sleep"] .body .t { color: var(--ink-2); }
.kl-quest-log .obs[data-status="sleep"] .mag { color: var(--ink-3); border-style: dashed; }
.kl-quest-log .obs[data-status="seed"] .mag { color: var(--ink-3); }
.kl-quest-log .obs[data-status="sealed"] .mag { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.kl-quest-log .obs[data-status="sealed"] .body .t { color: var(--ink-2); }
.kl-quest-log .obs[data-status="sealed"] .body .t::before {
  content: '◆'; color: var(--accent); margin-right: 6px; font-size: 14px;
}

/* ── 5-star rating ── */
.kl-quest-log .stars { display: inline-flex; gap: 2px; vertical-align: middle; }
.kl-quest-log .stars .star { width: 11px; height: 11px; color: var(--line-3); display: inline-block; }
.kl-quest-log .stars.large .star { width: 16px; height: 16px; }
.kl-quest-log .stars .star.filled { color: var(--accent); }
.kl-quest-log .stars .star.half {
  color: var(--accent);
  mask-image: linear-gradient(90deg, black 50%, transparent 50%);
  -webkit-mask-image: linear-gradient(90deg, black 50%, transparent 50%);
}

/* ── checklist in drawer ── */
.kl-quest-log .checklist { display: flex; flex-direction: column; gap: 2px; }
.kl-quest-log .check-row {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 10px 10px 10px 4px; cursor: pointer;
  border-bottom: 1px dashed var(--line-2);
  transition: background 120ms;
}
.kl-quest-log .check-row:hover { background: var(--bg-2); }
.kl-quest-log .check-box {
  width: 14px; height: 14px; border: 1px solid var(--ink-3); flex-shrink: 0;
  margin-top: 3px; position: relative; transition: all 140ms;
}
.kl-quest-log .check-row.done .check-box { background: var(--accent); border-color: var(--accent); }
.kl-quest-log .check-row.done .check-box::after {
  content: '✓'; position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  color: var(--bg); font-size: 13px; font-weight: 700;
}
.kl-quest-log .check-label {
  font-family: 'Noto Serif KR', serif; font-size: 16px; line-height: 1.45;
  color: var(--ink); letter-spacing: -0.005em;
}
.kl-quest-log .check-row.done .check-label { color: var(--ink-3); text-decoration: line-through; text-decoration-color: var(--line-3); }
.kl-quest-log .check-row { position: relative; padding-right: 28px; }
.kl-quest-log .check-delete {
  position: absolute; right: 4px; top: 50%; transform: translateY(-50%);
  background: none; border: none; cursor: pointer;
  color: var(--ink-3); font-size: 18px; line-height: 1; padding: 4px 8px;
  opacity: 0; transition: opacity 0.12s, color 0.12s;
}
.kl-quest-log .check-row:hover .check-delete { opacity: 1; }
.kl-quest-log .check-delete:hover { color: #d4504e; }

.kl-quest-log .add-check input {
  flex: 1; background: var(--paper); border: none; outline: none;
  padding: 9px 12px; font-family: 'Noto Sans KR', sans-serif; font-size: 15px; color: var(--ink);
}
.kl-quest-log .add-check input::placeholder { color: var(--ink-3); }
.kl-quest-log .add-check button {
  background: var(--bg); border: none; color: var(--ink);
  padding: 9px 14px; font-family: 'JetBrains Mono', monospace; font-size: 13px;
  letter-spacing: 0.22em; cursor: pointer;
}
.kl-quest-log .add-check button:hover { background: var(--accent); color: var(--bg); }

/* ── status switcher ── */
.kl-quest-log .status-toggle {
  font-family: 'JetBrains Mono', monospace; font-size: 13px; letter-spacing: 0.2em;
  background: var(--paper); color: var(--ink-2); cursor: pointer;
  transition: all 140ms;
}
.kl-quest-log .status-toggle:hover { color: var(--ink); }
.kl-quest-log .status-toggle.on { background: var(--accent); color: var(--bg); }

/* ── child row ── */
.kl-quest-log .children-list { display: flex; flex-direction: column; gap: 0; }
.kl-quest-log .child-row {
  display: grid; grid-template-columns: 70px 1fr auto; gap: 12px; align-items: center;
  padding: 11px 4px; border-bottom: 1px dashed var(--line-2); cursor: pointer;
  transition: background 120ms, padding 120ms;
}
.kl-quest-log .child-row:hover { background: var(--bg-2); }
.kl-quest-log .cr-status {
  font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.22em;
  color: var(--ink-3); text-align: center; padding: 2px 5px; border: 1px solid var(--line-2);
}
.kl-quest-log .cr-status.fire { color: var(--accent); border-color: var(--accent); }
.kl-quest-log .cr-status.sealed { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.kl-quest-log .cr-title { font-family: 'Noto Serif KR', serif; font-size: 16.5px; color: var(--ink); }
.kl-quest-log .cr-right { display: flex; align-items: center; }

/* ── seal button ── */
.kl-quest-log .seal-btn {
  margin-top: 22px; width: 100%; background: var(--ink); color: var(--bg);
  border: none; padding: 14px; cursor: pointer;
  font-family: 'JetBrains Mono', monospace; font-size: 13.5px;
  letter-spacing: 0.3em; text-transform: uppercase; transition: background 140ms;
}
.kl-quest-log .seal-btn:hover { background: var(--accent); }

@media (max-width: 1000px) { .kl-quest-log .stats { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 640px) {
  .kl-quest-log .columns { grid-template-columns: 1fr; }
  .kl-quest-log .stats { grid-template-columns: repeat(2, 1fr); }
  .kl-quest-log header.hd { flex-direction: column; align-items: flex-start; gap: 8px; }
}
`;

  function injectStyles(): void {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = CSS;
    document.head.appendChild(style);

    // Observatory 폰트 — 페이지 어디든 같은 url을 여러 번 import해도 브라우저가 중복 다운로드 안 함
    if (!document.getElementById('kl-quest-log-fonts')) {
      const link = document.createElement('link');
      link.id = 'kl-quest-log-fonts';
      link.rel = 'stylesheet';
      link.href = 'https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;500;700;900&family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap';
      document.head.appendChild(link);
    }
  }

  // ── Toolbox.register ───────────────────────────────────────────────────
  Toolbox.register({
    ...Toolbox.getLazyWidgetPublicMeta('quest-log'),
    tabs: [
      {
        id: 'app',
        label: 'Quest Log',
        build: function (container: HTMLElement): void {
          Mdd.linePreset('tool_run', { msg: '관측 시작이에요. 별 잘 보이나요?' });
          injectStyles();
          renderQuestLog(container);
        }
      }
    ]
  });

  // ── renderQuestLog: HTML scaffold + memo 정본 fetch + runQuestLog ───────
  function renderQuestLog(container: HTMLElement): void {
    if (!isKarmolabDesktop()) {
      container.innerHTML = `<div class="kl-quest-log"><div style="padding:48px 24px; text-align:center; color:#888;">Quest Log 는 KarmoLab 데스크톱 앱 (Tauri) 에서만 동작합니다.<br/>memo TASK 파일을 런타임에 읽어 트리로 표시합니다.</div></div>`;
      return;
    }
    container.innerHTML = `
      <div class="kl-quest-log">
        <div class="wrap">
          <header class="hd">
            <h1 class="serif">QUEST LOG <em>— in progress</em></h1>
          </header>

          <div class="stats" data-kl-ql="stats"></div>

          <div data-kl-ql="featured-wrap"></div>
          <div class="sub-grid" data-kl-ql="sub-columns"></div>

        </div>

        <div class="drawer-backdrop" data-kl-ql="backdrop"></div>
        <aside class="drawer" data-kl-ql="drawer">
          <div class="drawer-head">
            <div class="crumb" data-kl-ql="crumb">KMLB-QST / <b>—</b></div>
            <button class="drawer-close" data-kl-ql="drawer-close" aria-label="Close">✕</button>
          </div>
          <div class="drawer-body" data-kl-ql="drawer-body"></div>
        </aside>
      </div>
    `;

    const root = container.querySelector('.kl-quest-log') as HTMLElement;

    // 비동기 invoke + 변환 + run
    void (async () => {
      const tree = await fetchMemoTree();
      if (!tree) {
        root.innerHTML = `<div style="padding:48px 24px; text-align:center; color:#c08080;">데이터 로딩 실패. F12 콘솔에 get_quest_tree 에러 확인.</div>`;
        return;
      }
      const src = transformMemoToOld(tree);
      runQuestLog(root, src);
    })();
  }

  // ── runQuestLog: 원본 IIFE 로직 (document → root, ID → data-kl-ql) ──────
  function runQuestLog(root: HTMLElement, src: any): void {
    const STORAGE_KEY = 'quest-log-state-v1';
    const SRC = src;

    const $ = (sel: string): HTMLElement | null => root.querySelector(sel) as HTMLElement | null;
    const $$ = (sel: string): NodeListOf<HTMLElement> => root.querySelectorAll(sel) as NodeListOf<HTMLElement>;
    const byKey = (key: string): HTMLElement | null => root.querySelector(`[data-kl-ql="${key}"]`) as HTMLElement | null;

    function loadStored() {
      try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch (e) { return null; }
    }
    function save() {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ projects: DATA.projects, sealed: DATA.sealed })); } catch (e) {}
    }
    const stored = loadStored();
    const DATA = {
      projects: stored?.projects ?? SRC.projects,
      sealed: stored?.sealed ?? SRC.sealed,
    };

    const HEROES = [
      '/apps/karmolab/img/widgets/quest-log/240126-072633.png',
      '/apps/karmolab/img/widgets/quest-log/240714-071225.jpg',
      '/apps/karmolab/img/widgets/quest-log/240330-000000.png',
      '/apps/karmolab/img/widgets/quest-log/240330-111546.png',
      '/apps/karmolab/img/widgets/quest-log/240330-140142.png',
      '/apps/karmolab/img/widgets/quest-log/240513-131941.png',
      '/apps/karmolab/img/widgets/quest-log/240514-103335.png',
      '/apps/karmolab/img/widgets/quest-log/240514-104350.png',
      '/apps/karmolab/img/widgets/quest-log/240514-192005.png',
      '/apps/karmolab/img/widgets/quest-log/240605-133617.png',
      '/apps/karmolab/img/widgets/quest-log/240618-000000.png',
      '/apps/karmolab/img/widgets/quest-log/250315-170647.png',
      '/apps/karmolab/img/widgets/quest-log/250315-173653.png',
    ];
    const CONST_BY_PROJECT: Record<string, { name: string; sub: string; mag: string }> = {
      wm:     { name: 'Venefica',  sub: 'the witch',    mag: '1.2' },
      blog:   { name: 'Scriba',    sub: 'the scribe',   mag: '2.8' },
      learn:  { name: 'Discipulus',sub: 'the student',  mag: '3.1' },
      travel: { name: 'Viator',    sub: 'the wanderer', mag: '2.5' },
      body:   { name: 'Corpus',    sub: 'the body',     mag: '3.4' },
    };

    function findNode(id: string, nodes: any[] = DATA.projects, parents: any[] = []): { node: any; parents: any[] } | null {
      for (const n of nodes) {
        if (n.id === id) return { node: n, parents };
        if (n.children) {
          const f = findNode(id, n.children, [...parents, n]);
          if (f) return f;
        }
      }
      return null;
    }
    function isLeaf(n: any): boolean { return Array.isArray(n.checks); }
    function allLeaves(n: any, out: any[] = []): any[] {
      if (isLeaf(n)) { out.push(n); return out; }
      if (n.children) n.children.forEach((c: any) => allLeaves(c, out));
      return out;
    }
    function progressOf(n: any): number {
      if (isLeaf(n)) return n.checks.length ? n.checks.filter((c: any) => c.done).length / n.checks.length : 0;
      if (!n.children || !n.children.length) return 0;
      return n.children.reduce((s: number, c: any) => s + progressOf(c), 0) / n.children.length;
    }
    function findAreaOf(id: string) {
      const f = findNode(id);
      if (!f) return null;
      return f.parents[1] || f.parents[0];
    }

    function hash(s: string): number { let h = 0; for (let i = 0; i < s.length; i++) { h = ((h << 5) - h) + s.charCodeAt(i); h |= 0; } return Math.abs(h); }
    function coords(idx: number) {
      const ra = (idx * 3.7 + 1) % 24;
      const dec = ((idx * 11.2) % 60) + 12;
      const rah = Math.floor(ra);
      const ram = Math.floor((ra - rah) * 60);
      const decd = Math.floor(dec);
      const decm = Math.floor((dec - decd) * 60);
      return { rah: String(rah).padStart(2, '0'), ram: String(ram).padStart(2, '0'),
               decd: String(decd).padStart(2, '0'), decm: String(decm).padStart(2, '0') };
    }

    const state = {
      view: 'log' as 'log' | 'trophy',
      selectedId: null as string | null,
    };

    function esc(s: any): string { return String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]!)); }

    function starsHTML(progress: number, large = false): string {
      const filled = progress * 5;
      const full = Math.floor(filled);
      const hasHalf = (filled - full) >= 0.25 && (filled - full) < 0.75;
      const svg = '<svg viewBox="0 0 24 24" width="100%" height="100%"><path d="M12 2l2.9 6.95 7.6.6-5.75 4.95L18.4 22 12 17.9 5.6 22l1.65-7.5L1.5 9.55l7.6-.6z" fill="currentColor"/></svg>';
      let html = '<span class="stars' + (large ? ' large' : '') + '">';
      for (let i = 0; i < 5; i++) {
        let cls = '';
        if (i < full) cls = 'filled';
        else if (i === full && hasHalf) cls = 'half';
        html += '<span class="star ' + cls + '">' + svg + '</span>';
      }
      html += '</span>';
      return html;
    }

    function renderStats() {
      const all: any[] = [];
      DATA.projects.forEach((p: any) => allLeaves(p).forEach(l => all.push(l)));
      const sealed = DATA.sealed.length;
      const coverage = all.length ? Math.round(all.reduce((s, l) => s + progressOf(l), 0) / all.length * 100) : 0;
      const el = byKey('stats');
      if (!el) return;
      const trophyOn = state.view === 'trophy';
      el.innerHTML = `
        <div class="stat"><div class="k">COVERAGE</div><div class="v">${coverage}<small>%</small></div></div>
        <button class="stat stat-toggle ${trophyOn ? 'on' : ''}" data-kl-ql="trophy-toggle" type="button">
          <div class="k">SEALED${trophyOn ? ' · OPEN' : ''}</div>
          <div class="v">${sealed}<small>${trophyOn ? '← back to log' : 'open trophy'}</small></div>
        </button>
      `;
      const toggle = byKey('trophy-toggle');
      if (toggle) {
        toggle.addEventListener('click', () => {
          state.view = state.view === 'trophy' ? 'log' : 'trophy';
          renderStats();
          renderColumns();
        });
      }
    }

    function skyHTML(idx: number, photo: boolean) {
      const { rah, ram, decd, decm } = coords(idx);
      if (photo) {
        const img = HEROES[idx % HEROES.length];
        return `
          <div class="sky photo">
            <img src="${img}" alt="">
            <div class="coord"><span class="k">RA</span> ${rah}<sup>h</sup> ${ram}<sup>m</sup><br><span class="k">DEC</span> +${decd}° ${decm}'</div>
            <div class="tag">PLATE 00${idx + 1}</div>
          </div>
        `;
      }
      const stars: any[] = [];
      for (let i = 0; i < 14; i++) {
        const h = hash('s' + idx + i);
        stars.push({ x: h % 100, y: ((h >> 8) % 80) + 10, r: ((h >> 16) % 15) / 10 + 0.4, bright: i < 5 });
      }
      const bright = stars.filter(s => s.bright);
      let lines = '';
      for (let i = 0; i < bright.length - 1; i++) {
        lines += `<line x1="${bright[i].x}" y1="${bright[i].y}" x2="${bright[i + 1].x}" y2="${bright[i + 1].y}" stroke="var(--line-3)" stroke-width="0.18" stroke-dasharray="0.6 0.8" opacity="0.7" />`;
      }
      const hasMoon = idx === 2 || idx === 5;
      const moonX = 65 + (idx * 7) % 20;
      const moonY = 28 + (idx * 3) % 15;
      return `
        <div class="sky">
          <svg viewBox="0 0 100 50" preserveAspectRatio="xMidYMid slice">
            <defs>
              <radialGradient id="kl-ql-g${idx}">
                <stop offset="0" stop-color="#233049" stop-opacity="0.9"/>
                <stop offset="1" stop-color="#0a0d14" stop-opacity="1"/>
              </radialGradient>
            </defs>
            <rect width="100" height="50" fill="url(#kl-ql-g${idx})"/>
            ${lines}
            ${stars.map(s => `<circle cx="${s.x}" cy="${s.y}" r="${s.r}" fill="${s.bright ? '#f2f2ee' : '#9a9a94'}" opacity="${s.bright ? 1 : 0.7}"/>`).join('')}
            ${bright.slice(0, 3).map(s => `<circle cx="${s.x}" cy="${s.y}" r="${s.r * 2.5}" fill="none" stroke="#f2f2ee" stroke-width="0.15" opacity="0.3"/>`).join('')}
            ${hasMoon ? `<circle cx="${moonX}" cy="${moonY}" r="5" fill="#e8d9a8" opacity="0.95"/><circle cx="${moonX - 1.5}" cy="${moonY - 0.5}" r="4.2" fill="#0a0d14" opacity="0.25"/>` : ''}
          </svg>
          <div class="coord"><span class="k">RA</span> ${rah}<sup>h</sup> ${ram}<sup>m</sup><br><span class="k">DEC</span> +${decd}° ${decm}'</div>
          <div class="tag">FIELD 00${idx + 1}</div>
        </div>
      `;
    }

    function obsRow(leaf: any, projectId: string) {
      const area = findAreaOf(leaf.id);
      const progress = Math.round(progressOf(leaf) * 100);
      const status = leaf.status || 'seed';
      const areaLabel = area && area.id !== leaf.id ? area.title : '';
      const checkN = leaf.checks.length;
      const checkDone = leaf.checks.filter((c: any) => c.done).length;
      return `
        <div class="obs ${state.selectedId === leaf.id ? 'selected' : ''}" data-status="${status}" data-proj="${projectId}" data-id="${leaf.id}">
          <div class="time">
            <b>${checkDone}/${checkN}</b>
            ${progress}%
          </div>
          <div class="body">
            <div class="lane"><span class="sw"></span>${esc(areaLabel ? areaLabel.toUpperCase() : 'DIRECT')}</div>
            <div class="t serif">${esc(leaf.title)}</div>
            ${leaf.note ? `<div class="n">${esc(leaf.note)}</div>` : ''}
            ${status !== 'seed' ? `<div class="mini-bar"><div class="f" style="width:${progress}%"></div></div>` : ''}
          </div>
          <div class="mag">${status.toUpperCase()}</div>
        </div>
      `;
    }

    function renderColumns() {
      if (state.view === 'trophy') { renderTrophyView(); return; }

      const wm = DATA.projects.find((p: any) => p.id === 'wm');
      const others = DATA.projects.filter((p: any) => p.id !== 'wm');

      const fw = byKey('featured-wrap');
      if (!fw) return;
      if (wm) {
        const wmAll = allLeaves(wm);
        const wmFire = wmAll.filter(l => l.status === 'fire').length;
        const wmSealedCount = DATA.sealed.filter((s: any) => s.project === wm.title).length;
        const wmProg = wmAll.length ? Math.round(wmAll.reduce((s, l) => s + progressOf(l), 0) / wmAll.length * 100) : 0;
        const cst = CONST_BY_PROJECT.wm;
        const { rah, ram, decd, decm } = coords(0);

        fw.innerHTML = `
          <div class="featured">
            <div class="f-left">
              <div class="f-sky">
                <img src="/apps/karmolab/img/widgets/quest-log/240714-071225.jpg" alt="">
                <div class="coord"><span class="k">RA</span> ${rah}<sup>h</sup> ${ram}<sup>m</sup><br><span class="k">DEC</span> +${decd}° ${decm}'<br><span class="k">MAG</span> ${cst.mag}</div>
                <div class="tag">★ MAIN PROJECT</div>
                <div class="overlay-title">
                  <div class="cst">✓ ${esc(cst.name.toUpperCase())} · ${esc(cst.sub)}</div>
                  <div class="name">${esc(wm.title)}</div>
                </div>
              </div>
              <div class="f-meta">
                <div class="eye">№ 00 · PRIMARY TARGET</div>
                <h2 class="serif">${esc(wm.title)}<em>${esc(wm.subtitle || '')}</em></h2>
                <div class="dek">${wm.children.length}개 영역. 채광, 전투, 농사, 마을 경영, 인형 부리기, 수집, 낚시, 스토리 — 기초 시스템들이 하나씩 자리를 잡으면 게임의 재미가 드러난다.</div>
                <div class="bar-line"><div class="fill" style="width:${wmProg}%"></div></div>
                <div class="mini-stats">
                  <div class="s accent"><div class="k">FIRE</div><div class="v">${wmFire}</div></div>
                  <div class="s"><div class="k">SEALED</div><div class="v">${wmSealedCount}</div></div>
                  <div class="s"><div class="k">COVERAGE</div><div class="v">${wmProg}<span style="font-family:'JetBrains Mono',monospace;font-weight:400;font-size:13.5px;color:var(--ink-2);margin-left:2px;">%</span></div></div>
                </div>
              </div>
            </div>
            <div class="f-right">
              <div class="log-head"><span>TASK LOG</span><span><b>${wmAll.length}</b> TASKS</span></div>
              ${wmAll.length ? wmAll.map(l => obsRow(l, 'wm')).join('') : '<div class="empty" style="grid-column: 1 / -1;">no tasks</div>'}
            </div>
          </div>
        `;
      } else {
        fw.innerHTML = '';
      }

      const subEl = byKey('sub-columns');
      if (!subEl) return;
      subEl.innerHTML = others.map((p: any, subIdx: number) => {
        const idx = subIdx + 1;
        const all = allLeaves(p);
        const totalP = all.length ? Math.round(all.reduce((s, l) => s + progressOf(l), 0) / all.length * 100) : 0;
        const fireCount = all.filter(l => l.status === 'fire').length;
        const cst = CONST_BY_PROJECT[p.id] || { name: p.title, sub: p.subtitle || '', mag: '—' };

        return `
          <div class="col" data-proj="${p.id}">
            <div class="col-head">
              <h3 class="serif"><span class="idx">№ ${String(idx + 1).padStart(2, '0')}</span>${esc(p.title)}</h3>
              <div class="sub">${esc(cst.name)} · <span style="font-style:italic;text-transform:none;letter-spacing:0.05em;">${esc(cst.sub)}</span></div>
              <div class="bar-line"><div class="fill" style="width:${totalP}%"></div></div>
              <div class="bar-meta">
                <b>${all.length} TASKS</b>
                <span>${fireCount} FIRE · ${totalP}% COVERAGE · MAG ${cst.mag}</span>
              </div>
            </div>
            ${skyHTML(idx, true)}
            <div class="log">
              ${all.length ? all.map(l => obsRow(l, p.id)).join('') : '<div class="empty">no tasks</div>'}
            </div>
          </div>
        `;
      }).join('');

      $$('.obs').forEach(el => {
        el.addEventListener('click', () => openDrawer(el.dataset.id!));
      });
    }

    function renderTrophyView() {
      const fw = byKey('featured-wrap');
      const subEl = byKey('sub-columns');
      if (!fw || !subEl) return;
      fw.innerHTML = '';
      if (DATA.sealed.length === 0) {
        subEl.innerHTML = '<div class="col" style="grid-column:1 / -1;"><div class="log" style="padding:40px;"><div class="empty">no sealed entries yet</div></div></div>';
        return;
      }
      subEl.innerHTML = `
        <div class="col" style="grid-column: 1 / -1;">
          <div class="col-head">
            <h3 class="serif"><span class="idx">◆</span>TROPHY ROOM</h3>
            <div class="sub">SEALED · <span style="font-style:italic;text-transform:none;letter-spacing:0.05em;">봉인된 것들</span></div>
            <div class="bar-line"><div class="fill" style="width:100%;background:var(--accent);"></div></div>
            <div class="bar-meta"><b>${DATA.sealed.length} ENTRIES</b><span>ARCHIVED</span></div>
          </div>
          <div class="log" style="padding: 8px 16px 20px;">
            ${DATA.sealed.map((t: any, i: number) => `
              <div class="obs" data-status="done">
                <div class="time"><b>№ ${String(i + 1).padStart(3, '0')}</b>SEALED</div>
                <div class="body">
                  <div class="lane"><span class="sw"></span>${esc(t.project.toUpperCase())}</div>
                  <div class="t serif">${esc(t.title)}</div>
                  ${t.note ? `<div class="n">${esc(t.note)}</div>` : ''}
                  ${t.sealedNote ? `<div class="n" style="font-style:italic;color:var(--accent);margin-top:6px;">"${esc(t.sealedNote)}"</div>` : ''}
                </div>
                <div class="mag">SEALED</div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    const drawer = byKey('drawer')!;
    const backdrop = byKey('backdrop')!;

    function openDrawer(id: string) {
      const f = findNode(id);
      if (!f) return;
      const node = f.node;
      const project = f.parents[0];
      const area = f.parents[1];
      state.selectedId = id;
      $$('.obs').forEach(el => el.classList.toggle('selected', el.dataset.id === id));

      const progress = Math.round(progressOf(node) * 100);
      const status = node.status || 'seed';
      const statusColor = status === 'fire' ? 'var(--accent)' :
                          status === 'sleep' ? 'var(--mag-learn)' :
                          status === 'sealed' ? 'var(--accent)' : 'var(--ink-3)';

      const crumb = byKey('crumb');
      if (crumb) crumb.innerHTML = `KMLB-QST / <b>${esc(node.id.toUpperCase())}</b>`;

      const body = byKey('drawer-body');
      if (!body) return;
      body.innerHTML = `
        <div class="lane-pill"><span class="sw" style="background: var(--accent);"></span>${esc(project ? project.title : '')}${area ? ' · ' + esc(area.title) : ''}</div>
        <h2 class="serif">${esc(node.title)} <em>${status.toUpperCase()}</em></h2>
        ${node.note ? `<p class="lede">${esc(node.note)}</p>` : ''}

        <div class="status-switcher" style="display:flex; gap:1px; margin-top:20px; background:var(--line-2); border:1px solid var(--line-2); width:fit-content;">
          ${['fire', 'seed', 'sleep'].map(s => `
            <button class="chip status-toggle ${status === s ? 'on' : ''}" data-set-status="${s}" style="border:none; padding:7px 14px;">
              ${s === 'fire' ? '◉ FIRE' : s === 'seed' ? '○ SEED' : '─ SLEEP'}
            </button>
          `).join('')}
        </div>

        <div class="priority-switcher" style="display:flex; gap:1px; margin-top:8px; background:var(--line-2); border:1px solid var(--line-2); width:fit-content;">
          ${['low', 'normal', 'high'].map(p => `
            <button class="chip priority-toggle ${node.memoPriority === p ? 'on' : ''}" data-set-priority="${p}" style="border:none; padding:7px 14px; font-size:12px;">
              ${p === 'high' ? '! HIGH' : p === 'low' ? '· LOW' : '○ NORMAL'}
            </button>
          `).join('')}
        </div>

        <div class="progress-wrap">
          <div class="lbl"><span>PROGRESS · ${starsHTML(progress / 100, false)}</span><b>${progress}%</b></div>
          <div class="bar"><div class="f" style="width:${progress}%; background:${statusColor};"></div></div>
          <div class="ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
        </div>

        ${isLeaf(node) ? `
          <div style="margin-top:24px; padding-top:18px; border-top:1px solid var(--line-2);">
            <div style="font-family:'JetBrains Mono',monospace;font-size:13px;letter-spacing:0.22em;text-transform:uppercase;color:var(--ink-3);margin-bottom:10px;">
              CHECKLIST · ${node.checks.filter((c: any) => c.done).length} / ${node.checks.length}
            </div>
            <div class="checklist">
              ${node.checks.map((c: any, i: number) => `
                <label class="check-row ${c.done ? 'done' : ''}" data-check-idx="${i}">
                  <input type="checkbox" ${c.done ? 'checked' : ''} style="display:none;">
                  <span class="check-box"></span>
                  <span class="check-label">${esc(c.t)}</span>
                  <button class="check-delete" data-check-del="${i}" title="삭제">×</button>
                </label>
              `).join('')}
            </div>
            <div class="add-check" style="display:flex; gap:1px; margin-top:12px; background:var(--line-2); border:1px solid var(--line-2);">
              <input type="text" placeholder="+ 새 항목…" />
              <button>ADD</button>
            </div>
          </div>
        ` : `
          <div style="margin-top:24px; padding-top:18px; border-top:1px solid var(--line-2);">
            <div style="font-family:'JetBrains Mono',monospace;font-size:13px;letter-spacing:0.22em;text-transform:uppercase;color:var(--ink-3);margin-bottom:10px;">
              SUB-AREAS · ${node.children.length}
            </div>
            <div class="children-list">
              ${node.children.map((c: any) => {
                const cp = Math.round(progressOf(c) * 100);
                const cs = c.status || 'seed';
                return `
                  <div class="child-row" data-child="${c.id}">
                    <span class="cr-status ${cs}">${cs.toUpperCase()}</span>
                    <span class="cr-title">${esc(c.title)}</span>
                    <span class="cr-right">${starsHTML(cp / 100)} <span style="font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--ink-3);margin-left:6px;">${cp}%</span></span>
                  </div>
                `;
              }).join('')}
            </div>
          </div>
        `}

        ${isLeaf(node) && progress >= 100 ? `
          <button class="seal-btn" data-seal="${node.id}">◆ SEAL TO TROPHY</button>
        ` : ''}
      `;

      $$('[data-check-idx]').forEach(el => {
        el.addEventListener('click', async (e) => {
          e.preventDefault();
          // 삭제 버튼 클릭은 별도 핸들러에서 처리 — 토글 안 함
          if ((e.target as HTMLElement).matches('[data-check-del]')) return;

          const i = Number(el.dataset.checkIdx);
          const check = node.checks[i];

          // 메모 정본 write-back (TASK-KL-017). filePath/lineNumber 가 있는 경우만.
          // 없으면 (옛 localStorage 데이터) 시각만 토글.
          const invoke = (window as any).__TAURI__?.core?.invoke;
          if (node.filePath && check.lineNumber && typeof invoke === 'function') {
            try {
              const newDone = await invoke('toggle_quest_check', {
                filePath: node.filePath,
                lineNumber: check.lineNumber,
                expectedText: check.t,
              }) as boolean;
              check.done = newDone;
            } catch (err) {
              console.error('toggle_quest_check 실패', err);
              alert(`체크박스 쓰기 실패: ${err}\n\n파일이 외부에서 변경됐을 수 있습니다. 위젯을 재실행해 주세요.`);
              return;
            }
          } else {
            check.done = !check.done;
          }

          if (check.done) Mdd.linePreset('success', { msg: '하나 끝!' });
          save();
          openDrawer(id);
          renderColumns();
          renderStats();
        });
      });

      $$('[data-check-del]').forEach(el => {
        el.addEventListener('click', async (e) => {
          e.preventDefault();
          e.stopPropagation();
          const i = Number(el.dataset.checkDel);
          const check = node.checks[i];
          if (!confirm(`정말 삭제할까요?\n\n"${check.t}"`)) return;

          const invoke = (window as any).__TAURI__?.core?.invoke;
          if (node.filePath && check.lineNumber && typeof invoke === 'function') {
            try {
              await invoke('delete_quest_check', {
                filePath: node.filePath,
                lineNumber: check.lineNumber,
                expectedText: check.t,
              });
            } catch (err) {
              console.error('delete_quest_check 실패', err);
              alert(`체크박스 삭제 실패: ${err}\n\n파일이 외부에서 변경됐을 수 있습니다. 위젯을 재실행해 주세요.`);
              return;
            }
          }

          // 파일에서 라인 1개 사라지면 그 뒤 체크박스들의 절대 라인 번호가 1씩 당겨짐.
          // in-memory 도 동기화 안 하면 다음 토글에서 text mismatch 로 실패.
          for (let j = i + 1; j < node.checks.length; j++) {
            if (typeof node.checks[j].lineNumber === 'number') {
              node.checks[j].lineNumber -= 1;
            }
          }
          node.checks.splice(i, 1);
          save();
          openDrawer(id);
          renderColumns();
          renderStats();
        });
      });

      $$('[data-set-status]').forEach(el => {
        el.addEventListener('click', async () => {
          const s = el.dataset.setStatus!;
          const newWidgetStatus = node.status === s ? 'seed' : s;

          // memo 정본 status write-back (TASK-KL-018). filePath/memoStatus 가 있는 경우만.
          const invoke = (window as any).__TAURI__?.core?.invoke;
          if (node.filePath && node.memoStatus && typeof invoke === 'function') {
            const newMemoStatus = mapWidgetStatusToMemo(newWidgetStatus);
            try {
              const written = await invoke('set_quest_status', {
                filePath: node.filePath,
                newStatus: newMemoStatus,
                expectedStatus: node.memoStatus,
              }) as string;
              node.memoStatus = written;
              node.status = mapMemoStatus(written);
            } catch (err) {
              console.error('set_quest_status 실패', err);
              alert(`상태 쓰기 실패: ${err}\n\n파일이 외부에서 변경됐을 수 있습니다. 위젯을 재실행해 주세요.`);
              return;
            }
          } else {
            node.status = newWidgetStatus;
          }

          if (node.status === 'fire') Mdd.linePreset('tool_run', { msg: '불 붙었어요 🔥' });
          save();
          openDrawer(id);
          renderColumns();
          renderStats();
        });
      });

      $$('[data-set-priority]').forEach(el => {
        el.addEventListener('click', async () => {
          const newPriority = el.dataset.setPriority!;
          // 같은 priority 클릭은 무동작 (status 와 달리 토글 의미 없음)
          if (node.memoPriority === newPriority) return;

          const invoke = (window as any).__TAURI__?.core?.invoke;
          if (node.filePath && node.memoPriority && typeof invoke === 'function') {
            try {
              const written = await invoke('set_quest_priority', {
                filePath: node.filePath,
                newPriority,
                expectedPriority: node.memoPriority,
              }) as string;
              node.memoPriority = written;
            } catch (err) {
              console.error('set_quest_priority 실패', err);
              alert(`우선순위 쓰기 실패: ${err}\n\n파일이 외부에서 변경됐을 수 있습니다. 위젯을 재실행해 주세요.`);
              return;
            }
          } else {
            node.memoPriority = newPriority;
          }

          save();
          openDrawer(id);
          renderColumns();
          renderStats();
        });
      });

      const input = root.querySelector('.add-check input') as HTMLInputElement | null;
      const btn = root.querySelector('.add-check button') as HTMLButtonElement | null;
      if (input && btn) {
        const add = async () => {
          const t = input.value.trim();
          if (!t) return;

          // memo 정본 write-back (TASK-KL-019). filePath 가 있는 경우만.
          const invoke = (window as any).__TAURI__?.core?.invoke;
          if (node.filePath && typeof invoke === 'function') {
            try {
              const newLineNumber = await invoke('add_quest_check', {
                filePath: node.filePath,
                text: t,
              }) as number;
              node.checks.push({ t, done: false, lineNumber: newLineNumber });
            } catch (err) {
              console.error('add_quest_check 실패', err);
              alert(`체크박스 추가 실패: ${err}`);
              return;
            }
          } else {
            node.checks.push({ t, done: false });
          }

          input.value = '';
          save();
          openDrawer(id);
          renderColumns();
          renderStats();
        };
        btn.addEventListener('click', add);
        input.addEventListener('keydown', e => { if (e.key === 'Enter') add(); });
      }

      $$('[data-child]').forEach(el => {
        el.addEventListener('click', () => openDrawer(el.dataset.child!));
      });

      const sealBtn = root.querySelector('[data-seal]') as HTMLButtonElement | null;
      if (sealBtn) {
        sealBtn.addEventListener('click', () => {
          DATA.sealed.unshift({
            id: 's-' + Date.now(),
            title: node.title,
            project: project ? project.title : '',
            note: node.note || '',
            sealedNote: '',
          });
          node.status = 'sealed';
          Mdd.linePreset('achievement', { msg: '봉인 완료. 트로피로 들어갔어요.' });
          save();
          closeDrawer();
          state.view = 'trophy';
          renderStats();
          renderColumns();
        });
      }

      drawer.classList.add('open');
      backdrop.classList.add('open');
    }

    function closeDrawer() {
      drawer.classList.remove('open');
      backdrop.classList.remove('open');
      state.selectedId = null;
      $$('.obs.selected').forEach(el => el.classList.remove('selected'));
    }
    const closeBtn = byKey('drawer-close');
    if (closeBtn) closeBtn.addEventListener('click', closeDrawer);
    backdrop.addEventListener('click', closeDrawer);

    // Esc 닫기 — 위젯이 DOM에 살아있는 동안만 (탭 전환·재빌드 시 자연 정리)
    function onEsc(e: KeyboardEvent) {
      if (!root.isConnected) {
        window.removeEventListener('keydown', onEsc);
        return;
      }
      if (e.key === 'Escape') closeDrawer();
    }
    window.addEventListener('keydown', onEsc);

    renderStats();
    renderColumns();
  }
})();
