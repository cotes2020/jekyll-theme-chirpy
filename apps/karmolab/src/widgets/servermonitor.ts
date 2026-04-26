(function (): void {
  'use strict';

  const REPO_ROOT_PREF = 'karmolab_repo_root';

  type DevProfile = {
    id: string;
    label: string;
    cwd: string;
    program: string;
    args: string[];
    healthUrl?: string;
    npmInstall?: boolean;
    /** `npm` + these args in profile `cwd` (e.g. Discord slash deploy) */
    deployArgs?: string[];
  };

  type RawLocalMonitor = {
    id: string;
    title?: string;
    subtitle?: string;
    label?: string;
    url?: string;
    noHealthUrl?: boolean;
  };

  type EnvFileEntry = {
    id: string;
    label?: string;
    relPath: string;
    hint?: string;
  };

  type ServerMonitorConfig = {
    localMonitors?: RawLocalMonitor[];
    envFiles?: EnvFileEntry[];
    devProfiles?: DevProfile[];
  };

  type LocalCardState = 'online' | 'offline' | 'na';

  type LocaldevLogPayload = {
    runId: string;
    profileId: string;
    stream: string;
    line: string;
  };

  type LocaldevDonePayload = {
    runId: string;
    profileId: string;
    kind: string;
    success: boolean;
    code?: number;
  };

  /**
   * dev profile별 현재 활성 로그 패널.
   * `renderMergedServices`가 카드를 다시 그릴 때마다 새 panel ref로 갱신된다.
   * `localdev-log` 이벤트(runId="follow")는 이 map을 lookup해서 해당 panel에 라인 append.
   */
  const followPanels = new Map<string, HTMLElement>();
  let followListenerUnlisten: (() => void) | null = null;
  let followListenerInstalling: Promise<void> | null = null;

  function appendLineToPanel(panel: HTMLElement, stream: string, line: string): void {
    const SM_LOG_MAX_LINES = 500;
    const SM_LOG_MAX_BYTES = 256 * 1024;
    const row = document.createElement('div');
    row.className =
      stream === 'err' ? 'sm-log-line sm-log-line-err' : 'sm-log-line sm-log-line-out';
    row.textContent = line;
    panel.appendChild(row);
    while (panel.childElementCount > SM_LOG_MAX_LINES) {
      panel.removeChild(panel.firstElementChild!);
    }
    let bytes = 0;
    for (let i = 0; i < panel.children.length; i++) {
      bytes += (panel.children[i].textContent || '').length + 1;
    }
    while (bytes > SM_LOG_MAX_BYTES && panel.firstElementChild) {
      panel.removeChild(panel.firstElementChild);
      bytes = 0;
      for (let j = 0; j < panel.children.length; j++) {
        bytes += (panel.children[j].textContent || '').length + 1;
      }
    }
    panel.scrollTop = panel.scrollHeight;
  }

  /** 한 번만 등록되는 글로벌 follow listener. install/deploy stream과는 runId="follow"로 구분. */
  async function ensureFollowListener(): Promise<void> {
    if (followListenerUnlisten) return;
    if (followListenerInstalling) return followListenerInstalling;
    const listen = window.__TAURI__?.event?.listen as
      | ((event: string, cb: (e: { payload: unknown }) => void) => Promise<() => void>)
      | undefined;
    if (typeof listen !== 'function') return;
    followListenerInstalling = (async () => {
      try {
        followListenerUnlisten = await listen('localdev-log', (e: { payload: unknown }) => {
          const pl = e.payload as LocaldevLogPayload;
          if (pl.runId !== 'follow') return;
          const panel = followPanels.get(pl.profileId);
          if (panel) appendLineToPanel(panel, pl.stream, pl.line);
        });
      } finally {
        followListenerInstalling = null;
      }
    })();
    return followListenerInstalling;
  }

  function isKarmolabDesktop(): boolean {
    return typeof window !== 'undefined' && !!window.__KARMOLAB_DESKTOP__;
  }

  function esc(s: string): string {
    if (Toolbox.escapeHtml) return Toolbox.escapeHtml(s);
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /** `localdev_list_tracked` 반환값이 배열이 아니거나 섞여 있을 때 대비 */
  function normalizeLocaldevTrackedIds(raw: unknown): string[] {
    if (!Array.isArray(raw)) return [];
    return raw.filter((x): x is string => typeof x === 'string');
  }

  function normalizeLocalMonitor(m: RawLocalMonitor): {
    id: string;
    title: string;
    subtitle: string;
    url?: string;
    canPing: boolean;
  } {
    const title = (m.title || m.label || m.id || '').trim();
    const subtitle = (m.subtitle || '').trim();
    const url = m.url?.trim();
    const canPing = !m.noHealthUrl && !!url;
    return { id: m.id, title, subtitle, url, canPing };
  }

  async function pingLocal(url: string): Promise<'online' | 'offline'> {
    try {
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), 2000);
      await fetch(url, { mode: 'no-cors', signal: controller.signal, cache: 'no-cache' });
      clearTimeout(timeoutId);
      return 'online';
    } catch {
      return 'offline';
    }
  }

  async function loadConfig(): Promise<ServerMonitorConfig> {
    const configPath = '/apps/karmolab/data/servermonitor-config.json';
    try {
      const res = await fetch(configPath, { cache: 'no-cache' });
      if (!res.ok) throw new Error('설정 파일을 찾을 수 없습니다.');
      return (await res.json()) as ServerMonitorConfig;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error('[ServerMonitor] 설정 로드 실패:', msg);
      return { localMonitors: [] };
    }
  }

  function localCardClass(state: LocalCardState): string {
    if (state === 'online') return 'sm-card sm-card--up';
    if (state === 'offline') return 'sm-card sm-card--down';
    return 'sm-card sm-card--na';
  }

  function localStatusLabel(state: LocalCardState): string {
    if (state === 'online') return '응답';
    if (state === 'offline') return '무응답';
    return 'HTTP 체크 없음';
  }

  type MergedServiceRow = {
    id: string;
    monitor?: ReturnType<typeof normalizeLocalMonitor>;
    profile?: DevProfile;
  };

  /** devProfiles 순서 우선, localMonitors만 있는 항목은 뒤에 붙임 */
  function mergeServiceRows(config: ServerMonitorConfig): MergedServiceRow[] {
    const profiles = config.devProfiles ?? [];
    const rawLocals = config.localMonitors ?? [];
    const monById = new Map<string, ReturnType<typeof normalizeLocalMonitor>>();
    for (const m of rawLocals) {
      monById.set(m.id, normalizeLocalMonitor(m));
    }
    const seen = new Set<string>();
    const rows: MergedServiceRow[] = [];
    for (const p of profiles) {
      rows.push({ id: p.id, profile: p, monitor: monById.get(p.id) });
      seen.add(p.id);
    }
    for (const m of rawLocals) {
      if (!seen.has(m.id)) {
        rows.push({ id: m.id, monitor: normalizeLocalMonitor(m) });
      }
    }
    return rows;
  }

  function mergedPingRowClass(
    monitor: MergedServiceRow['monitor'],
    raw: LocalCardState | undefined
  ): string {
    if (!monitor) return 'sm-card sm-card--na';
    if (!monitor.canPing) return 'sm-card sm-card--na';
    if (raw === undefined) return 'sm-card sm-card--na';
    return localCardClass(raw);
  }

  function mergedPingRowText(
    monitor: MergedServiceRow['monitor'],
    raw: LocalCardState | undefined
  ): string {
    if (!monitor) return 'URL 모니터 없음';
    if (!monitor.canPing) return 'HTTP 체크 없음';
    if (raw === undefined) return '조회 전';
    return localStatusLabel(raw);
  }

  type NormalizedMonitor = ReturnType<typeof normalizeLocalMonitor>;

  function mergedCardShellClass(
    monitor: MergedServiceRow['monitor'],
    raw: LocalCardState | undefined
  ): string {
    return mergedPingRowClass(monitor, raw).replace(/^sm-card\s+/, 'sm-card sm-card--merged ');
  }

  function smEscapeAttr(id: string): string {
    if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') return CSS.escape(id);
    return id.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  }

  const smPingFlashTimers = new WeakMap<HTMLElement, number>();

  function patchMergedCardPingUi(
    card: HTMLElement,
    monitor: NormalizedMonitor | undefined,
    raw: LocalCardState
  ): void {
    card.className = mergedCardShellClass(monitor, raw);
    const statusRow = card.querySelector('.sm-card-status');
    const spans = statusRow?.querySelectorAll('span');
    if (spans && spans.length >= 2) {
      spans[spans.length - 1].textContent = mergedPingRowText(monitor, raw);
    }
  }

  /** ping 직후: 연두 플래시 + ✓가 잠깐 나타났다 사라짐 */
  function flashMergedCardPingDone(card: HTMLElement): void {
    const prev = smPingFlashTimers.get(card);
    if (prev !== undefined) window.clearTimeout(prev);
    card.querySelectorAll('.sm-card-ping-check').forEach((n) => n.remove());
    card.classList.remove('sm-card--ping-flash');
    void card.offsetWidth;
    card.classList.add('sm-card--ping-flash');
    const check = document.createElement('span');
    check.className = 'sm-card-ping-check';
    check.textContent = '✓';
    check.setAttribute('aria-hidden', 'true');
    card.appendChild(check);
    const tid = window.setTimeout(() => {
      check.remove();
      card.classList.remove('sm-card--ping-flash');
      smPingFlashTimers.delete(card);
    }, 900);
    smPingFlashTimers.set(card, tid);
  }

  /** 데스크톱: 저장소 기준 .env 바로가기(탐색기 / 기본 앱 / 인앱 편집) */
  function mountEnvFilesPanel(host: HTMLElement): void {
    const invoke = window.__TAURI__?.core?.invoke;

    host.className = 'sm-env-section';

    const title = document.createElement('div');
    title.className = 'sm-desktop-section-title';
    title.textContent = '환경 변수 (.env)';

    const hint = document.createElement('p');
    hint.className = 'sm-dev-hint';
    hint.textContent =
      '페이지 하단에서 저장소 루트를 저장한 뒤 쓰세요. 경로는 servermonitor-config.json → envFiles 입니다. 비밀 값은 화면 공유에 주의하세요.';

    const grid = document.createElement('div');
    grid.className = 'sm-env-cards';

    host.appendChild(title);
    host.appendChild(hint);
    host.appendChild(grid);

    void loadConfig().then((cfg) => {
      const files = cfg.envFiles ?? [];
      if (files.length === 0) {
        grid.textContent = 'envFiles 항목이 없습니다.';
        return;
      }
      if (typeof invoke !== 'function') {
        grid.textContent = 'Tauri invoke를 사용할 수 없습니다.';
        return;
      }

      for (const f of files) {
        const rel = (f.relPath || '').trim();
        if (!rel) continue;

        const card = document.createElement('div');
        card.className = 'sm-card sm-card--env';

        const t = document.createElement('div');
        t.className = 'sm-card-title';
        t.textContent = f.label?.trim() || f.id;

        const pathEl = document.createElement('div');
        pathEl.className = 'sm-env-path mono';
        pathEl.textContent = rel;

        card.appendChild(t);
        card.appendChild(pathEl);

        if (f.hint?.trim()) {
          const h = document.createElement('div');
          h.className = 'sm-card-sub';
          h.textContent = f.hint.trim();
          card.appendChild(h);
        }

        const actions = document.createElement('div');
        actions.className = 'sm-card-actions';

        const mk = (label: string, fn: () => void): HTMLButtonElement => {
          const b = document.createElement('button');
          b.type = 'button';
          b.className = 'btn btn-ghost';
          b.textContent = label;
          b.onclick = () => fn();
          return b;
        };

        actions.appendChild(
          mk('탐색기', () => {
            void (async () => {
              try {
                await invoke('repofile_reveal', { relPath: rel });
              } catch (e: unknown) {
                Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
              }
            })();
          })
        );
        actions.appendChild(
          mk('앱으로 열기', () => {
            void (async () => {
              try {
                await invoke('repofile_open_default', { relPath: rel });
              } catch (e: unknown) {
                Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
              }
            })();
          })
        );

        const editorWrap = document.createElement('div');
        editorWrap.className = 'sm-env-editor-wrap';
        editorWrap.hidden = true;
        const ta = document.createElement('textarea');
        ta.className = 'mono-input sm-env-ta';
        ta.spellcheck = false;
        ta.setAttribute('aria-label', `${f.label || f.id} 환경 변수 내용`);

        let loaded = false;
        const btnEdit = mk('편집', () => {
          void (async () => {
            if (!editorWrap.hidden) {
              editorWrap.hidden = true;
              btnEdit.textContent = '편집';
              return;
            }
            editorWrap.hidden = false;
            btnEdit.textContent = '접기';
            if (loaded) return;
            try {
              const text = (await invoke('repofile_read', { relPath: rel })) as string;
              ta.value = text;
              loaded = true;
            } catch (e: unknown) {
              const msg = typeof e === 'string' ? e : e instanceof Error ? e.message : String(e);
              if (msg.includes('FILE_NOT_FOUND')) {
                ta.value = '';
                loaded = true;
                Toolbox.showToast?.('새 파일입니다. 저장 시 생성됩니다.', undefined, undefined);
              } else {
                Toolbox.showToast?.(msg, 'error', undefined);
              }
            }
          })();
        });

        const btnSave = mk('저장', () => {
          void (async () => {
            try {
              await invoke('repofile_write', { relPath: rel, content: ta.value });
              loaded = true;
              Toolbox.showToast?.('저장됨', undefined, undefined);
            } catch (e: unknown) {
              Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
            }
          })();
        });
        btnSave.className = 'btn btn-primary';

        const saveRow = document.createElement('div');
        saveRow.className = 'sm-env-editor-actions';
        saveRow.appendChild(btnSave);
        editorWrap.appendChild(ta);
        editorWrap.appendChild(saveRow);

        actions.appendChild(btnEdit);
        card.appendChild(actions);
        card.appendChild(editorWrap);
        grid.appendChild(card);
      }
    });
  }

  /**
   * 데스크톱: 루트 + 서비스당 카드 1장(localMonitors URL 응답 + devProfiles 프로세스 병합).
   * `pingState.byId`는 새로고침 시 갱신되고, 이 함수가 DOM을 다시 그립니다.
   */
  function mountDesktopLocalDev(
    section: HTMLElement,
    rootFooter: HTMLElement,
    pingState: { byId: Record<string, LocalCardState> },
    registerRefresh: (fn: () => Promise<void>) => void,
    triggerStatusFetchSoon: (delayMs: number) => void
  ): HTMLElement {
    const invoke = window.__TAURI__?.core?.invoke;

    const SM_LOG_MAX_LINES = 500;
    const SM_LOG_MAX_BYTES = 256 * 1024;

    function appendSmLogLine(panel: HTMLElement, stream: string, line: string): void {
      const row = document.createElement('div');
      row.className =
        stream === 'err' ? 'sm-log-line sm-log-line-err' : 'sm-log-line sm-log-line-out';
      row.textContent = line;
      panel.appendChild(row);
      while (panel.childElementCount > SM_LOG_MAX_LINES) {
        panel.removeChild(panel.firstElementChild!);
      }
      let bytes = 0;
      for (let i = 0; i < panel.children.length; i++) {
        bytes += (panel.children[i].textContent || '').length + 1;
      }
      while (bytes > SM_LOG_MAX_BYTES && panel.firstElementChild) {
        panel.removeChild(panel.firstElementChild);
        bytes = 0;
        for (let j = 0; j < panel.children.length; j++) {
          bytes += (panel.children[j].textContent || '').length + 1;
        }
      }
      panel.scrollTop = panel.scrollHeight;
    }

    async function runStreamedNpmOp(
      cmd: 'localdev_deploy_stream' | 'localdev_npm_install_stream',
      profileId: string,
      logPanel: HTMLElement,
      disableBtns: HTMLButtonElement[],
      okFallback: string
    ): Promise<void> {
      const inv = window.__TAURI__?.core?.invoke;
      const listen = window.__TAURI__?.event?.listen as
        | ((event: string, cb: (e: { payload: unknown }) => void) => Promise<() => void>)
        | undefined;
      if (typeof inv !== 'function') return;
      if (typeof listen !== 'function') {
        Toolbox.showToast?.('Tauri event.listen을 쓸 수 없습니다.', 'error', undefined);
        return;
      }
      for (const b of disableBtns) b.disabled = true;
      logPanel.replaceChildren();
      let unLog: (() => void) | undefined;
      let unDone: (() => void) | undefined;
      try {
        unLog = await listen('localdev-log', (e: { payload: unknown }) => {
          const pl = e.payload as LocaldevLogPayload;
          if (pl.profileId !== profileId) return;
          // follow 라인은 글로벌 follow listener가 같은 panel에 이미 append하므로 여기선 skip
          if (pl.runId === 'follow') return;
          appendSmLogLine(logPanel, pl.stream, pl.line);
        });
        unDone = await listen('localdev-log-done', (e: { payload: unknown }) => {
          const pl = e.payload as LocaldevDonePayload;
          if (pl.profileId !== profileId) return;
        });
        const msg = (await inv(cmd, { profileId })) as string;
        Toolbox.showToast?.(msg || okFallback, undefined, undefined);
      } catch (e: unknown) {
        Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
      } finally {
        unLog?.();
        unDone?.();
        for (const b of disableBtns) b.disabled = false;
      }
    }

    const rootLabel = document.createElement('div');
    rootLabel.className = 'sm-root-footer-label';
    rootLabel.textContent = '저장소 루트 (레포 최상위)';

    const rootInput = document.createElement('input');
    rootInput.type = 'text';
    rootInput.className = 'mono-input sm-root-footer-input';
    rootInput.placeholder = '예: C:\\Users\\…\\Mascari4615.github.io';
    rootInput.value = Toolbox.getPref?.(REPO_ROOT_PREF, '') ?? '';

    const saveRootBtn = document.createElement('button');
    saveRootBtn.className = 'btn btn-ghost btn-sm';
    saveRootBtn.type = 'button';
    saveRootBtn.textContent = '저장';
    saveRootBtn.onclick = () => {
      void (async () => {
        const v = rootInput.value.trim();
        if (Toolbox.setPref) Toolbox.setPref(REPO_ROOT_PREF, v);
        if (typeof invoke !== 'function') {
          Toolbox.showToast?.('Tauri invoke를 쓸 수 없습니다.', 'error', undefined);
          return;
        }
        try {
          await invoke('localdev_set_repo_root', { path: v });
          Toolbox.showToast?.('저장소 루트 저장됨', undefined, undefined);
        } catch (e: unknown) {
          const msg = e instanceof Error ? e.message : String(e);
          Toolbox.showToast?.(msg, 'error', undefined);
        }
      })();
    };

    const refreshListBtn = document.createElement('button');
    refreshListBtn.className = 'btn btn-ghost';
    refreshListBtn.type = 'button';
    refreshListBtn.textContent = '목록 새로고침';

    const listRow = document.createElement('div');
    listRow.className = 'sm-list-refresh-row';

    const servicesWrap = document.createElement('div');
    servicesWrap.className = 'sm-local-services';

    /** DOM에 이미 그려진 카드의 추적 라벨만 Rust 상태와 다시 맞춤(시작/종료 직후 이중 확인용) */
    async function refreshTrackLabelsFromRust(): Promise<void> {
      if (typeof invoke !== 'function') return;
      try {
        const tracked = normalizeLocaldevTrackedIds(await invoke('localdev_list_tracked'));
        for (const card of servicesWrap.querySelectorAll<HTMLElement>('[data-sm-service-id]')) {
          const id = card.dataset.smServiceId;
          if (!id) continue;
          const trackEl = card.querySelector('.sm-card-track');
          if (!trackEl) continue;
          trackEl.textContent = tracked.includes(id) ? '앱 추적 중' : '미실행';
        }
      } catch (e) {
        console.warn('[ServerMonitor] 추적 목록 재조회 실패 — 카드의 「앱 추적 중」이 어긋날 수 있음', e);
      }
    }

    async function renderMergedServices(): Promise<void> {
      const config = await loadConfig();
      const rows = mergeServiceRows(config);
      let tracked: string[] = [];
      if (typeof invoke === 'function') {
        try {
          tracked = normalizeLocaldevTrackedIds(await invoke('localdev_list_tracked'));
        } catch (e) {
          console.warn('[ServerMonitor] localdev_list_tracked 실패 — 추적 상태를 표시할 수 없음', e);
          tracked = [];
        }
      }

      servicesWrap.replaceChildren();
      if (rows.length === 0) {
        servicesWrap.textContent = 'localMonitors·devProfiles가 비어 있거나 설정을 불러오지 못했습니다.';
        return;
      }

      const mkBtn = (label: string, onClick: () => void): HTMLButtonElement => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn btn-ghost';
        b.textContent = label;
        b.onclick = () => onClick();
        return b;
      };

      for (const row of rows) {
        const p = row.profile;
        const mon = row.monitor;
        const rawPing = mon ? pingState.byId[mon.id] : undefined;
        const cardClass = mergedPingRowClass(mon, rawPing).replace(/^sm-card\s+/, 'sm-card sm-card--merged ');

        const card = document.createElement('div');
        card.className = cardClass;
        card.dataset.smServiceId = row.id;

        // 한 줄 리스트 레이아웃: head/ping/track/actions/로그 토글을 한 row에 정렬, 로그 패널은 아래로 collapsible
        const cardRow = document.createElement('div');
        cardRow.className = 'sm-card-row';

        const head = document.createElement('div');
        head.className = 'sm-card-head';
        const t = document.createElement('div');
        t.className = 'sm-card-title';
        t.textContent = p?.label || mon?.title || row.id;
        const sub = document.createElement('div');
        sub.className = 'sm-card-sub mono';
        sub.style.opacity = '0.85';
        const subParts: string[] = [];
        if (p) subParts.push(p.id);
        if (mon?.subtitle) subParts.push(mon.subtitle);
        sub.textContent = subParts.join(' · ') || row.id;
        head.appendChild(t);
        head.appendChild(sub);

        const pingRow = document.createElement('div');
        pingRow.className = 'sm-card-status';
        const dot = document.createElement('span');
        dot.className = 'sm-card-status-dot';
        dot.setAttribute('aria-hidden', 'true');
        const pingLabel = document.createElement('span');
        pingLabel.textContent = mergedPingRowText(mon, rawPing);
        pingRow.appendChild(dot);
        pingRow.appendChild(pingLabel);
        cardRow.appendChild(head);
        cardRow.appendChild(pingRow);
        card.appendChild(cardRow);

        if (p) {
          const deployArgsFiltered =
            p.deployArgs?.filter((a) => (a || '').trim().length > 0) ?? [];
          const streamActionBtns: HTMLButtonElement[] = [];
          let logPanelEl: HTMLElement | null = null;

          const track = document.createElement('div');
          track.className = 'sm-card-track';
          track.textContent = tracked.includes(p.id) ? '앱 추적 중' : '미실행';
          cardRow.appendChild(track);

          const actions = document.createElement('div');
          actions.className = 'sm-card-actions';

          actions.appendChild(
            mkBtn('시작', () => {
              void (async () => {
                if (typeof invoke !== 'function') return;
                try {
                  await invoke('localdev_start', { profileId: p.id });
                  Toolbox.showToast?.(`${p.label} 시작됨`, undefined, undefined);
                  await renderMergedServices();
                  await refreshTrackLabelsFromRust();
                  triggerStatusFetchSoon(800);
                } catch (e: unknown) {
                  Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
                }
              })();
            })
          );
          actions.appendChild(
            mkBtn('종료', () => {
              void (async () => {
                if (typeof invoke !== 'function') return;
                try {
                  await invoke('localdev_stop', { profileId: p.id });
                  Toolbox.showToast?.(`${p.label} 종료 요청`, undefined, undefined);
                  await renderMergedServices();
                  await refreshTrackLabelsFromRust();
                  triggerStatusFetchSoon(400);
                } catch (e: unknown) {
                  Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
                }
              })();
            })
          );

          if (p.npmInstall) {
            const btnInstall = mkBtn('npm i', () => {
              if (!logPanelEl) return;
              void runStreamedNpmOp(
                'localdev_npm_install_stream',
                p.id,
                logPanelEl,
                streamActionBtns,
                'npm install 완료'
              );
            });
            streamActionBtns.push(btnInstall);
            actions.appendChild(btnInstall);
          }

          if (deployArgsFiltered.length > 0) {
            const btnDeploy = mkBtn('deploy', () => {
              if (!logPanelEl) return;
              void runStreamedNpmOp(
                'localdev_deploy_stream',
                p.id,
                logPanelEl,
                streamActionBtns,
                'deploy 완료'
              );
            });
            btnDeploy.title = `npm ${deployArgsFiltered.join(' ')} (${p.cwd})`;
            streamActionBtns.push(btnDeploy);
            actions.appendChild(btnDeploy);
          }

          cardRow.appendChild(actions);

          // dev profile 카드는 항상 로그 패널을 가진다 (기본 접힘, 토글로 펼침).
          // - 시작 버튼으로 띄운 봇의 stdout/stderr는 Rust가 로그 파일로 redirect하고
          //   `localdev_follow_log`가 그 파일을 tail해서 `localdev-log`로 emit한다.
          // - npm i / deploy 스트림도 같은 패널을 공유 (시간순으로 섞여 흐름).
          {
            const logWrap = document.createElement('div');
            logWrap.className = 'sm-log-wrap';
            logWrap.hidden = true;
            const hint = document.createElement('p');
            hint.className = 'sm-log-hint';
            hint.textContent =
              '로그에 토큰·경로 등이 섞일 수 있습니다. 화면 공유·녹화 시 주의하세요.';
            logPanelEl = document.createElement('div');
            logPanelEl.className = 'sm-log-panel mono';
            logPanelEl.setAttribute('role', 'log');
            logPanelEl.setAttribute('aria-live', 'polite');
            logWrap.appendChild(hint);
            logWrap.appendChild(logPanelEl);

            const logToggle = document.createElement('button');
            logToggle.type = 'button';
            logToggle.className = 'sm-log-toggle';
            logToggle.textContent = '▸ 로그';
            logToggle.setAttribute('aria-expanded', 'false');
            logToggle.onclick = () => {
              const collapsed = logWrap.hidden;
              logWrap.hidden = !collapsed;
              logToggle.textContent = collapsed ? '▾ 로그' : '▸ 로그';
              logToggle.setAttribute('aria-expanded', collapsed ? 'true' : 'false');
              if (collapsed && logPanelEl) logPanelEl.scrollTop = logPanelEl.scrollHeight;
            };
            cardRow.appendChild(logToggle);
            card.appendChild(logWrap);
          }

          // 카드 그릴 때마다 follow 패널 등록 + Rust follow 시작 (이미 follow 중이면 noop).
          followPanels.set(p.id, logPanelEl!);
          void ensureFollowListener();
          if (typeof invoke === 'function') {
            void invoke('localdev_follow_log', { profileId: p.id }).catch((e) => {
              console.warn('[ServerMonitor] localdev_follow_log 실패', e);
            });
          }
        }

        servicesWrap.appendChild(card);
      }
    }

    registerRefresh(renderMergedServices);
    refreshListBtn.onclick = () => void renderMergedServices();

    listRow.appendChild(refreshListBtn);
    section.appendChild(listRow);
    section.appendChild(servicesWrap);

    const rootRow = document.createElement('div');
    rootRow.className = 'sm-root-footer-row';
    rootRow.appendChild(rootInput);
    rootRow.appendChild(saveRootBtn);
    rootFooter.className = 'sm-root-footer';
    rootFooter.appendChild(rootLabel);
    rootFooter.appendChild(rootRow);

    void (async () => {
      if (typeof invoke === 'function' && rootInput.value.trim()) {
        try {
          await invoke('localdev_set_repo_root', { path: rootInput.value.trim() });
        } catch {
          /* ignore */
        }
      }
      if (typeof invoke === 'function') {
        try {
          const fromRust = (await invoke('localdev_get_repo_root')) as string | null;
          if (fromRust) rootInput.value = fromRust;
        } catch {
          /* ignore */
        }
      }
    })();

    return servicesWrap;
  }

  function build(container: HTMLElement): void {
    const statusBox = document.createElement('div');
    statusBox.id = 'smStatusBox';
    statusBox.className = 'sm-status-wrap';

    let refreshDevTable: (() => Promise<void>) | null = null;
    const pingState: { byId: Record<string, LocalCardState> } = { byId: {} };
    let mergedServicesEl: HTMLElement | null = null;

    const refreshWrap = document.createElement('div');
    refreshWrap.className = 'sm-refresh-wrap';

    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.className = 'sm-refresh-icon-btn';
    refreshBtn.title = '새로고침 (URL ping·프로세스 추적)';
    refreshBtn.setAttribute('aria-label', '새로고침');
    refreshBtn.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>';

    const refreshProgressEl = document.createElement('span');
    refreshProgressEl.className = 'sm-refresh-progress';
    refreshProgressEl.setAttribute('aria-live', 'polite');
    refreshProgressEl.hidden = true;

    refreshWrap.appendChild(refreshBtn);
    refreshWrap.appendChild(refreshProgressEl);

    function setRefreshBusy(busy: boolean, progressLabel = ''): void {
      refreshBtn.disabled = busy;
      refreshBtn.setAttribute('aria-busy', busy ? 'true' : 'false');
      if (!busy) {
        refreshProgressEl.hidden = true;
        refreshProgressEl.replaceChildren();
        return;
      }
      refreshProgressEl.hidden = false;
      refreshProgressEl.replaceChildren();
      const spin = document.createElement('span');
      spin.className = 'sm-spinner';
      spin.setAttribute('aria-hidden', 'true');
      refreshProgressEl.appendChild(spin);
      if (progressLabel) {
        const lab = document.createElement('span');
        lab.className = 'sm-refresh-progress-text';
        lab.textContent = progressLabel;
        refreshProgressEl.appendChild(lab);
      }
    }

    Mdd.injectCSS(
      'servermonitor',
      `
            .sm-status-wrap { margin-top: 16px; font-size: var(--font-size-sm); }
            .sm-status-wrap.loading { color: var(--text-tertiary); padding: 16px; }
            .sm-status-wrap.error { color: var(--error, #e74c3c); padding: 16px; border: 1px solid var(--error); border-radius: var(--radius-md); }
            .sm-section-label { font-weight: 700; margin: 0 0 10px 0; color: var(--accent); font-size: var(--font-size-sm); letter-spacing: 0.02em; }
            .sm-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)); gap: 12px; margin-bottom: 20px; }
            .sm-card { border-radius: var(--radius-md); border: 1px solid var(--border); background: var(--bg-secondary); padding: 14px 16px; min-height: 108px; display: flex; flex-direction: column; transition: border-color 0.15s, box-shadow 0.15s; }
            .sm-card:hover { border-color: var(--border-hover); box-shadow: var(--shadow-sm, 0 1px 4px rgba(0,0,0,.08)); }
            .sm-card--up { border-left: 4px solid var(--success, #22c55e); }
            .sm-card--down { border-left: 4px solid var(--error, #e74c3c); }
            .sm-card--na { border-left: 4px solid var(--text-tertiary); }
            .sm-card-title { font-weight: 700; font-size: var(--font-size-md); color: var(--text-primary); line-height: 1.25; }
            .sm-card-sub { font-size: var(--font-size-xs); color: var(--text-tertiary); margin-top: 6px; line-height: 1.35; }
            .sm-card-status { margin-top: auto; padding-top: 12px; display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: var(--font-size-xs); }
            .sm-card-status-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
            .sm-card--up .sm-card-status-dot { background: var(--success, #22c55e); box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.25); }
            .sm-card--down .sm-card-status-dot { background: var(--error, #e74c3c); }
            .sm-card--na .sm-card-status-dot { background: var(--text-tertiary); }
            .sm-local-section { margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border); }
            .sm-local-header-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }
            .sm-browser-local-header { margin-top: 12px; }
            .sm-local-title-text { margin-bottom: 0 !important; flex: 1; min-width: 0; }
            .sm-refresh-wrap { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
            .sm-refresh-icon-btn { display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; padding: 0; border: 1px solid var(--border); border-radius: var(--radius-md); background: var(--bg-secondary); color: var(--text-primary); cursor: pointer; transition: border-color 0.15s, color 0.15s; }
            .sm-refresh-icon-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
            .sm-refresh-icon-btn:disabled { opacity: 0.55; cursor: not-allowed; }
            .sm-refresh-icon-btn svg { width: 16px; height: 16px; }
            .sm-refresh-progress { display: inline-flex; align-items: center; gap: 6px; font-size: var(--font-size-2xs); color: var(--text-tertiary); min-height: 16px; }
            .sm-spinner { width: 14px; height: 14px; flex-shrink: 0; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: sm-spin 0.7s linear infinite; }
            @keyframes sm-spin { to { transform: rotate(360deg); } }
            .sm-desktop-section-title { font-weight: 700; margin-bottom: 10px; color: var(--accent); }
            .sm-local-services { display: flex; flex-direction: column; gap: 6px; margin-top: 4px; }
            .sm-card--merged { min-height: auto; position: relative; display: flex; flex-direction: column; padding: 0; }
            .sm-card-row { display: flex; flex-direction: row; align-items: center; gap: 12px; padding: 8px 14px; flex-wrap: wrap; }
            .sm-card--merged .sm-card-head { flex: 1 1 200px; min-width: 0; margin-bottom: 0; display: flex; flex-direction: column; gap: 2px; }
            .sm-card--merged .sm-card-title { font-size: var(--font-size-sm); }
            .sm-card--merged .sm-card-sub { margin-top: 0; font-size: var(--font-size-2xs); }
            .sm-card--merged .sm-card-status { margin-top: 0; padding-top: 0; flex-shrink: 0; min-width: 80px; }
            .sm-card--merged .sm-card-track { margin-top: 0; flex-shrink: 0; min-width: 70px; font-size: var(--font-size-2xs); color: var(--text-secondary); }
            .sm-card--merged .sm-card-actions { margin-top: 0; flex-shrink: 0; flex-wrap: nowrap; }
            .sm-log-toggle { background: transparent; border: 1px solid var(--border); color: var(--text-secondary); border-radius: var(--radius-sm); padding: 3px 9px; cursor: pointer; font-size: var(--font-size-2xs); flex-shrink: 0; }
            .sm-log-toggle:hover { border-color: var(--accent); color: var(--accent); }
            .sm-card--merged .sm-log-wrap { padding: 0 14px 10px; margin-top: 0; }
            .sm-card--ping-flash { animation: sm-ping-flash-bg 0.95s ease forwards; }
            @keyframes sm-ping-flash-bg {
              0%, 100% { box-shadow: none; }
              12% { box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.45), 0 4px 14px rgba(34, 197, 94, 0.12); background-color: rgba(34, 197, 94, 0.1); }
              40% { box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.2); background-color: rgba(34, 197, 94, 0.05); }
            }
            .sm-card-ping-check {
              position: absolute;
              top: 10px;
              right: 11px;
              font-size: 1.1rem;
              line-height: 1;
              pointer-events: none;
              color: var(--success, #22c55e);
              animation: sm-ping-check-pop 0.88s ease forwards;
              text-shadow: 0 0 6px var(--bg-secondary, #1a1a1a);
            }
            @keyframes sm-ping-check-pop {
              0% { opacity: 0; transform: scale(0.45); }
              22% { opacity: 1; transform: scale(1.12); }
              50% { opacity: 1; transform: scale(1); }
              100% { opacity: 0; transform: scale(0.88); }
            }
            .sm-dev-hint { font-size: var(--font-size-sm); color: var(--text-tertiary); margin-bottom: 12px; line-height: 1.5; }
            .sm-dev-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
            .sm-card--dev { min-height: auto; }
            .sm-card-head { margin-bottom: 10px; }
            .sm-card-track { font-size: var(--font-size-2xs); color: var(--text-secondary); margin-top: 6px; }
            .sm-card-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
            .sm-env-section { margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border); }
            .sm-env-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
            .sm-card--env { min-height: auto; }
            .sm-env-path { font-size: var(--font-size-2xs); color: var(--text-secondary); margin-top: 4px; word-break: break-all; line-height: 1.35; }
            .sm-env-editor-wrap { margin-top: 10px; }
            .sm-env-ta { width: 100%; min-height: 140px; margin-top: 8px; font-size: var(--font-size-xs); resize: vertical; box-sizing: border-box; }
            .sm-env-editor-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 8px; }
            .sm-list-refresh-row { margin-bottom: 10px; }
            .sm-root-footer { margin-top: 20px; padding-top: 14px; border-top: 1px solid var(--border); }
            .sm-root-footer-label { font-size: var(--font-size-2xs); color: var(--text-tertiary); margin-bottom: 6px; letter-spacing: 0.02em; }
            .sm-root-footer-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
            .sm-root-footer-input { flex: 1; min-width: 140px; margin-bottom: 0 !important; }
            .btn-sm { padding: 4px 10px; font-size: var(--font-size-xs); }
            .sm-log-wrap { margin-top: 10px; width: 100%; min-width: 0; }
            .sm-log-hint { margin: 0 0 6px 0; font-size: var(--font-size-2xs); color: var(--text-tertiary); line-height: 1.4; }
            .sm-log-panel {
              max-height: 200px;
              overflow: auto;
              padding: 8px 10px;
              border-radius: var(--radius-md);
              border: 1px solid var(--border);
              background: var(--bg-primary, #0f0f12);
              font-size: var(--font-size-2xs);
              line-height: 1.45;
              text-align: left;
              white-space: pre-wrap;
              word-break: break-word;
            }
            .sm-log-line { margin: 0; padding: 0; }
            .sm-log-line-err { color: var(--error, #e74c3c); }
            .sm-log-line-out { color: var(--text-secondary, #94a3b8); }
        `
    );

    container.innerHTML = '';

    if (isKarmolabDesktop()) {
      const localSection = document.createElement('div');
      localSection.className = 'sm-local-section';

      const localHeaderRow = document.createElement('div');
      localHeaderRow.className = 'sm-local-header-row';

      const localTitle = document.createElement('div');
      localTitle.className = 'sm-desktop-section-title sm-local-title-text';
      localTitle.textContent = '로컬';

      localHeaderRow.appendChild(localTitle);
      localHeaderRow.appendChild(refreshWrap);

      const localHint = document.createElement('p');
      localHint.className = 'sm-dev-hint';
      localHint.textContent =
        '같은 id의 localMonitors·devProfiles는 카드 한 장에 묶입니다. 오른쪽 버튼으로 URL 응답·프로세스 추적을 갱신합니다.';

      localSection.appendChild(localHeaderRow);
      localSection.appendChild(localHint);
      const rootFooter = document.createElement('div');
      mergedServicesEl = mountDesktopLocalDev(
        localSection,
        rootFooter,
        pingState,
        (fn) => {
          refreshDevTable = fn;
        },
        triggerStatusFetchSoon
      );

      container.appendChild(localSection);

      const envHost = document.createElement('div');
      mountEnvFilesPanel(envHost);
      container.appendChild(envHost);

      container.appendChild(rootFooter);
    } else {
      const browserHeader = document.createElement('div');
      browserHeader.className = 'sm-local-header-row sm-browser-local-header';
      const bt = document.createElement('div');
      bt.className = 'sm-desktop-section-title sm-local-title-text';
      bt.textContent = '로컬';
      browserHeader.appendChild(bt);
      browserHeader.appendChild(refreshWrap);
      container.appendChild(browserHeader);
      container.appendChild(statusBox);
    }

    /**
     * @param rerenderCards true 면 ping 후 카드 전체 다시 그리기(refreshDevTable). 수동 새로고침/시작·종료 직후 트리거에서만 true.
     *   자동 polling 은 false — 카드 그대로 두고 ping/track 만 patch 해서 사용자가 펼쳐둔 로그가 살아남음.
     */
    async function fetchStatus(rerenderCards: boolean = false): Promise<void> {
      setRefreshBusy(true, '불러오는 중');
      /* 데스크톱: 카드 영역은 비우지 않음(첫 접속도 골격 카드 먼저 그린 뒤 이 함수로 ping만 갱신) */
      if (!mergedServicesEl) {
        if (!statusBox.querySelector('.sm-cards .sm-card')) {
          statusBox.innerHTML = '조회 중…';
          statusBox.className = 'sm-status-wrap loading';
        }
      }

      let skipFinalMergeRefresh = false;
      try {
        const config = await loadConfig();
        const rawLocals = config.localMonitors ?? [];
        const normalized = rawLocals.map(normalizeLocalMonitor);

        const totalPings = normalized.filter((m) => m.canPing && m.url).length;
        let pingDone = 0;

        // 모든 ping 동시 발사 — 직렬은 5개 × 2s = 10s, 병렬은 max 2s.
        const localResults: Array<{ meta: (typeof normalized)[0]; state: LocalCardState }> = [];
        await Promise.all(
          normalized.map(async (meta) => {
            if (!meta.canPing || !meta.url) {
              const state: LocalCardState = 'na';
              localResults.push({ meta, state });
              pingState.byId[meta.id] = state;
              if (mergedServicesEl) {
                const card = mergedServicesEl.querySelector(
                  `[data-sm-service-id="${smEscapeAttr(meta.id)}"]`
                ) as HTMLElement | null;
                if (card) patchMergedCardPingUi(card, meta, state);
              }
              return;
            }
            const s = await pingLocal(meta.url!);
            pingDone++;
            setRefreshBusy(true, totalPings ? `URL ${pingDone}/${totalPings}` : '확인 중');
            localResults.push({ meta, state: s });
            pingState.byId[meta.id] = s;
            if (mergedServicesEl) {
              const card = mergedServicesEl.querySelector(
                `[data-sm-service-id="${smEscapeAttr(meta.id)}"]`
              ) as HTMLElement | null;
              if (card) {
                patchMergedCardPingUi(card, meta, s);
                flashMergedCardPingDone(card);
              }
            }
          })
        );

        setRefreshBusy(true, '동기화 중');

        const localCardsHtml = localResults
          .map(({ meta, state }) => {
            const cls = localCardClass(state);
            const sub = meta.subtitle
              ? `<div class="sm-card-sub">${esc(meta.subtitle)}</div>`
              : '';
            return `<div class="${cls}">
              <div class="sm-card-title">${esc(meta.title)}</div>
              ${sub}
              <div class="sm-card-status"><span class="sm-card-status-dot" aria-hidden="true"></span><span>${esc(localStatusLabel(state))}</span></div>
            </div>`;
          })
          .join('');

        if (!mergedServicesEl) {
          statusBox.innerHTML = `
          <div class="sm-section-label">로컬</div>
          <div class="sm-cards">${localCardsHtml || '<p class="sm-card-sub" style="grid-column:1/-1">localMonitors가 비어 있습니다.</p>'}</div>
        `;
          statusBox.className = 'sm-status-wrap';
        }
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : '알 수 없는 오류';
        if (mergedServicesEl) {
          Toolbox.showToast?.(`조회 실패: ${msg}`, 'error', undefined);
          skipFinalMergeRefresh = true;
        } else {
          statusBox.innerHTML = `조회 실패: ${esc(msg)}`;
          statusBox.className = 'sm-status-wrap error';
        }
      } finally {
        // rerenderCards 가 true 일 때만 카드 전체 다시 그리기 (config 변경/시작·종료 직후 사용자 액션). polling 은 false 라 카드 유지 → 사용자가 펼쳐둔 로그 그대로.
        if (rerenderCards && !skipFinalMergeRefresh) {
          try {
            const doRefresh = refreshDevTable as (() => Promise<void>) | null;
            if (doRefresh) await doRefresh();
          } catch {
            /* ignore */
          }
        }
        setRefreshBusy(false);
      }
    }

    refreshBtn.onclick = () => void fetchStatus(true);

    /** 시작/종료 직후 짧게 기다린 뒤 ping 한 번. 새 프로세스가 listen 시작할 시간 줌. 카드는 시작/종료 콜백이 이미 다시 그리니 ping 만 patch. */
    let pendingFetchTimer: number | null = null;
    function triggerStatusFetchSoon(delayMs: number): void {
      if (pendingFetchTimer != null) window.clearTimeout(pendingFetchTimer);
      pendingFetchTimer = window.setTimeout(() => {
        pendingFetchTimer = null;
        if (refreshBtn.disabled) return;
        void fetchStatus(false);
      }, delayMs);
    }
    /** 데스크톱 카드의 시작/종료 콜백에서 ping을 직접 트리거할 수 있게 노출 */
    (window as unknown as { __sm_triggerStatusFetchSoon?: typeof triggerStatusFetchSoon })
      .__sm_triggerStatusFetchSoon = triggerStatusFetchSoon;

    void (async () => {
      if (mergedServicesEl) {
        try {
          const doRefresh = refreshDevTable as (() => Promise<void>) | null;
          if (doRefresh) await doRefresh();
        } catch {
          /* ignore */
        }
      }
      await fetchStatus(false);
      // 자동 폴링: 5초마다 ping 만 patch (rerenderCards=false). 카드 DOM 그대로 → 사용자가 펼쳐둔 로그 보존.
      window.setInterval(() => {
        if (refreshBtn.disabled) return;
        if (typeof document !== 'undefined' && document.hidden) return;
        void fetchStatus(false);
      }, 5000);
    })();
  }

  Toolbox.register({
    id: 'servermonitor',
    title: '서버 모니터',
    category: 'desktop',
    desc: '로컬 URL·프로세스·.env (데스크톱)',
    layout: 'form',
    icon: '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
    tabs: [{ id: 'main', label: '상태', build }]
  });
})();
