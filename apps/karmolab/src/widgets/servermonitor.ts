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
      '위에서 저장소 루트를 먼저 저장하세요. 경로는 servermonitor-config.json → envFiles 입니다. 비밀 값은 화면 공유에 주의하세요.';

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
    pingState: { byId: Record<string, LocalCardState> },
    registerRefresh: (fn: () => Promise<void>) => void
  ): HTMLElement {
    const invoke = window.__TAURI__?.core?.invoke;

    const rootLabel = document.createElement('label');
    rootLabel.className = 'field-label';
    rootLabel.textContent = '프로젝트(저장소) 루트 경로';

    const rootInput = document.createElement('input');
    rootInput.type = 'text';
    rootInput.className = 'mono-input';
    rootInput.style.width = '100%';
    rootInput.style.marginBottom = '8px';
    rootInput.placeholder = '예: C:\\Users\\…\\Mascari4615.github.io';
    rootInput.value = Toolbox.getPref?.(REPO_ROOT_PREF, '') ?? '';

    const saveRootBtn = document.createElement('button');
    saveRootBtn.className = 'btn btn-primary';
    saveRootBtn.textContent = '루트 저장';
    saveRootBtn.style.marginRight = '8px';
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
    refreshListBtn.textContent = '목록 새로고침';

    const servicesWrap = document.createElement('div');
    servicesWrap.className = 'sm-local-services';

    async function renderMergedServices(): Promise<void> {
      const config = await loadConfig();
      const rows = mergeServiceRows(config);
      let tracked: string[] = [];
      if (typeof invoke === 'function') {
        try {
          tracked = (await invoke('localdev_list_tracked')) as string[];
        } catch {
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
        card.appendChild(head);
        card.appendChild(pingRow);

        if (p) {
          const track = document.createElement('div');
          track.className = 'sm-card-track';
          track.textContent = tracked.includes(p.id) ? '앱 추적 중' : '미실행';
          card.appendChild(track);

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
                } catch (e: unknown) {
                  Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
                }
              })();
            })
          );

          if (p.npmInstall) {
            const btnInstall = mkBtn('npm i', () => {
              void (async () => {
                if (typeof invoke !== 'function') return;
                btnInstall.disabled = true;
                try {
                  const msg = (await invoke('localdev_npm_install', { profileId: p.id })) as string;
                  Toolbox.showToast?.(msg || 'npm install 완료', undefined, undefined);
                } catch (e: unknown) {
                  Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
                } finally {
                  btnInstall.disabled = false;
                }
              })();
            });
            actions.appendChild(btnInstall);
          }

          card.appendChild(actions);
        }

        servicesWrap.appendChild(card);
      }
    }

    registerRefresh(renderMergedServices);
    refreshListBtn.onclick = () => void renderMergedServices();

    section.appendChild(rootLabel);
    section.appendChild(rootInput);
    const rootRow = document.createElement('div');
    rootRow.style.marginBottom = '12px';
    rootRow.appendChild(saveRootBtn);
    rootRow.appendChild(refreshListBtn);
    section.appendChild(rootRow);
    section.appendChild(servicesWrap);

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
      await renderMergedServices();
    })();

    return servicesWrap;
  }

  function build(container: HTMLElement): void {
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'btn btn-primary';
    refreshBtn.textContent = '새로고침';

    const statusBox = document.createElement('div');
    statusBox.id = 'smStatusBox';
    statusBox.className = 'sm-status-wrap';

    let refreshDevTable: (() => Promise<void>) | null = null;
    const pingState: { byId: Record<string, LocalCardState> } = { byId: {} };
    let mergedServicesEl: HTMLElement | null = null;

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
            .sm-desktop-section-title { font-weight: 700; margin-bottom: 10px; color: var(--accent); }
            .sm-local-services { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-top: 4px; }
            .sm-card--merged { min-height: auto; }
            .sm-card--merged .sm-card-status { margin-top: 10px; padding-top: 0; }
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
        `
    );

    container.innerHTML = '';
    const btnRow = document.createElement('div');
    btnRow.style.marginBottom = '8px';
    btnRow.appendChild(refreshBtn);
    container.appendChild(btnRow);

    if (isKarmolabDesktop()) {
      const localSection = document.createElement('div');
      localSection.className = 'sm-local-section';

      const localTitle = document.createElement('div');
      localTitle.className = 'sm-desktop-section-title';
      localTitle.textContent = '로컬';

      const localHint = document.createElement('p');
      localHint.className = 'sm-dev-hint';
      localHint.textContent =
        '같은 id의 localMonitors·devProfiles는 카드 한 장에 묶입니다. 새로고침으로 URL 응답·프로세스 추적을 갱신합니다.';

      localSection.appendChild(localTitle);
      localSection.appendChild(localHint);
      mergedServicesEl = mountDesktopLocalDev(localSection, pingState, (fn) => {
        refreshDevTable = fn;
      });

      container.appendChild(localSection);

      const envHost = document.createElement('div');
      mountEnvFilesPanel(envHost);
      container.appendChild(envHost);
    } else {
      container.appendChild(statusBox);
    }

    async function fetchStatus(): Promise<void> {
      refreshBtn.disabled = true;
      if (mergedServicesEl) {
        mergedServicesEl.innerHTML =
          '<p class="sm-card-sub" style="grid-column:1/-1;padding:12px 4px">조회 중…</p>';
      } else {
        statusBox.innerHTML = '조회 중…';
        statusBox.className = 'sm-status-wrap loading';
      }

      let skipFinalMergeRefresh = false;
      try {
        const config = await loadConfig();
        const rawLocals = config.localMonitors ?? [];
        const normalized = rawLocals.map(normalizeLocalMonitor);

        const localResults: Array<{ meta: (typeof normalized)[0]; state: LocalCardState }> = [];
        for (const meta of normalized) {
          if (!meta.canPing || !meta.url) {
            localResults.push({ meta, state: 'na' });
          } else {
            const s = await pingLocal(meta.url);
            localResults.push({ meta, state: s });
          }
        }

        for (const { meta, state } of localResults) {
          pingState.byId[meta.id] = state;
        }

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
          mergedServicesEl.innerHTML = `<p class="sm-card-sub" style="padding:12px 4px;color:var(--error,#e74c3c)">조회 실패: ${esc(msg)}</p>`;
          skipFinalMergeRefresh = true;
        } else {
          statusBox.innerHTML = `조회 실패: ${esc(msg)}`;
          statusBox.className = 'sm-status-wrap error';
        }
      } finally {
        refreshBtn.disabled = false;
        if (!skipFinalMergeRefresh) {
          try {
            await refreshDevTable?.();
          } catch {
            /* ignore */
          }
        }
      }
    }

    refreshBtn.onclick = () => void fetchStatus();
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
