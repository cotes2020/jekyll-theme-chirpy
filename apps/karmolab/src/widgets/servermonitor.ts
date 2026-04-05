(function (): void {
  'use strict';

  const PREFS_KEY = 'servermonitor_base';
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

  type ServerMonitorConfig = {
    localMonitors?: Array<{ label: string; url: string }>;
    devProfiles?: DevProfile[];
  };

  type RemoteStatusData = {
    error?: string;
    cpu?: number;
    memory?: { used_gb?: number; total_gb?: number; percent?: number };
    disk?: { used_gb?: number; total_gb?: number; percent?: number };
    uptime?: string;
    services?: Record<string, string>;
  };

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

  /** 데스크톱: 상태 화면 하단에 프로필 시작·종료 UI. `registerRefresh`로 테이블 갱신 함수를 넘깁니다. */
  function mountDesktopLocalDev(
    container: HTMLElement,
    registerRefresh: (fn: () => Promise<void>) => void
  ): void {
    Mdd.injectCSS(
      'servermonitor-localdev',
      `
            .sm-dev-wrap { margin-top: 12px; overflow-x: auto; }
            .sm-dev-table { width: 100%; border-collapse: collapse; font-size: var(--font-size-sm); }
            .sm-dev-table th, .sm-dev-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: middle; }
            .sm-dev-table th { color: var(--text-tertiary); font-weight: 600; }
            .sm-dev-actions { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
            .sm-dev-hint { font-size: var(--font-size-sm); color: var(--text-tertiary); margin-bottom: 12px; line-height: 1.5; }
            .sm-desktop-section { margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border); }
            .sm-desktop-section-title { font-weight: 700; margin-bottom: 10px; color: var(--accent); }
        `
    );

    const invoke = window.__TAURI__?.core?.invoke;

    const section = document.createElement('div');
    section.className = 'sm-desktop-section';

    const title = document.createElement('div');
    title.className = 'sm-desktop-section-title';
    title.textContent = '로컬 프로세스 (데스크톱)';

    const hint = document.createElement('p');
    hint.className = 'sm-dev-hint';
    hint.textContent =
      '저장소 루트를 저장한 뒤 프로필을 시작·종료합니다. 명령은 apps/karmolab/data/servermonitor-config.json의 devProfiles에서 읽습니다. 위의 상태 조회로 URL 응답을 확인하세요. Node·Ruby 등은 PATH에 있어야 합니다.';

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
    refreshListBtn.textContent = '프로필 목록 새로고침';

    const tableWrap = document.createElement('div');
    tableWrap.className = 'sm-dev-wrap';

    async function refreshTable(): Promise<void> {
      const config = await loadConfig();
      const profiles = config.devProfiles ?? [];
      let tracked: string[] = [];
      if (typeof invoke === 'function') {
        try {
          tracked = (await invoke('localdev_list_tracked')) as string[];
        } catch {
          tracked = [];
        }
      }

      tableWrap.innerHTML = '';
      if (profiles.length === 0) {
        tableWrap.textContent = 'devProfiles가 비어 있거나 설정을 불러오지 못했습니다.';
        return;
      }

      const tbl = document.createElement('table');
      tbl.className = 'sm-dev-table';
      const thead = document.createElement('thead');
      thead.innerHTML =
        '<tr><th>프로필</th><th>앱 추적</th><th style="text-align:right">동작</th></tr>';
      tbl.appendChild(thead);
      const tb = document.createElement('tbody');

      for (const p of profiles) {
        const tr = document.createElement('tr');
        const tdName = document.createElement('td');
        tdName.innerHTML = `<strong>${esc(p.label)}</strong><div class="mono" style="opacity:0.85;font-size:0.9em">${esc(p.id)}</div>`;

        const tdTrack = document.createElement('td');
        tdTrack.textContent = tracked.includes(p.id) ? '추적 중' : '없음';

        const tdAct = document.createElement('td');
        tdAct.style.textAlign = 'right';
        const actions = document.createElement('div');
        actions.className = 'sm-dev-actions';

        const btnStart = document.createElement('button');
        btnStart.className = 'btn btn-ghost';
        btnStart.type = 'button';
        btnStart.textContent = '시작';
        btnStart.onclick = () => {
          void (async () => {
            if (typeof invoke !== 'function') return;
            try {
              await invoke('localdev_start', { profileId: p.id });
              Toolbox.showToast?.(`${p.label} 시작됨 (백그라운드)`, undefined, undefined);
              await refreshTable();
            } catch (e: unknown) {
              Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
            }
          })();
        };

        const btnStop = document.createElement('button');
        btnStop.className = 'btn btn-ghost';
        btnStop.type = 'button';
        btnStop.textContent = '종료';
        btnStop.onclick = () => {
          void (async () => {
            if (typeof invoke !== 'function') return;
            try {
              await invoke('localdev_stop', { profileId: p.id });
              Toolbox.showToast?.(`${p.label} 종료 요청`, undefined, undefined);
              await refreshTable();
            } catch (e: unknown) {
              Toolbox.showToast?.(e instanceof Error ? e.message : String(e), 'error', undefined);
            }
          })();
        };

        actions.appendChild(btnStart);
        actions.appendChild(btnStop);

        if (p.npmInstall) {
          const btnInstall = document.createElement('button');
          btnInstall.className = 'btn btn-ghost';
          btnInstall.type = 'button';
          btnInstall.textContent = 'npm install';
          btnInstall.onclick = () => {
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
          };
          actions.appendChild(btnInstall);
        }

        tdAct.appendChild(actions);
        tr.appendChild(tdName);
        tr.appendChild(tdTrack);
        tr.appendChild(tdAct);
        tb.appendChild(tr);
      }

      tbl.appendChild(tb);
      tableWrap.appendChild(tbl);
    }

    registerRefresh(refreshTable);
    refreshListBtn.onclick = () => void refreshTable();

    section.appendChild(title);
    section.appendChild(hint);
    section.appendChild(rootLabel);
    section.appendChild(rootInput);
    const rootRow = document.createElement('div');
    rootRow.style.marginBottom = '12px';
    rootRow.appendChild(saveRootBtn);
    rootRow.appendChild(refreshListBtn);
    section.appendChild(rootRow);
    section.appendChild(tableWrap);
    container.appendChild(section);

    void (async () => {
      if (typeof invoke === 'function' && rootInput.value.trim()) {
        try {
          await invoke('localdev_set_repo_root', { path: rootInput.value.trim() });
        } catch {
          /* Rust 쪽 검증 실패 시 무시 — 사용자가 다시 저장 */
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
      await refreshTable();
    })();
  }

  function build(container: HTMLElement): void {
    const baseInput = document.createElement('input');
    baseInput.type = 'url';
    baseInput.id = 'smBaseUrl';
    baseInput.className = 'mono-input';
    baseInput.placeholder = 'http://서버IP:5000';
    baseInput.style.width = '100%';
    baseInput.style.marginBottom = '12px';
    if (Toolbox.getPref) {
      baseInput.value = Toolbox.getPref(PREFS_KEY, '') || Toolbox.getPref('ytdl_cobalt_base', '') || '';
    }

    const saveBtn = document.createElement('button');
    saveBtn.className = 'btn btn-ghost';
    saveBtn.textContent = '저장';
    saveBtn.style.marginBottom = '16px';
    saveBtn.onclick = function (): void {
      const v = baseInput.value.trim();
      if (Toolbox.setPref) Toolbox.setPref(PREFS_KEY, v);
      Toolbox.showToast?.('저장됨', undefined, undefined);
    };

    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'btn btn-primary';
    refreshBtn.textContent = '상태 조회';
    refreshBtn.style.marginLeft = '8px';

    const statusBox = document.createElement('div');
    statusBox.id = 'smStatusBox';
    statusBox.className = 'sm-status-box';

    let refreshDevTable: (() => Promise<void>) | null = null;

    Mdd.injectCSS(
      'servermonitor',
      `
            .sm-status-box { margin-top:16px; padding:16px; border-radius:var(--radius-md); background:var(--bg-tertiary); border:1px solid var(--border); font-size:var(--font-size-sm); }
            .sm-status-box.loading { color:var(--text-tertiary); }
            .sm-status-box.error { color:var(--error, #e74c3c); border-color:var(--error, #e74c3c); }
            .sm-row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--border); }
            .sm-row:last-child { border-bottom:none; }
            .sm-services { display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:8px; margin-top:12px; }
            .sm-service { padding:8px 12px; border-radius:var(--radius-sm); background:var(--bg-secondary); text-align:center; }
            .sm-service.ok { border-left:3px solid var(--success, #22c55e); }
            .sm-service.running { border-left:3px solid var(--success, #22c55e); }
            .sm-service.unknown { border-left:3px solid var(--text-tertiary); }
            .sm-service.offline { border-left:3px solid var(--error, #e74c3c); }
        `
    );

    container.innerHTML = '';
    const label = document.createElement('label');
    label.className = 'field-label';
    label.textContent = '서버 URL (yt-api 배포 주소)';
    container.appendChild(label);
    container.appendChild(baseInput);
    const btnRow = document.createElement('div');
    btnRow.appendChild(saveBtn);
    btnRow.appendChild(refreshBtn);
    container.appendChild(btnRow);
    container.appendChild(statusBox);

    if (isKarmolabDesktop()) {
      mountDesktopLocalDev(container, (fn) => {
        refreshDevTable = fn;
      });
    }

    async function fetchStatus(): Promise<void> {
      let base =
        baseInput.value.trim() || (Toolbox.getPref && Toolbox.getPref(PREFS_KEY, '')) || '';
      if (!base && Toolbox.getPref) base = Toolbox.getPref('ytdl_cobalt_base', '') || '';

      statusBox.innerHTML = '조회 중...';
      statusBox.className = 'sm-status-box loading';
      refreshBtn.disabled = true;

      try {
        const config = await loadConfig();
        const localTargets = config.localMonitors ?? [];

        const localResults = await Promise.all(
          localTargets.map(async (m) => {
            const status = await pingLocal(m.url);
            return { ...m, status };
          })
        );

        let remoteHtml = '';
        if (base) {
          try {
            const url = base.replace(/\/$/, '');
            const res = await fetch(`${url}/api/status`);
            const data = (await res.json().catch(() => ({}))) as RemoteStatusData;

            if (data.error) {
              remoteHtml = `<div class="sm-row"><span style="color:var(--error)">원격 서버 오류: ${data.error}</span></div>`;
            } else {
              const m = data.memory ?? {};
              const d = data.disk ?? {};
              const svc = data.services ?? {};
              const ytStatus = svc['yt-api'] === 'ok' ? 'ok' : 'offline';
              const dcStatus =
                svc['discord-bot'] === 'running'
                  ? 'running'
                  : svc['discord-bot'] === 'unknown'
                    ? 'unknown'
                    : 'offline';

              remoteHtml = `
                                <div class="sm-row"><span>원격 서버</span><span style="color:var(--success)">● 온라인</span></div>
                                <div class="sm-row"><span>CPU</span><span>${data.cpu ?? '-'}%</span></div>
                                <div class="sm-row"><span>메모리</span><span>${m.used_gb}/${m.total_gb} GB (${m.percent}%)</span></div>
                                <div class="sm-row"><span>디스크</span><span>${d.used_gb}/${d.total_gb} GB (${d.percent}%)</span></div>
                                <div class="sm-row"><span>가동시간</span><span>${data.uptime ?? '-'}</span></div>
                                <div class="sm-services">
                                    <div class="sm-service ${ytStatus}"><strong>yt-api</strong><br>${ytStatus === 'ok' ? '정상' : '오프라인'}</div>
                                    <div class="sm-service ${dcStatus}"><strong>봇 서버</strong><br>${dcStatus === 'running' ? '실행 중' : dcStatus === 'unknown' ? '확인 불가' : '오프라인'}</div>
                                </div>
                            `;
            }
          } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : 'Error';
            remoteHtml = `<div class="sm-row"><span style="color:var(--error)">원격 연결 실패: ${msg}</span></div>`;
          }
        } else {
          remoteHtml = `<div class="sm-row"><span style="color:var(--text-tertiary)">원격 서버가 설정되지 않았습니다.</span></div>`;
        }

        const localHtml = localResults
          .map(
            (r) => `
                    <div class="sm-row">
                        <span>${r.label}</span>
                        <span style="color:${r.status === 'online' ? 'var(--success)' : 'var(--error)'}">
                            ● ${r.status === 'online' ? 'Run' : 'Down'}
                        </span>
                    </div>
                `
          )
          .join('');

        statusBox.innerHTML = `
                    <div style="font-weight:700; margin-bottom:10px; color:var(--accent)">내 컴퓨터 서버 상태</div>
                    ${localHtml}
                    <div style="font-weight:700; margin-top:20px; margin-bottom:10px; color:var(--accent)">원격 서버 상태</div>
                    ${remoteHtml}
                `;
        statusBox.className = 'sm-status-box';
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : '알 수 없는 오류';
        statusBox.innerHTML = `조회 실패: ${msg}`;
        statusBox.className = 'sm-status-box error';
      } finally {
        refreshBtn.disabled = false;
        try {
          await refreshDevTable?.();
        } catch {
          /* 프로필 표 갱신 실패는 치명적이지 않음 */
        }
      }
    }

    refreshBtn.onclick = () => void fetchStatus();
  }

  Toolbox.register({
    id: 'servermonitor',
    title: '서버 모니터',
    desc: '로컬·원격 서버 상태를 확인하고(데스크톱) dev 프로필을 실행합니다',
    layout: 'form',
    icon: '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
    tabs: [{ id: 'main', label: '상태', build }]
  });
})();
