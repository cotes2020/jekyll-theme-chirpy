/**
 * PC 활동 트래커 위젯 — 데스크톱 앱 전용.
 * Tauri Rust collector(activity.rs)가 매 5초마다 포그라운드 윈도우를 샘플링해 JSONL로 저장한다.
 * 이 위젯은 그 데이터를 일자별로 조회·집계해 보여줌. 외부 전송 없음.
 */
(function (): void {
  'use strict';

  type ActivitySample = {
    ts: number;
    process: string;
    title: string;
    idle: boolean;
  };

  type DayActivity = {
    day: string;
    samples: ActivitySample[];
  };

  type AppAggregate = {
    process: string;
    seconds: number;
    titles: Map<string, number>;
  };

  const SAMPLE_INTERVAL_SECS = 5;

  function desktopInvoke(cmd: string, args: unknown): Promise<unknown> {
    const core = window.__TAURI__?.core;
    const fn = core && typeof core.invoke === 'function' ? core.invoke : null;
    if (!fn) return Promise.reject(new Error('Tauri invoke 없음 (웹 브라우저 또는 withGlobalTauri 비활성)'));
    return fn(cmd, args);
  }

  function todayUtcDay(): string {
    // 데이터는 UTC 기준 일별 파일이라 위젯도 UTC 일자를 키로 사용.
    const d = new Date();
    const y = d.getUTCFullYear();
    const m = String(d.getUTCMonth() + 1).padStart(2, '0');
    const day = String(d.getUTCDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  }

  function formatDuration(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}시간 ${m}분`;
    if (m > 0) return `${m}분 ${s}초`;
    return `${s}초`;
  }

  function aggregate(samples: ActivitySample[]): {
    apps: AppAggregate[];
    activeSecs: number;
    idleSecs: number;
  } {
    const map = new Map<string, AppAggregate>();
    let activeSecs = 0;
    let idleSecs = 0;
    for (const s of samples) {
      if (s.idle) {
        idleSecs += SAMPLE_INTERVAL_SECS;
        continue;
      }
      activeSecs += SAMPLE_INTERVAL_SECS;
      const key = s.process || '(unknown)';
      let agg = map.get(key);
      if (!agg) {
        agg = { process: key, seconds: 0, titles: new Map() };
        map.set(key, agg);
      }
      agg.seconds += SAMPLE_INTERVAL_SECS;
      const titleKey = s.title || '(no title)';
      agg.titles.set(titleKey, (agg.titles.get(titleKey) || 0) + SAMPLE_INTERVAL_SECS);
    }
    const apps = Array.from(map.values()).sort((a, b) => b.seconds - a.seconds);
    return { apps, activeSecs, idleSecs };
  }

  function escapeHtml(s: string): string {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function build(container: HTMLElement): void {
    Mdd.injectCSS(
      'activity',
      `
            .activity-root { max-width: 720px; }
            .activity-intro { font-size: var(--font-size-sm); color: var(--text-tertiary); margin: 0 0 16px 0; line-height: 1.5; }
            .activity-controls { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }
            .activity-controls label { font-size: var(--font-size-xs); color: var(--text-secondary); font-weight: 600; }
            .activity-controls input[type=date] {
                padding: 6px 10px; border-radius: var(--radius-md);
                border: 1px solid var(--border); background: var(--bg-primary);
                color: var(--text-primary); font-size: var(--font-size-sm);
            }
            .activity-summary { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }
            .activity-stat {
                padding: 14px 16px; border-radius: var(--radius-md);
                background: var(--bg-tertiary); border: 1px solid var(--border);
            }
            .activity-stat-label { font-size: var(--font-size-xs); color: var(--text-tertiary); margin-bottom: 4px; }
            .activity-stat-value { font-size: 20px; font-weight: 700; color: var(--text-primary); }
            .activity-list { display: flex; flex-direction: column; gap: 8px; }
            .activity-row {
                position: relative; padding: 10px 12px; border-radius: var(--radius-md);
                background: var(--bg-tertiary); border: 1px solid var(--border);
                overflow: hidden;
            }
            .activity-row-bar {
                position: absolute; inset: 0;
                background: color-mix(in srgb, var(--accent) 18%, transparent);
                width: 0;
                pointer-events: none;
            }
            .activity-row-content { position: relative; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
            .activity-row-name { font-weight: 600; color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
            .activity-row-time { color: var(--text-secondary); font-size: var(--font-size-sm); font-variant-numeric: tabular-nums; flex-shrink: 0; }
            .activity-titles {
                position: relative; margin-top: 8px; padding-top: 8px;
                border-top: 1px dashed var(--border);
                display: none; flex-direction: column; gap: 4px;
                font-size: var(--font-size-xs); color: var(--text-tertiary);
            }
            .activity-row.open .activity-titles { display: flex; }
            .activity-title-row { display: flex; justify-content: space-between; gap: 8px; }
            .activity-title-row .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
            .activity-empty { padding: 30px 20px; text-align: center; color: var(--text-tertiary); }
            .activity-disabled-note { color: var(--text-tertiary); font-size: var(--font-size-sm); padding: 30px 20px; text-align: center; }
        `
    );

    container.innerHTML = '';
    const root = document.createElement('div');
    root.className = 'activity-root';

    const intro = document.createElement('p');
    intro.className = 'activity-intro';
    const isApp = typeof Toolbox.isDesktopApp === 'function' && Toolbox.isDesktopApp();
    intro.innerHTML = isApp
      ? `5초마다 포그라운드 윈도우를 샘플링합니다. 데이터는 전부 로컬에만 저장됩니다 (외부 전송 0). 60초 이상 입력 없으면 idle로 분류.`
      : `데스크톱 앱 전용입니다. KarmoLab Tauri 앱으로 열어주세요.`;
    root.appendChild(intro);

    if (!isApp) {
      const note = document.createElement('div');
      note.className = 'activity-disabled-note';
      note.textContent = '브라우저에서는 사용할 수 없습니다.';
      root.appendChild(note);
      container.appendChild(root);
      return;
    }

    const controls = document.createElement('div');
    controls.className = 'activity-controls';
    const lab = document.createElement('label');
    lab.textContent = '날짜 (UTC)';
    const dateIn = document.createElement('input');
    dateIn.type = 'date';
    dateIn.value = todayUtcDay();
    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.className = 'btn btn-secondary';
    refreshBtn.textContent = '새로고침';
    controls.appendChild(lab);
    controls.appendChild(dateIn);
    controls.appendChild(refreshBtn);
    root.appendChild(controls);

    const summary = document.createElement('div');
    summary.className = 'activity-summary';
    summary.innerHTML = `
            <div class="activity-stat">
                <div class="activity-stat-label">활성 시간</div>
                <div class="activity-stat-value" data-active>—</div>
            </div>
            <div class="activity-stat">
                <div class="activity-stat-label">유휴 시간</div>
                <div class="activity-stat-value" data-idle>—</div>
            </div>
        `;
    root.appendChild(summary);

    const listWrap = document.createElement('div');
    listWrap.className = 'activity-list';
    root.appendChild(listWrap);

    container.appendChild(root);

    const activeEl = summary.querySelector('[data-active]') as HTMLElement;
    const idleEl = summary.querySelector('[data-idle]') as HTMLElement;

    const render = (data: DayActivity): void => {
      const { apps, activeSecs, idleSecs } = aggregate(data.samples);
      activeEl.textContent = activeSecs > 0 ? formatDuration(activeSecs) : '—';
      idleEl.textContent = idleSecs > 0 ? formatDuration(idleSecs) : '—';

      listWrap.innerHTML = '';
      if (apps.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'activity-empty';
        empty.textContent = data.samples.length === 0
          ? '이 날짜의 샘플이 없습니다. 앱을 켜둔 시간대를 선택하세요.'
          : '활성 샘플이 없습니다 (전부 idle).';
        listWrap.appendChild(empty);
        return;
      }

      const max = apps[0].seconds;
      for (const app of apps) {
        const row = document.createElement('div');
        row.className = 'activity-row';
        const bar = document.createElement('div');
        bar.className = 'activity-row-bar';
        bar.style.width = `${(app.seconds / max) * 100}%`;
        row.appendChild(bar);

        const content = document.createElement('div');
        content.className = 'activity-row-content';
        content.innerHTML = `
                    <span class="activity-row-name">${escapeHtml(app.process)}</span>
                    <span class="activity-row-time">${escapeHtml(formatDuration(app.seconds))}</span>
                `;
        row.appendChild(content);

        const titlesEl = document.createElement('div');
        titlesEl.className = 'activity-titles';
        const titles = Array.from(app.titles.entries()).sort((a, b) => b[1] - a[1]).slice(0, 10);
        for (const [name, secs] of titles) {
          const t = document.createElement('div');
          t.className = 'activity-title-row';
          t.innerHTML = `
                        <span class="name">${escapeHtml(name)}</span>
                        <span>${escapeHtml(formatDuration(secs))}</span>
                    `;
          titlesEl.appendChild(t);
        }
        row.appendChild(titlesEl);

        row.style.cursor = 'pointer';
        row.addEventListener('click', () => row.classList.toggle('open'));
        listWrap.appendChild(row);
      }
    };

    const load = (): void => {
      const day = dateIn.value || todayUtcDay();
      activeEl.textContent = '…';
      idleEl.textContent = '…';
      listWrap.innerHTML = '';
      void desktopInvoke('activity_query_day', { day })
        .then(function (res: unknown) {
          render(res as DayActivity);
        })
        .catch(function (err: unknown) {
          listWrap.innerHTML = '';
          const errMsg = err instanceof Error ? err.message : String(err);
          const empty = document.createElement('div');
          empty.className = 'activity-empty';
          empty.textContent = `조회 실패: ${errMsg}`;
          listWrap.appendChild(empty);
          activeEl.textContent = '—';
          idleEl.textContent = '—';
        });
    };

    refreshBtn.addEventListener('click', load);
    dateIn.addEventListener('change', load);
    load();
  }

  Toolbox.register({
    id: 'activity',
    title: '활동 기록',
    category: 'desktop',
    desc: '내 PC에서 어떤 앱·창에 시간을 얼마나 썼는지 (데스크톱 앱 전용)',
    layout: 'form',
    icon: '<rect x="3" y="3" width="18" height="18" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M7 16v-3M11 16v-7M15 16v-5M19 16v-9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
    tabs: [{ id: 'activity-main', label: '오늘', build }]
  });
})();
