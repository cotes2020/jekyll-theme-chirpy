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
  const KST_OFFSET_SECS = 9 * 3600;

  /// 자주 쓰는 Windows 프로세스명을 사람 친화적인 라벨로 매핑.
  /// 매치 안 되는 건 원본 그대로 표시.
  const PROCESS_LABELS: Record<string, string> = {
    'Code.exe': 'VS Code',
    'Cursor.exe': 'Cursor',
    'devenv.exe': 'Visual Studio',
    'Unity.exe': 'Unity Editor',
    'Unity Hub.exe': 'Unity Hub',
    'idea64.exe': 'IntelliJ IDEA',
    'pycharm64.exe': 'PyCharm',
    'rider64.exe': 'Rider',
    'WindowsTerminal.exe': '터미널',
    'pwsh.exe': 'PowerShell',
    'powershell.exe': 'PowerShell',
    'cmd.exe': '명령 프롬프트',
    'mintty.exe': 'mintty',
    'chrome.exe': 'Chrome',
    'msedge.exe': 'Edge',
    'firefox.exe': 'Firefox',
    'whale.exe': 'Whale',
    'explorer.exe': '탐색기',
    'notepad.exe': '메모장',
    'Discord.exe': 'Discord',
    'Slack.exe': 'Slack',
    'KakaoTalk.exe': '카카오톡',
    'Telegram.exe': 'Telegram',
    'Notion.exe': 'Notion',
    'obsidian.exe': 'Obsidian',
    'Photoshop.exe': 'Photoshop',
    'Illustrator.exe': 'Illustrator',
    'Figma.exe': 'Figma',
    'Spotify.exe': 'Spotify',
    'foobar2000.exe': 'foobar2000',
    'Steam.exe': 'Steam',
    'EpicGamesLauncher.exe': 'Epic Games',
    'GitHubDesktop.exe': 'GitHub Desktop',
    'msedgewebview2.exe': 'WebView2',
    'karmolab-desktop.exe': 'KarmoLab'
  };

  /// 매핑 테이블을 lowercase 키로 미리 정규화 — Windows가 GetModuleBaseName으로 반환하는
  /// 실행파일명 케이스가 OS·드라이버에 따라 들쑥날쑥(예: explorer.exe vs Explorer.EXE).
  const PROCESS_LABELS_LOWER: Record<string, string> = (() => {
    const out: Record<string, string> = {};
    for (const k of Object.keys(PROCESS_LABELS)) out[k.toLowerCase()] = PROCESS_LABELS[k];
    return out;
  })();

  function readableProcessName(raw: string): string {
    if (!raw) return '(unknown)';
    return PROCESS_LABELS_LOWER[raw.toLowerCase()] || raw;
  }

  function desktopInvoke(cmd: string, args: unknown): Promise<unknown> {
    const core = window.__TAURI__?.core;
    const fn = core && typeof core.invoke === 'function' ? core.invoke : null;
    if (!fn) return Promise.reject(new Error('Tauri invoke 없음 (웹 브라우저 또는 withGlobalTauri 비활성)'));
    return fn(cmd, args);
  }

  function todayKstDay(): string {
    // KST 현재 일자. 사용자의 로컬 timezone과 무관하게 KST(UTC+9) 일자를 반환.
    // UTC epoch에 +9h를 더해 그 시각을 'UTC인 척' 읽으면 그게 KST 시계.
    const dt = new Date(Date.now() + KST_OFFSET_SECS * 1000);
    const y = dt.getUTCFullYear();
    const m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    const day = String(dt.getUTCDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  }

  /// KST 일자(YYYY-MM-DD) → 그 KST 일자가 시작하는 Unix epoch (UTC).
  /// KST 자정 = UTC 자정 - 9h 이므로 epoch = Date.UTC(Y,M-1,D) - 9h.
  function kstDayStartEpoch(kstDay: string): number {
    const [y, m, d] = kstDay.split('-').map((s) => parseInt(s, 10));
    const utcMidnight = Date.UTC(y, (m || 1) - 1, d || 1) / 1000;
    return utcMidnight - KST_OFFSET_SECS;
  }

  /// 같은 일자의 epoch 시각이 속하는 UTC 일자 문자열 (YYYY-MM-DD).
  function epochToUtcDay(epoch: number): string {
    const d = new Date(epoch * 1000);
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
      // 같은 exe가 케이스 다르게 들어와도 한 줄로 합치기 위해 소문자 키 사용.
      const rawName = s.process || '(unknown)';
      const key = rawName.toLowerCase();
      let agg = map.get(key);
      if (!agg) {
        agg = { process: rawName, seconds: 0, titles: new Map() };
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

  /// KST 기준 시각 epoch → hour (0~23) + dow (0=월, 6=일).
  function kstHourAndDowMon0(epochSec: number): { hour: number; dowMon0: number } {
    const dt = new Date((epochSec + KST_OFFSET_SECS) * 1000);
    const hour = dt.getUTCHours();
    const utcDow = dt.getUTCDay();
    const dowMon0 = utcDow === 0 ? 6 : utcDow - 1;
    return { hour, dowMon0 };
  }

  /// epoch 시각이 속하는 KST 일자 문자열 (YYYY-MM-DD).
  function epochToKstDay(epoch: number): string {
    const dt = new Date((epoch + KST_OFFSET_SECS) * 1000);
    const y = dt.getUTCFullYear();
    const m = String(dt.getUTCMonth() + 1).padStart(2, '0');
    const d = String(dt.getUTCDate()).padStart(2, '0');
    return `${y}-${m}-${d}`;
  }

  /// 일자별 활성 시간 Map (key=YYYY-MM-DD, value=초). idle 제외.
  function buildDailyActiveMap(samples: ActivitySample[]): Map<string, number> {
    const map = new Map<string, number>();
    for (const s of samples) {
      if (s.idle) continue;
      const day = epochToKstDay(s.ts);
      map.set(day, (map.get(day) || 0) + SAMPLE_INTERVAL_SECS);
    }
    return map;
  }

  /// active 샘플을 7행(월~일) × 24열(0시~23시) 매트릭스로 누적 (단위: 초).
  /// idle 샘플은 제외 — "내가 실제로 활동한 시간대" 패턴 시각화가 목적.
  function buildHeatmapMatrix(samples: ActivitySample[]): {
    matrix: number[][];
    max: number;
  } {
    const matrix: number[][] = Array.from({ length: 7 }, () => new Array<number>(24).fill(0));
    let max = 0;
    for (const s of samples) {
      if (s.idle) continue;
      const { hour, dowMon0 } = kstHourAndDowMon0(s.ts);
      const v = (matrix[dowMon0][hour] += SAMPLE_INTERVAL_SECS);
      if (v > max) max = v;
    }
    return { matrix, max };
  }

  /// CSV escape: comma/quote/newline 있으면 따옴표 감싸고 내부 따옴표는 두 번.
  function csvEscape(s: string): string {
    const v = String(s ?? '');
    if (/[",\r\n]/.test(v)) return '"' + v.replace(/"/g, '""') + '"';
    return v;
  }

  /// raw 샘플 → CSV. UTF-8 BOM 포함 (Excel 한국어 깨짐 방지).
  function buildSamplesCsv(samples: ActivitySample[]): string {
    const lines = ['ts_iso,ts_epoch,process,label,title,idle'];
    for (const s of samples) {
      const iso = new Date(s.ts * 1000).toISOString();
      lines.push([
        csvEscape(iso),
        String(s.ts),
        csvEscape(s.process || ''),
        csvEscape(readableProcessName(s.process || '')),
        csvEscape(s.title || ''),
        s.idle ? '1' : '0',
      ].join(','));
    }
    return '﻿' + lines.join('\n');
  }

  /// 집계 결과 → CSV. 앱별 누적 시간 + top 윈도우 타이틀 1개.
  function buildAggregateCsv(apps: AppAggregate[], activeSecs: number, idleSecs: number): string {
    const lines = ['process,label,seconds,share_pct,top_title,top_seconds'];
    for (const a of apps) {
      let topTitle = '';
      let topSecs = 0;
      for (const [t, secs] of a.titles.entries()) {
        if (secs > topSecs) {
          topTitle = t;
          topSecs = secs;
        }
      }
      const sharePct = activeSecs > 0 ? ((a.seconds / activeSecs) * 100).toFixed(2) : '0.00';
      lines.push([
        csvEscape(a.process),
        csvEscape(readableProcessName(a.process)),
        String(a.seconds),
        sharePct,
        csvEscape(topTitle),
        String(topSecs),
      ].join(','));
    }
    lines.push('');
    lines.push('# summary');
    lines.push(`# active_seconds,${activeSecs}`);
    lines.push(`# idle_seconds,${idleSecs}`);
    return '﻿' + lines.join('\n');
  }

  /// CSV 텍스트를 파일로 저장 (Blob + a[download]). Tauri 권한 추가 X.
  function downloadCsv(filename: string, csv: string): void {
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
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
            .activity-nav-btn { min-width: 32px; padding: 6px 8px; }
            .activity-period-select {
                padding: 6px 8px; border-radius: var(--radius-md);
                border: 1px solid var(--border); background: var(--bg-primary);
                color: var(--text-primary); font-size: var(--font-size-sm);
            }
            .activity-filter-input {
                flex: 1; min-width: 200px;
                padding: 6px 10px; border-radius: var(--radius-md);
                border: 1px solid var(--border); background: var(--bg-primary);
                color: var(--text-primary); font-size: var(--font-size-sm);
            }
            .activity-meta { font-size: var(--font-size-xs); color: var(--text-tertiary); margin-bottom: 12px; }
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
            .activity-row-subtitle { font-weight: 400; color: var(--text-tertiary); font-size: var(--font-size-xs); margin-left: 6px; }
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

            /* 시간대 히트맵 — 주 단위 모드 전용. 7행(월~일) × 24열(0~23시), accent 색 alpha 로 강도 표시. */
            .activity-heatmap-wrap { margin: 0 0 18px 0; }
            .activity-heatmap-title { font-size: var(--font-size-xs); color: var(--text-secondary); font-weight: 600; margin: 0 0 6px 0; }
            .activity-heatmap-hint { font-size: 10px; color: var(--text-tertiary); margin: 0 0 8px 0; }
            .activity-heatmap-grid {
                display: grid;
                grid-template-columns: 28px repeat(24, minmax(0, 1fr));
                gap: 2px;
                font-size: 9px;
                color: var(--text-tertiary);
            }
            .activity-heatmap-corner { padding: 2px; }
            .activity-heatmap-hour { padding: 1px 0; text-align: center; }
            .activity-heatmap-hour--major { color: var(--text-secondary); font-weight: 600; }
            .activity-heatmap-dow {
                padding: 1px 4px; text-align: right;
                line-height: 1; align-self: center;
                color: var(--text-secondary); font-weight: 600;
            }
            .activity-heatmap-cell {
                aspect-ratio: 1 / 1;
                border-radius: 2px;
                background: color-mix(in srgb, var(--accent) 0%, var(--bg-tertiary));
                cursor: default;
            }
            .activity-heatmap-cell--empty { background: var(--bg-tertiary); }

            /* 일자별 막대 — 주/월 모드 전용. flex row, 각 막대는 height = active / max. */
            .activity-bars-wrap { margin: 0 0 18px 0; }
            .activity-bars-title { font-size: var(--font-size-xs); color: var(--text-secondary); font-weight: 600; margin: 0 0 6px 0; }
            .activity-bars-hint { font-size: 10px; color: var(--text-tertiary); margin: 0 0 8px 0; }
            .activity-bars-grid {
                display: flex;
                align-items: flex-end;
                gap: 3px;
                height: 100px;
                padding: 4px 0 0;
                border-bottom: 1px solid var(--border);
            }
            .activity-bars-col {
                flex: 1 1 0;
                min-width: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-end;
                height: 100%;
                cursor: default;
            }
            .activity-bars-col-bar {
                width: 100%;
                min-height: 1px;
                background: var(--accent);
                border-radius: 2px 2px 0 0;
                opacity: 0.85;
            }
            .activity-bars-col--empty .activity-bars-col-bar {
                background: var(--bg-tertiary);
                opacity: 1;
            }
            .activity-bars-col--today .activity-bars-col-bar {
                background: color-mix(in srgb, var(--accent) 60%, var(--text-primary));
            }
            .activity-bars-labels {
                display: flex;
                gap: 3px;
                margin-top: 4px;
                font-size: 9px;
                color: var(--text-tertiary);
            }
            .activity-bars-label {
                flex: 1 1 0;
                min-width: 0;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .activity-bars-label--major {
                color: var(--text-secondary);
                font-weight: 600;
            }
        `
    );

    container.innerHTML = '';
    const root = document.createElement('div');
    root.className = 'activity-root';

    const intro = document.createElement('p');
    intro.className = 'activity-intro';
    const isApp = typeof Toolbox.isDesktopApp === 'function' && Toolbox.isDesktopApp();
    intro.innerHTML = isApp
      ? `5초마다 포그라운드 윈도우를 샘플링합니다. 데이터는 전부 로컬에만 저장됩니다 (외부 전송 0). 60초 이상 입력 없으면 idle로 분류. 일자는 KST 기준으로 보여주지만 저장은 UTC 일자별 파일이라 KST 하루가 두 UTC 파일에 걸쳐 있을 수 있습니다.`
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
    lab.textContent = '기간';
    const periodSel = document.createElement('select');
    periodSel.className = 'activity-period-select';
    ([
      ['day', '일'],
      ['week', '주'],
      ['month', '월'],
      ['all', '전체']
    ] as const).forEach(([v, t]) => {
      const o = document.createElement('option');
      o.value = v;
      o.textContent = t;
      periodSel.appendChild(o);
    });
    periodSel.value = 'day';
    const prevBtn = document.createElement('button');
    prevBtn.type = 'button';
    prevBtn.className = 'btn btn-secondary activity-nav-btn';
    prevBtn.textContent = '◀';
    prevBtn.title = '이전 구간';
    const dateIn = document.createElement('input');
    dateIn.type = 'date';
    dateIn.value = todayKstDay();
    const nextBtn = document.createElement('button');
    nextBtn.type = 'button';
    nextBtn.className = 'btn btn-secondary activity-nav-btn';
    nextBtn.textContent = '▶';
    nextBtn.title = '다음 구간';
    const todayBtn = document.createElement('button');
    todayBtn.type = 'button';
    todayBtn.className = 'btn btn-secondary';
    todayBtn.textContent = '오늘';
    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.className = 'btn btn-secondary';
    refreshBtn.textContent = '새로고침';
    const exportSamplesBtn = document.createElement('button');
    exportSamplesBtn.type = 'button';
    exportSamplesBtn.className = 'btn btn-secondary';
    exportSamplesBtn.textContent = 'CSV 샘플';
    exportSamplesBtn.title = '현재 범위의 raw 샘플을 CSV 로 내보냅니다 (UTF-8 BOM 포함, Excel 호환)';
    const exportAggBtn = document.createElement('button');
    exportAggBtn.type = 'button';
    exportAggBtn.className = 'btn btn-secondary';
    exportAggBtn.textContent = 'CSV 집계';
    exportAggBtn.title = '앱별 누적 시간 집계 결과를 CSV 로 내보냅니다';
    controls.appendChild(lab);
    controls.appendChild(periodSel);
    controls.appendChild(prevBtn);
    controls.appendChild(dateIn);
    controls.appendChild(nextBtn);
    controls.appendChild(todayBtn);
    controls.appendChild(refreshBtn);
    controls.appendChild(exportSamplesBtn);
    controls.appendChild(exportAggBtn);
    root.appendChild(controls);

    const filterRow = document.createElement('div');
    filterRow.className = 'activity-controls';
    const filterIn = document.createElement('input');
    filterIn.type = 'search';
    filterIn.placeholder = '앱 이름 / 윈도우 타이틀 검색…';
    filterIn.className = 'activity-filter-input';
    filterRow.appendChild(filterIn);
    root.appendChild(filterRow);

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

    const meta = document.createElement('div');
    meta.className = 'activity-meta';
    meta.textContent = '';
    root.appendChild(meta);

    const heatmapWrap = document.createElement('div');
    heatmapWrap.className = 'activity-heatmap-wrap';
    heatmapWrap.style.display = 'none';
    root.appendChild(heatmapWrap);

    const barsWrap = document.createElement('div');
    barsWrap.className = 'activity-bars-wrap';
    barsWrap.style.display = 'none';
    root.appendChild(barsWrap);

    const listWrap = document.createElement('div');
    listWrap.className = 'activity-list';
    root.appendChild(listWrap);

    container.appendChild(root);

    const activeEl = summary.querySelector('[data-active]') as HTMLElement;
    const idleEl = summary.querySelector('[data-idle]') as HTMLElement;

    type Period = 'day' | 'week' | 'month' | 'all';
    let lastSamples: ActivitySample[] = [];
    let currentFilter = '';
    let refreshTimer: ReturnType<typeof setInterval> | null = null;
    const REFRESH_INTERVAL_MS = 30_000;

    function shiftKstDay(kstDay: string, delta: number): string {
      const [y, m, d] = kstDay.split('-').map((s) => parseInt(s, 10));
      const dt = new Date(Date.UTC(y, (m || 1) - 1, d || 1));
      dt.setUTCDate(dt.getUTCDate() + delta);
      const yy = dt.getUTCFullYear();
      const mm = String(dt.getUTCMonth() + 1).padStart(2, '0');
      const dd = String(dt.getUTCDate()).padStart(2, '0');
      return `${yy}-${mm}-${dd}`;
    }

    function shiftKstMonth(kstDay: string, delta: number): string {
      const [y, m, d] = kstDay.split('-').map((s) => parseInt(s, 10));
      const dt = new Date(Date.UTC(y, (m || 1) - 1, d || 1));
      dt.setUTCMonth(dt.getUTCMonth() + delta);
      const yy = dt.getUTCFullYear();
      const mm = String(dt.getUTCMonth() + 1).padStart(2, '0');
      const dd = String(dt.getUTCDate()).padStart(2, '0');
      return `${yy}-${mm}-${dd}`;
    }

    /// 주의 시작(월요일)을 KST 일자 문자열로 반환.
    function weekStartKst(kstDay: string): string {
      const [y, m, d] = kstDay.split('-').map((s) => parseInt(s, 10));
      const dt = new Date(Date.UTC(y, (m || 1) - 1, d || 1));
      // getUTCDay: 0=일, 1=월, ..., 6=토. 월요일을 주의 시작으로.
      const dow = dt.getUTCDay();
      const offsetToMonday = dow === 0 ? -6 : 1 - dow;
      dt.setUTCDate(dt.getUTCDate() + offsetToMonday);
      const yy = dt.getUTCFullYear();
      const mm = String(dt.getUTCMonth() + 1).padStart(2, '0');
      const dd = String(dt.getUTCDate()).padStart(2, '0');
      return `${yy}-${mm}-${dd}`;
    }

    /// 월의 1일을 KST 일자 문자열로.
    function monthStartKst(kstDay: string): string {
      const [y, m] = kstDay.split('-').map((s) => parseInt(s, 10));
      return `${y}-${String(m).padStart(2, '0')}-01`;
    }

    /// period + anchor 일자 → KST 자정 epoch 시작/끝 윈도우.
    /// 'all'의 경우 호출 측에서 활동 디렉토리 listing을 받아 처리해야 함 (이 함수는 day/week/month만).
    function periodRange(period: Period, anchorKstDay: string): { startEpoch: number; endEpoch: number; label: string } {
      if (period === 'day') {
        const start = kstDayStartEpoch(anchorKstDay);
        return { startEpoch: start, endEpoch: start + 86400, label: anchorKstDay };
      }
      if (period === 'week') {
        const monday = weekStartKst(anchorKstDay);
        const start = kstDayStartEpoch(monday);
        const sunday = shiftKstDay(monday, 6);
        return { startEpoch: start, endEpoch: start + 7 * 86400, label: `${monday} ~ ${sunday} (주)` };
      }
      if (period === 'month') {
        const first = monthStartKst(anchorKstDay);
        const nextFirst = shiftKstMonth(first, 1);
        return {
          startEpoch: kstDayStartEpoch(first),
          endEpoch: kstDayStartEpoch(nextFirst),
          label: `${first.slice(0, 7)} (월)`
        };
      }
      // 'all' — placeholder; 실제 범위는 listDays 결과 기반으로 호출 측에서 결정
      return { startEpoch: 0, endEpoch: Number.MAX_SAFE_INTEGER, label: '전체' };
    }

    /// startEpoch~endEpoch 범위에 걸치는 UTC 일자 목록.
    function utcDaysInRange(startEpoch: number, endEpoch: number): string[] {
      const days: string[] = [];
      const oneDay = 86400;
      // 시작/끝을 포함하도록 끝-1초까지 순회
      let cur = Math.floor(startEpoch / oneDay) * oneDay;
      const last = Math.floor((endEpoch - 1) / oneDay) * oneDay;
      while (cur <= last) {
        days.push(epochToUtcDay(cur));
        cur += oneDay;
      }
      return days;
    }

    function formatNow(): string {
      const now = new Date();
      const hh = String(now.getHours()).padStart(2, '0');
      const mm = String(now.getMinutes()).padStart(2, '0');
      const ss = String(now.getSeconds()).padStart(2, '0');
      return `${hh}:${mm}:${ss}`;
    }

    const DOW_LABELS = ['월', '화', '수', '목', '금', '토', '일'];

    function renderHeatmap(samples: ActivitySample[]): void {
      // 주 단위 모드 전용. 다른 period 에선 숨김 (KL-002 sub 명세).
      if ((periodSel.value as Period) !== 'week') {
        heatmapWrap.style.display = 'none';
        heatmapWrap.innerHTML = '';
        return;
      }
      const { matrix, max } = buildHeatmapMatrix(samples);
      if (max === 0) {
        heatmapWrap.style.display = 'none';
        heatmapWrap.innerHTML = '';
        return;
      }
      heatmapWrap.style.display = 'block';
      const parts: string[] = [];
      parts.push('<div class="activity-heatmap-title">시간대 히트맵 (KST)</div>');
      parts.push('<div class="activity-heatmap-hint">색이 진할수록 그 시간대에 더 많이 활동 · idle 제외</div>');
      parts.push('<div class="activity-heatmap-grid">');
      parts.push('<div class="activity-heatmap-corner"></div>');
      for (let h = 0; h < 24; h++) {
        const isMajor = h % 6 === 0;
        parts.push(
          `<div class="activity-heatmap-hour${isMajor ? ' activity-heatmap-hour--major' : ''}">${isMajor ? h : '·'}</div>`
        );
      }
      for (let d = 0; d < 7; d++) {
        parts.push(`<div class="activity-heatmap-dow">${DOW_LABELS[d]}</div>`);
        for (let h = 0; h < 24; h++) {
          const v = matrix[d][h];
          if (v === 0) {
            parts.push('<div class="activity-heatmap-cell activity-heatmap-cell--empty"></div>');
            continue;
          }
          // 0 보다 큰 값은 최소 18% 강도부터 — 안 그러면 1~2 샘플 칸이 안 보임.
          const ratio = v / max;
          const pct = Math.round(18 + ratio * 82);
          const mins = Math.round(v / 60);
          const tooltip = `${DOW_LABELS[d]} ${h}시 — ${mins}분`;
          parts.push(
            `<div class="activity-heatmap-cell" style="background:color-mix(in srgb, var(--accent) ${pct}%, var(--bg-tertiary));" title="${escapeHtml(tooltip)}"></div>`
          );
        }
      }
      parts.push('</div>');
      heatmapWrap.innerHTML = parts.join('');
    }

    /// 주/월 모드의 anchor 가 가리키는 일자 키 시퀀스 (월요일부터 / 1일부터).
    function dailyKeysForPeriod(): string[] {
      const period = periodSel.value as Period;
      const anchor = dateIn.value || todayKstDay();
      if (period === 'week') {
        const monday = weekStartKst(anchor);
        const out: string[] = [];
        for (let i = 0; i < 7; i++) out.push(shiftKstDay(monday, i));
        return out;
      }
      if (period === 'month') {
        const first = monthStartKst(anchor);
        const nextFirst = shiftKstMonth(first, 1);
        const out: string[] = [];
        let cur = first;
        while (cur < nextFirst) {
          out.push(cur);
          cur = shiftKstDay(cur, 1);
        }
        return out;
      }
      return [];
    }

    function renderDailyBars(samples: ActivitySample[]): void {
      const keys = dailyKeysForPeriod();
      if (keys.length === 0) {
        barsWrap.style.display = 'none';
        barsWrap.innerHTML = '';
        return;
      }
      const dailyMap = buildDailyActiveMap(samples);
      const values = keys.map((k) => dailyMap.get(k) || 0);
      const max = Math.max(...values, 0);
      if (max === 0) {
        barsWrap.style.display = 'none';
        barsWrap.innerHTML = '';
        return;
      }
      barsWrap.style.display = 'block';
      const today = todayKstDay();
      const period = periodSel.value as Period;
      const isWeek = period === 'week';
      const parts: string[] = [];
      parts.push(`<div class="activity-bars-title">일자별 활성 시간 (${isWeek ? '주' : '월'})</div>`);
      parts.push('<div class="activity-bars-hint">막대 높이 = 그 날의 활성 시간 (idle 제외) · 오늘 칸은 강조</div>');
      parts.push('<div class="activity-bars-grid">');
      for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        const v = values[i];
        const heightPct = max > 0 ? Math.max(1, Math.round((v / max) * 100)) : 0;
        const colCls = ['activity-bars-col'];
        if (v === 0) colCls.push('activity-bars-col--empty');
        if (k === today) colCls.push('activity-bars-col--today');
        const tooltip = v > 0 ? `${k} — ${formatDuration(v)}` : `${k} — 활동 없음`;
        parts.push(
          `<div class="${colCls.join(' ')}" title="${escapeHtml(tooltip)}"><div class="activity-bars-col-bar" style="height:${heightPct}%"></div></div>`
        );
      }
      parts.push('</div>');
      // 라벨 — 주: 월~일, 월: 1/8/15/22/말일 만 major (가독성).
      parts.push('<div class="activity-bars-labels">');
      for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        const dom = parseInt(k.slice(8, 10), 10);
        let text: string;
        let major = false;
        if (isWeek) {
          text = DOW_LABELS[i] || '';
          major = true;
        } else {
          const isMajor = dom === 1 || dom % 7 === 0 || i === keys.length - 1;
          text = isMajor ? String(dom) : '·';
          major = isMajor;
        }
        const cls = 'activity-bars-label' + (major ? ' activity-bars-label--major' : '');
        parts.push(`<div class="${cls}">${escapeHtml(text)}</div>`);
      }
      parts.push('</div>');
      barsWrap.innerHTML = parts.join('');
    }

    const render = (data: DayActivity): void => {
      const { apps, activeSecs, idleSecs } = aggregate(data.samples);
      activeEl.textContent = activeSecs > 0 ? formatDuration(activeSecs) : '—';
      idleEl.textContent = idleSecs > 0 ? formatDuration(idleSecs) : '—';

      const totalSamples = data.samples.length;
      const total = activeSecs + idleSecs;
      const idlePct = total > 0 ? Math.round((idleSecs / total) * 100) : 0;
      meta.textContent = `${data.day} · 샘플 ${totalSamples}건 · idle ${idlePct}% · 마지막 갱신 ${formatNow()}`;

      renderHeatmap(data.samples);
      renderDailyBars(data.samples);

      const filterTerm = currentFilter.trim().toLowerCase();
      const filtered = filterTerm
        ? apps.filter((a) => {
            const labeled = readableProcessName(a.process).toLowerCase();
            if (a.process.toLowerCase().includes(filterTerm) || labeled.includes(filterTerm)) return true;
            for (const t of a.titles.keys()) {
              if (t.toLowerCase().includes(filterTerm)) return true;
            }
            return false;
          })
        : apps;

      listWrap.innerHTML = '';
      if (filtered.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'activity-empty';
        if (totalSamples === 0) {
          empty.textContent = '이 날짜의 샘플이 없습니다. 앱이 시작된 후 5초 이상 기다리거나, 다른 날짜를 선택하세요.';
        } else if (apps.length === 0) {
          empty.textContent = '활성 샘플이 없습니다 (전부 idle 분류).';
        } else {
          empty.textContent = `"${currentFilter}"에 매칭되는 앱·창이 없습니다.`;
        }
        listWrap.appendChild(empty);
        return;
      }

      const max = filtered[0].seconds;
      for (const app of filtered) {
        const row = document.createElement('div');
        row.className = 'activity-row';
        const bar = document.createElement('div');
        bar.className = 'activity-row-bar';
        bar.style.width = `${(app.seconds / max) * 100}%`;
        row.appendChild(bar);

        const content = document.createElement('div');
        content.className = 'activity-row-content';
        const labeled = readableProcessName(app.process);
        const subtitle = labeled !== app.process ? ` <span class="activity-row-subtitle">${escapeHtml(app.process)}</span>` : '';
        content.innerHTML = `
                    <span class="activity-row-name">${escapeHtml(labeled)}${subtitle}</span>
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

    /// 현재 period에 맞는 epoch 윈도우 + 해당 윈도우에 걸치는 UTC 일자 목록.
    /// 'all' 모드는 activity_list_days 호출이 필요해 비동기.
    async function resolveRangeAndDays(): Promise<{ startEpoch: number; endEpoch: number; utcDays: string[]; label: string }> {
      const period = (periodSel.value || 'day') as Period;
      const anchor = dateIn.value || todayKstDay();
      if (period === 'all') {
        const days = (await desktopInvoke('activity_list_days', {}) as string[]) || [];
        if (days.length === 0) {
          return { startEpoch: 0, endEpoch: 0, utcDays: [], label: '전체 (데이터 없음)' };
        }
        const sorted = [...days].sort();
        const first = sorted[0];
        const last = sorted[sorted.length - 1];
        // UTC 일자 파일이라 첫 파일 자정~마지막 파일 자정+1일 epoch.
        const [fy, fm, fd] = first.split('-').map((s) => parseInt(s, 10));
        const [ly, lm, ld] = last.split('-').map((s) => parseInt(s, 10));
        const startEpoch = Date.UTC(fy, fm - 1, fd) / 1000;
        const endEpoch = Date.UTC(ly, lm - 1, ld) / 1000 + 86400;
        return { startEpoch, endEpoch, utcDays: sorted, label: `전체 (${first} ~ ${last}, ${sorted.length}일)` };
      }
      const { startEpoch, endEpoch, label } = periodRange(period, anchor);
      const utcDays = utcDaysInRange(startEpoch, endEpoch);
      return { startEpoch, endEpoch, utcDays, label };
    }

    const load = (silent = false): void => {
      if (!silent) {
        activeEl.textContent = '…';
        idleEl.textContent = '…';
        listWrap.innerHTML = '';
      }

      void resolveRangeAndDays()
        .then(({ startEpoch, endEpoch, utcDays, label }) => {
          if (utcDays.length === 0) {
            lastSamples = [];
            render({ day: label, samples: [] });
            return;
          }
          return Promise.all(
            utcDays.map((day) => desktopInvoke('activity_query_day', { day }) as Promise<DayActivity>)
          ).then((results) => {
            const merged: ActivitySample[] = [];
            for (const r of results) {
              for (const s of r.samples || []) {
                if (s.ts >= startEpoch && s.ts < endEpoch) merged.push(s);
              }
            }
            lastSamples = merged;
            render({ day: label, samples: merged });
          });
        })
        .catch((err: unknown) => {
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

    function rerenderFromCache(): void {
      const kstDay = dateIn.value || todayKstDay();
      render({ day: kstDay, samples: lastSamples });
    }

    /// 현재 period 모드에서 prev/next가 의미 있고, anchor가 오늘 구간에 포함되면 자동 새로고침을 켠다.
    function isViewingCurrentPeriod(): boolean {
      const period = periodSel.value as Period;
      const anchor = dateIn.value || todayKstDay();
      const today = todayKstDay();
      if (period === 'all') return true; // 전체 보면 항상 최신 포함
      if (period === 'day') return anchor === today;
      if (period === 'week') return weekStartKst(anchor) === weekStartKst(today);
      if (period === 'month') return monthStartKst(anchor) === monthStartKst(today);
      return false;
    }

    function setupAutoRefresh(): void {
      if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
      }
      if (!isViewingCurrentPeriod()) return;
      refreshTimer = setInterval(() => {
        if (!document.body.contains(root)) {
          if (refreshTimer) clearInterval(refreshTimer);
          refreshTimer = null;
          return;
        }
        if (isViewingCurrentPeriod()) load(true);
      }, REFRESH_INTERVAL_MS);
    }

    function applyPeriodControlsState(): void {
      const period = periodSel.value as Period;
      const isAll = period === 'all';
      dateIn.disabled = isAll;
      prevBtn.disabled = isAll;
      nextBtn.disabled = isAll;
      todayBtn.disabled = isAll;
      // prev/next title도 모드에 따라 갱신.
      if (period === 'day') {
        prevBtn.title = '하루 전';
        nextBtn.title = '하루 후';
      } else if (period === 'week') {
        prevBtn.title = '한 주 전';
        nextBtn.title = '한 주 후';
      } else if (period === 'month') {
        prevBtn.title = '한 달 전';
        nextBtn.title = '한 달 후';
      }
    }

    function shiftAnchorByPeriod(delta: number): void {
      const period = periodSel.value as Period;
      const anchor = dateIn.value || todayKstDay();
      if (period === 'day') dateIn.value = shiftKstDay(anchor, delta);
      else if (period === 'week') dateIn.value = shiftKstDay(anchor, delta * 7);
      else if (period === 'month') dateIn.value = shiftKstMonth(anchor, delta);
    }

    function exportFilenameSlug(): string {
      const period = periodSel.value as Period;
      const anchor = dateIn.value || todayKstDay();
      if (period === 'day') return anchor;
      if (period === 'week') return `week-${weekStartKst(anchor)}`;
      if (period === 'month') return anchor.slice(0, 7);
      return 'all';
    }

    function exportSamples(): void {
      if (lastSamples.length === 0) {
        Toolbox.showToast?.('내보낼 샘플이 없습니다.', 'error', undefined);
        return;
      }
      const csv = buildSamplesCsv(lastSamples);
      downloadCsv(`activity-${exportFilenameSlug()}-samples.csv`, csv);
      Toolbox.showToast?.(`샘플 ${lastSamples.length}건 CSV 다운로드`, 'success', undefined);
    }

    function exportAggregate(): void {
      if (lastSamples.length === 0) {
        Toolbox.showToast?.('내보낼 데이터가 없습니다.', 'error', undefined);
        return;
      }
      const { apps, activeSecs, idleSecs } = aggregate(lastSamples);
      const csv = buildAggregateCsv(apps, activeSecs, idleSecs);
      downloadCsv(`activity-${exportFilenameSlug()}-aggregate.csv`, csv);
      Toolbox.showToast?.(`앱 ${apps.length}개 집계 CSV 다운로드`, 'success', undefined);
    }

    exportSamplesBtn.addEventListener('click', exportSamples);
    exportAggBtn.addEventListener('click', exportAggregate);
    refreshBtn.addEventListener('click', () => load());
    dateIn.addEventListener('change', () => { load(); setupAutoRefresh(); });
    periodSel.addEventListener('change', () => { applyPeriodControlsState(); load(); setupAutoRefresh(); });
    prevBtn.addEventListener('click', () => {
      shiftAnchorByPeriod(-1);
      load();
      setupAutoRefresh();
    });
    nextBtn.addEventListener('click', () => {
      shiftAnchorByPeriod(1);
      load();
      setupAutoRefresh();
    });
    todayBtn.addEventListener('click', () => {
      dateIn.value = todayKstDay();
      load();
      setupAutoRefresh();
    });
    filterIn.addEventListener('input', () => {
      currentFilter = filterIn.value || '';
      rerenderFromCache();
    });

    applyPeriodControlsState();
    load();
    setupAutoRefresh();
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
