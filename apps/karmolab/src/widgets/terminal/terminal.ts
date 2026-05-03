/**
 * terminal — Tauri 데스크톱 전용 (category: 'desktop' 자동 게이팅).
 *
 * KarmoLab 안 단일 PowerShell IO 셸 (TASK-KL-006 옵션 A).
 * stdin form + 출력 패널 + cwd indicator + start/stop. 5000 라인 cap.
 *
 * Rust 명령: terminal_start / terminal_send_stdin / terminal_stop / terminal_status
 * 이벤트: karmolab://terminal-line / karmolab://terminal-cwd / karmolab://terminal-exit
 */
// @ts-nocheck — Toolbox / window.__TAURI__ 글로벌은 ambient 타입에 다 안 잡혀 있음.
(function (): void {
  interface TerminalStartResult {
    running: boolean;
    cwd: string;
    shell: string;
    alreadyRunning: boolean;
  }
  interface TerminalLineEvt { stream: string; line: string; }
  interface TerminalCwdEvt { cwd: string; }
  interface TerminalExitEvt { code: number | null; }

  const MAX_OUTPUT_LINES = 5000;

  function isKarmolabDesktop(): boolean {
    return typeof window !== 'undefined' && !!window.__KARMOLAB_DESKTOP__;
  }
  function getInvoke(): ((cmd: string, args?: any) => Promise<any>) | null {
    const i = window.__TAURI__?.core?.invoke;
    return typeof i === 'function' ? i : null;
  }
  function getListen(): ((evt: string, cb: (e: any) => void) => Promise<() => void>) | null {
    const l = window.__TAURI__?.event?.listen;
    return typeof l === 'function' ? l : null;
  }

  Toolbox.register({
    ...Toolbox.getLazyWidgetPublicMeta('terminal'),
    tabs: [
      {
        id: 'app',
        label: 'shell',
        build(container: HTMLElement): void {
          injectStyles();
          renderShell(container);
          if (isKarmolabDesktop()) void boot(container);
        },
      },
    ],
  });

  function injectStyles(): void {
    if (document.getElementById('kt-term-styles')) return;
    const style = document.createElement('style');
    style.id = 'kt-term-styles';
    style.textContent = `
      .kt-term { display: flex; flex-direction: column; height: 100%; min-height: 480px; padding: 12px; gap: 8px; color: var(--text-primary, #e8e8e8); }
      .kt-term .kt-bar { display: flex; align-items: center; gap: 8px; font-size: 12px; flex-wrap: wrap; }
      .kt-term .kt-bar .kt-status-dot { width: 8px; height: 8px; border-radius: 50%; background: #555; flex-shrink: 0; }
      .kt-term .kt-bar.kt-running .kt-status-dot { background: var(--success, #22c55e); }
      .kt-term .kt-bar .kt-shell { color: var(--text-tertiary, #888); font-family: var(--font-mono, "JetBrains Mono", monospace); }
      .kt-term .kt-bar .kt-cwd { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--accent, #d4a849); font-family: var(--font-mono, "JetBrains Mono", monospace); }
      .kt-term .kt-bar button { background: var(--bg-secondary, #2a2a2a); color: inherit; border: 1px solid var(--border, #444); border-radius: 3px; padding: 4px 10px; cursor: pointer; font-size: 12px; }
      .kt-term .kt-bar button:hover { background: var(--bg-tertiary, #333); }
      .kt-term .kt-bar button:disabled { opacity: 0.5; cursor: not-allowed; }
      .kt-term .kt-bar .kt-btn-stop { color: #c06060; }
      .kt-term .kt-out { flex: 1; overflow-y: auto; background: #0d0d0d; border: 1px solid var(--border, #333); border-radius: 3px; padding: 8px 10px; font-family: var(--font-mono, "JetBrains Mono", monospace); font-size: 12px; line-height: 1.45; white-space: pre-wrap; word-break: break-word; min-height: 240px; }
      .kt-term .kt-out .kt-line { color: #d8d8d8; }
      .kt-term .kt-out .kt-line.kt-stderr { color: #d4a060; }
      .kt-term .kt-out .kt-line.kt-input { color: #6da6e0; }
      .kt-term .kt-out .kt-line.kt-meta { color: #888; font-style: italic; }
      .kt-term .kt-form { display: flex; gap: 6px; }
      .kt-term .kt-form input { flex: 1; background: #0d0d0d; color: var(--text-primary, #e8e8e8); border: 1px solid var(--border, #333); border-radius: 3px; padding: 6px 10px; font-family: var(--font-mono, "JetBrains Mono", monospace); font-size: 12.5px; }
      .kt-term .kt-form input:disabled { opacity: 0.5; }
      .kt-term .kt-form input:focus { outline: none; border-color: var(--accent, #d4a849); }
      .kt-term .kt-form button { background: var(--accent, #d4a849); color: #000; border: none; border-radius: 3px; padding: 6px 14px; cursor: pointer; font-size: 12px; font-weight: 600; }
      .kt-term .kt-form button:disabled { opacity: 0.4; cursor: not-allowed; }
      .kt-term .kt-disabled { padding: 28px; text-align: center; color: var(--text-tertiary, #888); font-size: 13px; }
    `;
    document.head.appendChild(style);
  }

  function renderShell(container: HTMLElement): void {
    if (!isKarmolabDesktop()) {
      container.innerHTML = `<div class="kt-term"><div class="kt-disabled">terminal 위젯은 Tauri 데스크톱 앱 전용입니다.</div></div>`;
      return;
    }
    container.innerHTML = `
      <div class="kt-term">
        <div class="kt-bar" data-kt="bar">
          <span class="kt-status-dot"></span>
          <button data-kt="btn-start" type="button">▶ 시작</button>
          <button data-kt="btn-stop" class="kt-btn-stop" type="button" disabled>⏹ 종료</button>
          <span class="kt-shell" data-kt="shell">—</span>
          <span class="kt-cwd" data-kt="cwd">cwd: —</span>
        </div>
        <div class="kt-out" data-kt="out"></div>
        <form class="kt-form" data-kt="form">
          <input data-kt="input" type="text" autocomplete="off" spellcheck="false" placeholder="명령어 입력 후 Enter (시작 후 활성)" disabled />
          <button data-kt="btn-send" type="submit" disabled>실행</button>
        </form>
      </div>
    `;
  }

  async function boot(container: HTMLElement): Promise<void> {
    const out = container.querySelector('[data-kt="out"]') as HTMLElement | null;
    const bar = container.querySelector('[data-kt="bar"]') as HTMLElement | null;
    const btnStart = container.querySelector('[data-kt="btn-start"]') as HTMLButtonElement | null;
    const btnStop = container.querySelector('[data-kt="btn-stop"]') as HTMLButtonElement | null;
    const btnSend = container.querySelector('[data-kt="btn-send"]') as HTMLButtonElement | null;
    const input = container.querySelector('[data-kt="input"]') as HTMLInputElement | null;
    const form = container.querySelector('[data-kt="form"]') as HTMLFormElement | null;
    const shellLabel = container.querySelector('[data-kt="shell"]') as HTMLElement | null;
    const cwdLabel = container.querySelector('[data-kt="cwd"]') as HTMLElement | null;
    if (!out || !bar || !btnStart || !btnStop || !btnSend || !input || !form || !shellLabel || !cwdLabel) return;

    const invoke = getInvoke();
    const listen = getListen();
    if (!invoke) {
      appendMeta(out, '[Tauri invoke 없음 — 터미널 사용 불가]');
      return;
    }

    let unlistenLine: (() => void) | null = null;
    let unlistenCwd: (() => void) | null = null;
    let unlistenExit: (() => void) | null = null;
    let running = false;

    const setRunning = (on: boolean): void => {
      running = on;
      if (on) bar.classList.add('kt-running'); else bar.classList.remove('kt-running');
      btnStart.disabled = on;
      btnStop.disabled = !on;
      input.disabled = !on;
      btnSend.disabled = !on;
    };

    const wireEvents = async (): Promise<void> => {
      if (!listen) return;
      try {
        unlistenLine = await listen('karmolab://terminal-line', (e: any) => {
          const p = (e?.payload || {}) as TerminalLineEvt;
          appendLine(out, p.line ?? '', p.stream === 'stderr' ? 'kt-stderr' : '');
        });
        unlistenCwd = await listen('karmolab://terminal-cwd', (e: any) => {
          const p = (e?.payload || {}) as TerminalCwdEvt;
          if (p.cwd) cwdLabel.textContent = `cwd: ${p.cwd}`;
        });
        unlistenExit = await listen('karmolab://terminal-exit', (e: any) => {
          const p = (e?.payload || {}) as TerminalExitEvt;
          appendMeta(out, `[셸 종료 — exit code ${p.code ?? '?'}]`);
          setRunning(false);
        });
      } catch (err) {
        console.error('terminal listen 실패', err);
      }
    };

    const teardownEvents = (): void => {
      try { unlistenLine?.(); } catch (_) { /* ignore */ }
      try { unlistenCwd?.(); } catch (_) { /* ignore */ }
      try { unlistenExit?.(); } catch (_) { /* ignore */ }
      unlistenLine = unlistenCwd = unlistenExit = null;
    };

    btnStart.addEventListener('click', async () => {
      btnStart.disabled = true;
      try {
        await wireEvents();
        const res = await invoke('terminal_start') as TerminalStartResult;
        shellLabel.textContent = res.shell;
        cwdLabel.textContent = `cwd: ${res.cwd}`;
        if (res.alreadyRunning) {
          appendMeta(out, '[이미 실행 중인 셸에 연결됨]');
        } else {
          appendMeta(out, `[${res.shell} 시작 — pwsh.exe / powershell.exe 자동선택]`);
        }
        setRunning(true);
        input.focus();
      } catch (err) {
        appendMeta(out, `[시작 실패: ${String(err)}]`);
        teardownEvents();
        btnStart.disabled = false;
      }
    });

    btnStop.addEventListener('click', async () => {
      btnStop.disabled = true;
      try {
        await invoke('terminal_stop');
        appendMeta(out, '[종료 요청 — taskkill /T /F]');
      } catch (err) {
        appendMeta(out, `[종료 실패: ${String(err)}]`);
      }
      // exit 이벤트가 setRunning(false) 처리. 안 와도 일정 시간 뒤 정리.
      setTimeout(() => { if (running) setRunning(false); teardownEvents(); }, 1500);
    });

    form.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      if (!running) return;
      const line = input.value;
      input.value = '';
      appendLine(out, `> ${line}`, 'kt-input');
      try {
        await invoke('terminal_send_stdin', { line });
      } catch (err) {
        appendMeta(out, `[입력 실패: ${String(err)}]`);
      }
    });

    // 위젯 unmount 자동 감지 — container 가 DOM 에서 떨어질 때 stop + listener 정리.
    const observer = new MutationObserver(() => {
      if (!document.body.contains(container)) {
        observer.disconnect();
        if (running) {
          // best-effort: stop 호출. unmount 시점이라 응답 무시.
          try { void invoke('terminal_stop'); } catch (_) { /* ignore */ }
        }
        teardownEvents();
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    // 첫 진입 안내.
    appendMeta(out, '시작 버튼을 누르면 PowerShell 단일 셸 spawn. cd / Set-Location 자동 추적 (cwd 라인).');

    // 이미 실행 중인 셸이 있을 수 있음 — status 체크 후 자동 reattach.
    try {
      const s = await invoke('terminal_status') as { running: boolean };
      if (s.running) {
        await wireEvents();
        appendMeta(out, '[기존 셸에 reattach]');
        setRunning(true);
        input.focus();
      }
    } catch (_) { /* ignore */ }
  }

  function appendLine(out: HTMLElement, text: string, extraClass: string): void {
    const div = document.createElement('div');
    div.className = `kt-line${extraClass ? ' ' + extraClass : ''}`;
    div.textContent = text;
    out.appendChild(div);
    capLines(out);
    scrollToEnd(out);
  }

  function appendMeta(out: HTMLElement, text: string): void {
    appendLine(out, text, 'kt-meta');
  }

  function capLines(out: HTMLElement): void {
    while (out.childNodes.length > MAX_OUTPUT_LINES) {
      out.removeChild(out.firstChild!);
    }
  }

  function scrollToEnd(out: HTMLElement): void {
    out.scrollTop = out.scrollHeight;
  }
})();
