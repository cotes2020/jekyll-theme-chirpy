/**
 * 개발·디버그용 패널 (Tauri 전용 기능 등). 항목은 섹션 단위로 추가.
 */
(function (): void {
  'use strict';

  type NotifyPayload = {
    title: string;
    body: string;
    sound?: string;
    image_path?: string;
  };

  function desktopInvoke(cmd: string, args: unknown): Promise<unknown> {
    const core = window.__TAURI__?.core;
    const fn = core && typeof core.invoke === 'function' ? core.invoke : null;
    if (!fn) return Promise.reject(new Error('Tauri invoke 없음 (웹 브라우저 또는 withGlobalTauri 비활성)'));
    return fn(cmd, args);
  }

  function buildNotifyPayload(
    titleIn: HTMLInputElement,
    bodyIn: HTMLTextAreaElement,
    soundSel: HTMLSelectElement,
    imageIn: HTMLInputElement
  ): NotifyPayload {
    const title = (titleIn.value || '').trim() || 'KarmoLab';
    const body = (bodyIn.value || '').trim() || 'KarmoLab';
    const o: NotifyPayload = { title: title.slice(0, 120), body: body.slice(0, 2000) };
    const snd = soundSel.value;
    if (snd && snd !== 'silent') o.sound = snd;
    const img = (imageIn.value || '').trim();
    if (img) o.image_path = img;
    return o;
  }

  function buildNotifySection(wrap: HTMLElement): void {
    const sec = document.createElement('section');
    sec.className = 'devtools-section';

    const h = document.createElement('h3');
    h.className = 'devtools-section-title';
    h.textContent = 'OS 알림';

    const notifyLevelKey = 'karmolab_os_notify_level';
    const initLevel = localStorage.getItem(notifyLevelKey) || 'important';

    const pLevel = document.createElement('p');
    pLevel.className = 'devtools-section-desc';
    pLevel.innerHTML = '<strong>OS 알림 연동 수준</strong>: ';
    const levelSel = document.createElement('select');
    levelSel.className = 'devtools-select';
    levelSel.style.width = 'auto';
    [
      ['all', '모든 작업결과 (All)'],
      ['important', '에러 및 중요 알림만 (Error Only)'],
      ['off', '완전 끄기 (Off)'],
    ].forEach(([v, text]) => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = text;
      if (v === initLevel) opt.selected = true;
      levelSel.appendChild(opt);
    });
    levelSel.addEventListener('change', () => {
      localStorage.setItem(notifyLevelKey, levelSel.value);
      if (typeof Toolbox !== 'undefined') Toolbox.showToast('알림 설정이 저장되었습니다.', 'success');
    });
    pLevel.appendChild(levelSel);

    const p = document.createElement('p');
    const isApp = typeof Toolbox.isDesktopApp === 'function' && Toolbox.isDesktopApp();
    p.className = 'devtools-section-desc';
    p.innerHTML = isApp
      ? '<code>desktop_notify</code> 인자는 아래 미리보기와 동일하게 전송됩니다. Windows에서 <code>Default</code>는 WebView 쪽에서 <code>IM</code> 알림음으로 바꿔 보냅니다(그렇지 않으면 무음).'
      : '웹 브라우저에서는 사용할 수 없습니다. KarmoLab Tauri 앱으로 열어 주세요.';

    sec.appendChild(h);
    sec.appendChild(pLevel);
    sec.appendChild(p);

    const mkField = function (labelText: string, inner: HTMLElement): HTMLElement {
      const row = document.createElement('div');
      row.className = 'devtools-field';
      const lab = document.createElement('label');
      lab.className = 'devtools-field-label';
      lab.textContent = labelText;
      row.appendChild(lab);
      row.appendChild(inner);
      return row;
    };

    const titleIn = document.createElement('input');
    titleIn.type = 'text';
    titleIn.className = 'devtools-input';
    titleIn.value = 'KarmoLab 알림 테스트';
    titleIn.disabled = !isApp;

    const bodyIn = document.createElement('textarea');
    bodyIn.className = 'devtools-textarea';
    bodyIn.rows = 3;
    bodyIn.value = '알림이 보이면 정상입니다. ' + new Date().toLocaleString();
    bodyIn.disabled = !isApp;

    const soundSel = document.createElement('select');
    soundSel.className = 'devtools-select';
    soundSel.disabled = !isApp;
    (
      [
        ['silent', '무음 (sound 생략)'],
        ['Default', 'Default → Win에서 IM 알림음'],
        ['IM', 'IM'],
        ['Mail', 'Mail'],
        ['SMS', 'SMS'],
        ['Reminder', 'Reminder'],
        ['Alarm', 'Alarm'],
        ['Call', 'Call']
      ] as const
    ).forEach(function (opt) {
      const o = document.createElement('option');
      o.value = opt[0];
      o.textContent = opt[1];
      soundSel.appendChild(o);
    });
    soundSel.value = 'Default';

    const imageIn = document.createElement('input');
    imageIn.type = 'text';
    imageIn.className = 'devtools-input';
    imageIn.placeholder = 'image_path (선택) 예: C:\\\\path\\\\icon.png';
    imageIn.disabled = !isApp;

    const previewLabel = document.createElement('div');
    previewLabel.className = 'devtools-preview-label';
    previewLabel.textContent = '전송 JSON (invoke 두 번째 인자)';

    const preview = document.createElement('pre');
    preview.className = 'devtools-preview';
    preview.setAttribute('aria-label', 'desktop_notify 페이로드');

    const syncPreview = function (): void {
      preview.textContent = JSON.stringify(buildNotifyPayload(titleIn, bodyIn, soundSel, imageIn), null, 2);
    };

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-primary';
    btn.textContent = '이 페이로드로 보내기';
    btn.disabled = !isApp;

    const status = document.createElement('div');
    status.className = 'devtools-log';
    status.textContent = isApp ? '위 JSON이 그대로 전달됩니다.' : '데스크톱 앱이 아니면 비활성입니다.';

    [titleIn, bodyIn, soundSel, imageIn].forEach(function (el) {
      el.addEventListener('input', syncPreview);
      el.addEventListener('change', syncPreview);
    });

    btn.addEventListener('click', function () {
      syncPreview();
      const payload = buildNotifyPayload(titleIn, bodyIn, soundSel, imageIn);
      if (typeof window.__karmolabSetNotifyInvokeDebug === 'function') {
        window.__karmolabSetNotifyInvokeDebug(payload);
      }
      status.className = 'devtools-log';
      status.textContent = '요청 중…\n\n' + JSON.stringify(payload, null, 2);
      void desktopInvoke('desktop_notify', payload)
        .then(function () {
          status.className = 'devtools-log devtools-log-ok';
          status.textContent = 'invoke 성공.\n\n전송 페이로드:\n' + JSON.stringify(payload, null, 2);
        })
        .catch(function (e: unknown) {
          status.className = 'devtools-log devtools-log-err';
          const errMsg = e instanceof Error ? e.message : String(e);
          status.textContent = errMsg + '\n\n전송 시도 페이로드:\n' + JSON.stringify(payload, null, 2);
          Toolbox.showToast?.('알림 요청 실패', 'error', e);
        });
    });

    sec.appendChild(mkField('title', titleIn));
    sec.appendChild(mkField('body', bodyIn));
    sec.appendChild(mkField('sound', soundSel));
    sec.appendChild(mkField('image_path', imageIn));
    sec.appendChild(previewLabel);
    sec.appendChild(preview);
    sec.appendChild(btn);
    sec.appendChild(status);
    wrap.appendChild(sec);

    syncPreview();
  }

  function buildReleaseSection(wrap: HTMLElement): void {
    const sec = document.createElement('section');
    sec.className = 'devtools-section';

    const h = document.createElement('h3');
    h.className = 'devtools-section-title';
    h.textContent = '릴리스 워크플로';

    const p = document.createElement('p');
    const isApp = typeof Toolbox.isDesktopApp === 'function' && Toolbox.isDesktopApp();
    const currentVersion = isApp ? window.__KARMOLAB_VERSION__ || '?' : '-';
    p.className = 'devtools-section-desc';
    p.innerHTML = isApp
      ? `현재 설치 버전: <code>${currentVersion}</code><br>GitHub CLI(<code>gh</code>)로 <code>KarmoLab Tauri Release</code> 워크플로를 원격 실행합니다. 이 PC에 <code>gh auth login</code>이 되어 있어야 합니다.`
      : '웹 브라우저에서는 사용할 수 없습니다. KarmoLab Tauri 앱으로 열어 주세요.';

    const row = document.createElement('div');
    row.className = 'devtools-field';

    const lab = document.createElement('label');
    lab.className = 'devtools-field-label';
    lab.textContent = 'ref (branch/tag)';

    const refIn = document.createElement('input');
    refIn.type = 'text';
    refIn.className = 'devtools-input';
    refIn.value = 'master';
    refIn.placeholder = 'master';
    refIn.disabled = !isApp;

    const bumpRow = document.createElement('div');
    bumpRow.className = 'devtools-field';
    const bumpLab = document.createElement('label');
    bumpLab.className = 'devtools-field-label';
    bumpLab.textContent = 'version bump';
    const bumpSel = document.createElement('select');
    bumpSel.className = 'devtools-select';
    bumpSel.disabled = !isApp;

    const verParts = (isApp ? currentVersion.split('.').map((n) => parseInt(n, 10)) : []);
    const [maj, min, pat] = [verParts[0], verParts[1], verParts[2]];
    const valid = Number.isFinite(maj) && Number.isFinite(min) && Number.isFinite(pat);
    const preview = (label: string, next: string): string =>
      valid ? `${label} (${currentVersion} → ${next})` : label;
    const opts: ReadonlyArray<readonly [string, string]> = [
      ['patch', preview('patch', `${maj}.${min}.${pat + 1}`)],
      ['minor', preview('minor', `${maj}.${min + 1}.0`)],
      ['major', preview('major', `${maj + 1}.0.0`)],
      ['none', 'none (버전 그대로 — 같은 태그 덮어쓰기)']
    ];
    opts.forEach(function (opt) {
      const o = document.createElement('option');
      o.value = opt[0];
      o.textContent = opt[1];
      bumpSel.appendChild(o);
    });
    bumpSel.value = 'patch';

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-primary';
    btn.textContent = '워크플로 실행';
    btn.disabled = !isApp;

    const status = document.createElement('div');
    status.className = 'devtools-log';
    status.textContent = isApp
      ? '실행 시 GitHub Actions workflow_dispatch를 호출합니다. bump≠none이면 워크플로가 master에 버전 bump 커밋을 직접 푸시합니다.'
      : '데스크톱 앱이 아니면 비활성입니다.';

    btn.addEventListener('click', function () {
      const selectedRef = (refIn.value || '').trim() || 'master';
      const selectedBump = bumpSel.value || 'patch';
      status.className = 'devtools-log';
      status.textContent = `요청 중…\nworkflow: KarmoLab Tauri Release\nref: ${selectedRef}\nbump: ${selectedBump}`;
      void desktopInvoke('desktop_trigger_release_workflow', {
        refName: selectedRef,
        bumpType: selectedBump
      })
        .then(function (res: unknown) {
          status.className = 'devtools-log devtools-log-ok';
          status.textContent = typeof res === 'string' ? res : JSON.stringify(res, null, 2);
        })
        .catch(function (e: unknown) {
          status.className = 'devtools-log devtools-log-err';
          const errMsg = e instanceof Error ? e.message : String(e);
          status.textContent = errMsg;
          Toolbox.showToast?.('릴리스 실행 실패', 'error', e);
        });
    });

    row.appendChild(lab);
    row.appendChild(refIn);
    bumpRow.appendChild(bumpLab);
    bumpRow.appendChild(bumpSel);
    sec.appendChild(h);
    sec.appendChild(p);
    sec.appendChild(row);
    sec.appendChild(bumpRow);
    sec.appendChild(btn);
    sec.appendChild(status);
    wrap.appendChild(sec);
  }

  function build(container: HTMLElement): void {
    Mdd.injectCSS(
      'devtools',
      `
            .devtools-root { max-width: 560px; }
            .devtools-intro { font-size: var(--font-size-sm); color: var(--text-tertiary); margin: 0 0 20px 0; line-height: 1.5; }
            .devtools-section { margin-bottom: 28px; padding-bottom: 24px; border-bottom: 1px solid var(--border); }
            .devtools-section:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
            .devtools-section-title { font-size: 14px; font-weight: 600; color: var(--text-primary); margin: 0 0 8px 0; }
            .devtools-section-desc { font-size: var(--font-size-sm); color: var(--text-secondary); line-height: 1.55; margin: 0 0 12px 0; }
            .devtools-field { margin-bottom: 12px; }
            .devtools-field-label { display: block; font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; }
            .devtools-input, .devtools-textarea, .devtools-select {
                width: 100%; max-width: 520px; box-sizing: border-box;
                padding: 8px 10px; border-radius: var(--radius-md);
                border: 1px solid var(--border); background: var(--bg-primary); color: var(--text-primary);
                font-size: var(--font-size-sm); font-family: inherit;
            }
            .devtools-textarea { resize: vertical; min-height: 72px; }
            .devtools-preview-label { font-size: 12px; font-weight: 600; color: var(--text-secondary); margin: 16px 0 6px 0; }
            .devtools-preview {
                margin: 0 0 12px 0; padding: 12px 14px; border-radius: var(--radius-md);
                background: var(--bg-tertiary); border: 1px solid var(--border);
                font-size: var(--font-size-xs); font-family: ui-monospace, monospace;
                color: var(--text-secondary); white-space: pre-wrap; word-break: break-word;
                max-width: 560px; max-height: 220px; overflow: auto;
            }
            .devtools-log { margin-top: 12px; padding: 12px 14px; border-radius: var(--radius-md); background: var(--bg-tertiary); border: 1px solid var(--border); font-size: var(--font-size-xs); font-family: ui-monospace, monospace; color: var(--text-secondary); white-space: pre-wrap; word-break: break-word; min-height: 2.5em; }
            .devtools-log-ok { border-color: var(--success-subtle, rgba(34,197,94,0.35)); color: var(--text-primary); }
            .devtools-log-err { border-color: var(--error-subtle); color: var(--error); }
        `
    );

    container.innerHTML = '';
    const root = document.createElement('div');
    root.className = 'devtools-root';

    const intro = document.createElement('p');
    intro.className = 'devtools-intro';
    intro.textContent = '배포·사용자용 기능이 아니라, 데스크톱 셸·연동을 점검할 때 쓰는 모음입니다.';

    root.appendChild(intro);
    buildReleaseSection(root);
    buildNotifySection(root);
    container.appendChild(root);
  }

  Toolbox.register({
    id: 'devtools',
    title: '디버그',
    category: 'desktop',
    desc: '데스크톱 앱·연동 점검용 테스트 모음 (알림 등)',
    layout: 'form',
    icon: '<rect x="2" y="4" width="20" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M6 9l3 3-3 3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><line x1="11" y1="15" x2="18" y2="15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
    tabs: [{ id: 'devtools-main', label: '패널', build }]
  });
})();
