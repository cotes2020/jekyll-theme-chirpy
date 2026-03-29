/**
 * 개발·디버그용 패널 (Tauri 전용 기능 등). 항목은 섹션 단위로 추가.
 * 데스크톱 앱은 src-tauri에서 동일 파일을 include하여 캐시된 웹과 무관하게 등록 보장.
 */
(function () {
    'use strict';

    function desktopInvoke(cmd, args) {
        const core = window.__TAURI__ && window.__TAURI__.core;
        const fn = core && typeof core.invoke === 'function' ? core.invoke : null;
        if (!fn) return Promise.reject(new Error('Tauri invoke 없음 (웹 브라우저 또는 withGlobalTauri 비활성)'));
        return fn(cmd, args);
    }

    function buildNotifySection(wrap) {
        const sec = document.createElement('section');
        sec.className = 'devtools-section';

        const h = document.createElement('h3');
        h.className = 'devtools-section-title';
        h.textContent = 'OS 알림';

        const p = document.createElement('p');
        const isApp = typeof Toolbox.isDesktopApp === 'function' && Toolbox.isDesktopApp();
        p.className = 'devtools-section-desc';
        p.innerHTML = isApp
            ? '<code>desktop_notify</code> 호출로 시스템 알림을 띄웁니다. Windows는 알림·집중 방해 설정에 따라 다릅니다.'
            : '웹 브라우저에서는 사용할 수 없습니다. KarmoLab Tauri 앱으로 열어 주세요.';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-primary';
        btn.textContent = '테스트 알림 보내기';
        btn.disabled = !isApp;

        const status = document.createElement('div');
        status.className = 'devtools-log';
        status.textContent = isApp ? '준비됨.' : '데스크톱 앱이 아니면 비활성입니다.';

        btn.addEventListener('click', function () {
            status.className = 'devtools-log';
            status.textContent = '요청 중…';
            const body = '알림이 보이면 정상입니다. ' + new Date().toLocaleString();
            desktopInvoke('desktop_notify', { title: 'KarmoLab 알림 테스트', body: body })
                .then(function () {
                    status.className = 'devtools-log devtools-log-ok';
                    status.textContent = 'invoke 성공. 작업 표시줄·알림 센터를 확인하세요.';
                })
                .catch(function (e) {
                    status.className = 'devtools-log devtools-log-err';
                    status.textContent = e && e.message ? e.message : String(e);
                    Toolbox.showToast('알림 요청 실패', 'error', e);
                });
        });

        sec.appendChild(h);
        sec.appendChild(p);
        sec.appendChild(btn);
        sec.appendChild(status);
        wrap.appendChild(sec);
    }

    function build(container) {
        Mdd.injectCSS('devtools', `
            .devtools-root { max-width: 560px; }
            .devtools-intro { font-size: var(--font-size-sm); color: var(--text-tertiary); margin: 0 0 20px 0; line-height: 1.5; }
            .devtools-section { margin-bottom: 28px; padding-bottom: 24px; border-bottom: 1px solid var(--border); }
            .devtools-section:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
            .devtools-section-title { font-size: 14px; font-weight: 600; color: var(--text-primary); margin: 0 0 8px 0; }
            .devtools-section-desc { font-size: var(--font-size-sm); color: var(--text-secondary); line-height: 1.55; margin: 0 0 12px 0; }
            .devtools-log { margin-top: 12px; padding: 12px 14px; border-radius: var(--radius-md); background: var(--bg-tertiary); border: 1px solid var(--border); font-size: var(--font-size-xs); font-family: ui-monospace, monospace; color: var(--text-secondary); white-space: pre-wrap; word-break: break-word; min-height: 2.5em; }
            .devtools-log-ok { border-color: var(--success-subtle, rgba(34,197,94,0.35)); color: var(--text-primary); }
            .devtools-log-err { border-color: var(--error-subtle); color: var(--error); }
        `);

        container.innerHTML = '';
        const root = document.createElement('div');
        root.className = 'devtools-root';

        const intro = document.createElement('p');
        intro.className = 'devtools-intro';
        intro.textContent = '배포·사용자용 기능이 아니라, 데스크톱 셸·연동을 점검할 때 쓰는 모음입니다.';

        root.appendChild(intro);
        buildNotifySection(root);
        container.appendChild(root);
    }

    Toolbox.register({
        id: 'devtools',
        title: '디버그',
        category: 'tool',
        desc: '데스크톱 앱·연동 점검용 테스트 모음 (알림 등)',
        layout: 'form',
        icon: '<rect x="2" y="4" width="20" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M6 9l3 3-3 3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><line x1="11" y1="15" x2="18" y2="15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
        tabs: [{ id: 'devtools-main', label: '패널', build: build }],
    });
})();
