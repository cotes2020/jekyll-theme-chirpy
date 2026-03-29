(function () {
    'use strict';

    const PREFS_KEY = 'servermonitor_base';

    function build(container) {
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
        saveBtn.onclick = function () {
            const v = baseInput.value.trim();
            if (Toolbox.setPref) Toolbox.setPref(PREFS_KEY, v);
            Toolbox.showToast('저장됨');
        };

        const refreshBtn = document.createElement('button');
        refreshBtn.className = 'btn btn-primary';
        refreshBtn.textContent = '상태 조회';
        refreshBtn.style.marginLeft = '8px';

        const statusBox = document.createElement('div');
        statusBox.id = 'smStatusBox';
        statusBox.className = 'sm-status-box';

        Mdd.injectCSS('servermonitor', `
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
        `);

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

        async function pingLocal(url) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 2000); // 2s timeout
                // Use no-cors to check if the port is open without being blocked by CORS
                await fetch(url, { mode: 'no-cors', signal: controller.signal, cache: 'no-cache' });
                clearTimeout(timeoutId);
                return 'online';
            } catch (e) {
                return 'offline';
            }
        }

        async function loadConfig() {
            const configPath = '/apps/karmolab/data/servermonitor-config.json';
            try {
                const res = await fetch(configPath, { cache: 'no-cache' });
                if (!res.ok) throw new Error('설정 파일을 찾을 수 없습니다.');
                return await res.json();
            } catch (e) {
                console.error('[ServerMonitor] 설정 로드 실패:', e.message);
                return { localMonitors: [] };
            }
        }

        async function fetchStatus() {
            let base = (baseInput.value.trim() || (Toolbox.getPref && Toolbox.getPref(PREFS_KEY, ''))) || '';
            if (!base && Toolbox.getPref) base = Toolbox.getPref('ytdl_cobalt_base', '') || '';
            
            statusBox.innerHTML = '조회 중...';
            statusBox.className = 'sm-status-box loading';
            refreshBtn.disabled = true;

            try {
                // 0. Load Configuration from JSON
                const config = await loadConfig();
                const localTargets = config.localMonitors || [];

                // 1. Check Local Servers
                const localResults = await Promise.all(localTargets.map(async m => {
                    const status = await pingLocal(m.url);
                    return { ...m, status };
                }));

                // 2. Check Remote Server (if configured)
                let remoteHtml = '';
                if (base) {
                    try {
                        const url = base.replace(/\/$/, '');
                        const res = await fetch(url + '/api/status');
                        const data = await res.json().catch(() => ({}));
                        
                        if (data.error) {
                            remoteHtml = `<div class="sm-row"><span style="color:var(--error)">원격 서버 오류: ${data.error}</span></div>`;
                        } else {
                            const m = data.memory || {};
                            const d = data.disk || {};
                            const svc = data.services || {};
                            const ytStatus = svc['yt-api'] === 'ok' ? 'ok' : 'offline';
                            const dcStatus = svc['discord-bot'] === 'running' ? 'running' : (svc['discord-bot'] === 'unknown' ? 'unknown' : 'offline');
                            
                            remoteHtml = `
                                <div class="sm-row"><span>원격 서버</span><span style="color:var(--success)">● 온라인</span></div>
                                <div class="sm-row"><span>CPU</span><span>${data.cpu}%</span></div>
                                <div class="sm-row"><span>메모리</span><span>${m.used_gb}/${m.total_gb} GB (${m.percent}%)</span></div>
                                <div class="sm-row"><span>디스크</span><span>${d.used_gb}/${d.total_gb} GB (${d.percent}%)</span></div>
                                <div class="sm-row"><span>가동시간</span><span>${data.uptime || '-'}</span></div>
                                <div class="sm-services">
                                    <div class="sm-service ${ytStatus}"><strong>yt-api</strong><br>${ytStatus === 'ok' ? '정상' : '오프라인'}</div>
                                    <div class="sm-service ${dcStatus}"><strong>봇 서버</strong><br>${dcStatus === 'running' ? '실행 중' : dcStatus === 'unknown' ? '확인 불가' : '오프라인'}</div>
                                </div>
                            `;
                        }
                    } catch (e) {
                        remoteHtml = `<div class="sm-row"><span style="color:var(--error)">원격 연결 실패: ${e.message || 'Error'}</span></div>`;
                    }
                } else {
                    remoteHtml = `<div class="sm-row"><span style="color:var(--text-tertiary)">원격 서버가 설정되지 않았습니다.</span></div>`;
                }

                // 3. Render Status
                const localHtml = localResults.map(r => `
                    <div class="sm-row">
                        <span>${r.label}</span>
                        <span style="color:${r.status === 'online' ? 'var(--success)' : 'var(--error)'}">
                            ● ${r.status === 'online' ? 'Run' : 'Down'}
                        </span>
                    </div>
                `).join('');

                statusBox.innerHTML = `
                    <div style="font-weight:700; margin-bottom:10px; color:var(--accent)">내 컴퓨터 서버 상태</div>
                    ${localHtml}
                    <div style="font-weight:700; margin-top:20px; margin-bottom:10px; color:var(--accent)">원격 서버 상태</div>
                    ${remoteHtml}
                `;
                statusBox.className = 'sm-status-box';
            } catch (e) {
                statusBox.innerHTML = '조회 실패: ' + (e.message || '알 수 없는 오류');
                statusBox.className = 'sm-status-box error';
            } finally {
                refreshBtn.disabled = false;
            }
        }

        refreshBtn.onclick = fetchStatus;
    }

    Toolbox.register({
        id: 'servermonitor',
        title: '서버 모니터',
        category: null,  // 기타
        desc: '서버 상태를 모니터링합니다',
        layout: 'form',
        icon: '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
        tabs: [{ id: 'main', label: '상태', build: build }]
    });
})();
