(function (): void {
  Mdd.injectCSS(
    'dashboard',
    `
        .dash-layout { display:flex; flex-direction:column; gap:16px; }
        .dash-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; }
        .dash-card { background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-md); padding:16px; text-align:center; }
        .dash-card-value { font-size:28px; font-weight:700; color:var(--accent); margin-bottom:4px; font-family:monospace; }
        .dash-card-label { font-size:var(--font-size-xs); color:var(--text-secondary); }
        .dash-table-wrap { overflow-x:auto; }
        .dash-table { width:100%; border-collapse:collapse; font-size:var(--font-size-xs); }
        .dash-table th, .dash-table td { padding:8px 12px; text-align:right; border-bottom:1px solid var(--border); }
        .dash-table th { background:var(--bg-secondary); color:var(--text-secondary); font-weight:600; text-align:left; }
        .dash-table th:first-child { text-align:left; }
        .dash-table td:first-child { text-align:left; font-weight:500; color:var(--text-primary); }
        .dash-table tr:hover td { background:var(--bg-hover); }
        .dash-actions { display:flex; gap:8px; justify-content:flex-end; }
        .dash-empty { text-align:center; padding:40px; color:var(--text-tertiary); font-size:var(--font-size-sm); }
    `
  );

  function buildDashboard(container: HTMLElement): void {
    Mdd.linePreset('tool_run', { msg: '사용 기록을 볼까요?' });
    render(container);
  }

  function render(container: HTMLElement): void {
    const stats = Toolbox.getUsageStats?.() ?? {};
    const days = Object.keys(stats).sort().reverse();

    let totalChat = 0;
    let totalImage = 0;
    let totalChatTokens = 0;
    let totalImageTokens = 0;
    days.forEach((d) => {
      const s = stats[d];
      totalChat += s.chatCount ?? 0;
      totalImage += s.imageCount ?? 0;
      totalChatTokens += s.chatTokens ?? 0;
      totalImageTokens += s.imageTokens ?? 0;
    });

    if (days.length === 0) {
      container.innerHTML =
        '<div class="dash-empty"><div style="font-size:40px;margin-bottom:12px;opacity:0.3;">📊</div>아직 사용 기록이 없습니다.<br>챗봇이나 이미지 생성을 사용하면 여기에 기록됩니다.</div>';
      return;
    }

    container.innerHTML = `
            <div class="dash-layout">
                <div class="dash-cards">
                    <div class="dash-card">
                        <div class="dash-card-value">${totalChat.toLocaleString()}</div>
                        <div class="dash-card-label">💬 총 채팅 횟수</div>
                    </div>
                    <div class="dash-card">
                        <div class="dash-card-value">${totalImage.toLocaleString()}</div>
                        <div class="dash-card-label">🎨 총 이미지 생성</div>
                    </div>
                    <div class="dash-card">
                        <div class="dash-card-value">${formatTokens(totalChatTokens)}</div>
                        <div class="dash-card-label">💬 총 채팅 토큰</div>
                    </div>
                    <div class="dash-card">
                        <div class="dash-card-value">${formatTokens(totalImageTokens)}</div>
                        <div class="dash-card-label">🎨 총 이미지 토큰</div>
                    </div>
                </div>
                <div class="dash-table-wrap">
                    <table class="dash-table">
                        <thead>
                            <tr><th>날짜</th><th>채팅</th><th>채팅 토큰</th><th>이미지</th><th>이미지 토큰</th></tr>
                        </thead>
                        <tbody>
                            ${days
                              .map((d) => {
                                const s = stats[d];
                                return `<tr><td>${d}</td><td>${(s.chatCount ?? 0).toLocaleString()}</td><td>${formatTokens(s.chatTokens ?? 0)}</td><td>${(s.imageCount ?? 0).toLocaleString()}</td><td>${formatTokens(s.imageTokens ?? 0)}</td></tr>`;
                              })
                              .join('')}
                        </tbody>
                    </table>
                </div>
                <div class="dash-actions">
                    <button class="btn btn-ghost" id="dashRefresh">🔄 새로고침</button>
                    <button class="btn btn-danger" id="dashClear">🗑️ 기록 초기화</button>
                </div>
            </div>`;

    const refreshEl = container.querySelector('#dashRefresh') as HTMLButtonElement | null;
    const clearEl = container.querySelector('#dashClear') as HTMLButtonElement | null;
    if (refreshEl) refreshEl.onclick = () => render(container);
    if (clearEl) {
      clearEl.onclick = () => {
        if (!confirm('모든 사용량 기록을 삭제하시겠습니까?')) return;
        localStorage.removeItem('toolbox_usage_stats');
        render(container);
        Toolbox.showToast?.('사용량 기록 초기화 완료', undefined, undefined);
      };
    }
  }

  function formatTokens(n: number): string {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toLocaleString();
  }

  window.DashboardBuild = buildDashboard;
})();
