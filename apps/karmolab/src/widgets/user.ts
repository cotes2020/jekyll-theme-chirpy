/**
 * User Page — 내 정보, 도전과제, 뱃지
 */
(function (): void {
    const PROGRESS_KEY = 'pet_strokes';
    /** [karmolab-react-src DEFAULT_TRACKS] id → 표시 이름 */
    const STREAK_TRACK_LABELS: Record<string, string> = { daily_review: '일일 리뷰', exercise: '운동' };

    type UserAchievement = {
        id: string;
        title: string;
        desc: string;
        icon: string;
        source: string;
    };

    type UserBadge = {
        id: string;
        title: string;
        desc: string;
        icon: string;
        source: string;
    };

    type UserStreak = {
        current?: number;
        longest?: number;
        lastActivityDate?: string;
    };

    type UserData = {
        achievements?: string[];
        badges?: string[];
        progress?: Record<string, number>;
        streaks?: Record<string, UserStreak>;
    };

    type StorageItemStat = { key: string; bytes: number; valLen: number };

    const DEFS: {
        achievements: UserAchievement[];
        badges: UserBadge[];
    } = {
        achievements: [
            { id: 'pet_100', title: '100번 쓰다듬기', desc: '고양이를 100번 쓰다듬었다', icon: '🐱', source: 'pet' },
            { id: 'pet_1000', title: '1,000번 쓰다듬기', desc: '고양이를 1,000번 쓰다듬었다', icon: '🐱', source: 'pet' },
            { id: 'pet_10000', title: '10,000번 쓰다듬기', desc: '집사 가끔 대단해요', icon: '🐱', source: 'pet' },
            { id: 'pet_100000', title: '100,000번 쓰다듬기', desc: '진짜로 하고 있었어요?!', icon: '🐱', source: 'pet' },
            { id: 'pet_500000', title: '500,000번 쓰다듬기', desc: '반이에요... 설마 진심이에요?!', icon: '🐱', source: 'pet' },
            { id: 'first_chat', title: '첫 대화', desc: '챗봇과 첫 대화를 나눴다', icon: '💬', source: 'chatbot' },
            { id: 'first_image', title: '첫 이미지 생성', desc: '첫 이미지를 생성했다', icon: '🎨', source: 'imagegen' },
            { id: 'streak_first', title: '첫 줄기', desc: '처음으로 스트릭 하루를 채웠다', icon: '🌱', source: 'streak' },
            { id: 'streak_7', title: '7일 연속', desc: '어느 트랙이든 7일 연속 달성', icon: '🔥', source: 'streak' },
            { id: 'streak_30', title: '30일 연속', desc: '어느 트랙이든 30일 연속 달성', icon: '🔥', source: 'streak' },
            { id: 'streak_100', title: '100일 연속', desc: '어느 트랙이든 100일 연속 달성', icon: '🔥', source: 'streak' },
            { id: 'reaction_200', title: '초고속 반응 200ms', desc: '번개같은 반사신경', icon: '⚡', source: 'reaction' },
            { id: 'reaction_150', title: '번개 반응 150ms', desc: '인간의 한계를 넘었다', icon: '⚡', source: 'reaction' },
        ],
        badges: [
            { id: 'pet_marriage', title: '검의 서약', desc: '100만번 쓰다듬고 결혼했어요 💍', icon: '💖', source: 'pet' },
            { id: 'toolbox_explorer', title: '탐험가', desc: '5개 이상 도구를 사용했다', icon: '🧭', source: 'system' },
        ],
    };

    DEFS.achievements.forEach((a) => Toolbox.registerAchievement?.(a.id, a));
    DEFS.badges.forEach((b) => Toolbox.unlockBadge?.(b.id, b));

    Mdd.injectCSS('user-page', `
        .user-layout { display:flex; flex-direction:column; gap:24px; }
        .user-header { display:flex; align-items:center; gap:20px; padding:24px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-lg); }
        .user-avatar { width:72px; height:72px; border-radius:50%; background:linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%); display:flex; align-items:center; justify-content:center; font-size:36px; }
        .user-info { flex:1; }
        .user-info h2 { font-size:18px; font-weight:600; margin:0 0 4px 0; color:var(--text-primary); }
        .user-info p { font-size:var(--font-size-sm); color:var(--text-secondary); margin:0; }
        .user-quick-stats { display:flex; gap:16px; flex-wrap:wrap; margin-top:12px; }
        .user-quick-stat { font-size:var(--font-size-sm); color:var(--text-secondary); }
        .user-quick-stat strong { color:var(--accent); margin-right:4px; }
        .user-section h3 { font-size:14px; color:var(--text-secondary); margin-bottom:12px; display:flex; align-items:center; gap:8px; }
        .user-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:12px; }
        .user-item { background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-md); padding:16px; text-align:center; transition:opacity 0.2s; }
        .user-item.locked { opacity:0.5; filter:grayscale(0.8); }
        .user-item .user-item-icon { font-size:32px; margin-bottom:8px; }
        .user-item .user-item-title { font-size:var(--font-size-xs); font-weight:600; color:var(--text-primary); margin-bottom:4px; }
        .user-item .user-item-desc { font-size:var(--font-size-xs); color:var(--text-tertiary); }
        .user-actions { display:flex; gap:8px; justify-content:flex-end; flex-wrap:wrap; }
        .user-link { font-size:var(--font-size-sm); color:var(--accent); text-decoration:none; }
        .user-link:hover { text-decoration:underline; }
        .settings-row { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:12px 16px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-md); margin-bottom:8px; }
        .settings-row label { font-size:var(--font-size-sm); font-weight:500; color:var(--text-primary); white-space:nowrap; flex-shrink:0; }
        .settings-row .settings-control { min-width:140px; }
        .settings-section { margin-bottom:24px; }
        .settings-section h3 { font-size:14px; color:var(--text-secondary); margin-bottom:12px; }
        .settings-danger { border-color:var(--error-subtle); background:var(--error-subtle); }
        .settings-danger .btn-ghost { color:var(--error); }
        .settings-code-preview { margin-top:12px; font-size:var(--font-size-xs); }
        .settings-code-preview pre { margin:0; border-radius:var(--radius-md); overflow-x:auto; }
        .settings-code-preview pre code { padding:12px 14px; line-height:1.5; display:block; font-size:var(--font-size-xs); }
        .storage-summary { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }
        .storage-card { background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-md); padding:16px 20px; min-width:140px; }
        .storage-card-value { font-size:22px; font-weight:700; color:var(--accent); font-family:monospace; }
        .storage-card-label { font-size:var(--font-size-xs); color:var(--text-secondary); margin-top:4px; }
        .storage-table { width:100%; border-collapse:collapse; font-size:var(--font-size-xs); }
        .storage-table th, .storage-table td { padding:8px 12px; text-align:left; border-bottom:1px solid var(--border); }
        .storage-table th { background:var(--bg-secondary); color:var(--text-secondary); font-weight:600; }
        .storage-table td:last-child, .storage-table th:last-child { text-align:right; font-family:monospace; }
        .storage-table .storage-key { font-family:monospace; font-size:var(--font-size-xs); color:var(--text-primary); word-break:break-all; }
        .storage-table .storage-desc { font-size:var(--font-size-xs); color:var(--text-tertiary); max-width:200px; }
    `);

    function buildOverview(container: HTMLElement): void {
        const data = (Toolbox.getUserData?.() as UserData | undefined) ?? {};
        const achievements = data.achievements ?? [];
        const badges = data.badges ?? [];
        const progress = data.progress ?? {};
        const petStrokes = progress[PROGRESS_KEY] ?? 0;
        const usageStats = Toolbox.getUsageStats?.() ?? {};
        let totalChat = 0, totalImage = 0;
        const usageValues = Object.values(usageStats) as Array<{ chatCount?: number; imageCount?: number }>;
        usageValues.forEach((s) => {
            totalChat += s.chatCount ?? 0;
            totalImage += s.imageCount ?? 0;
        });

        const achCount = achievements.length;
        const badgeCount = badges.length;
        const totalA = DEFS.achievements.length;
        const totalB = DEFS.badges.length;
        const streaks = data.streaks ?? {};
        const streakIds = Object.keys(streaks);
        let maxStreakCurrent = 0;
        streakIds.forEach((id) => {
            const sc = streaks[id] && streaks[id].current;
            if (typeof sc === 'number' && sc > maxStreakCurrent) maxStreakCurrent = sc;
        });

        container.innerHTML = `
            <div class="user-layout">
                <div class="user-header">
                    <div class="user-avatar">👤</div>
                    <div class="user-info">
                        <h2>Toolbox 사용자</h2>
                        <p>마스코트 관계: <strong style="color:var(--secondary)">${Mdd.getRelationshipTitle()}</strong> · 호감도 ${Mdd.getAffection()}</p>
                        <div class="user-quick-stats">
                            <span class="user-quick-stat"><strong>${achCount}/${totalA}</strong> 도전과제</span>
                            <span class="user-quick-stat"><strong>${badgeCount}/${totalB}</strong> 뱃지</span>
                            <span class="user-quick-stat"><strong>${streakIds.length}</strong> 스트릭 트랙</span>
                            <span class="user-quick-stat"><strong>${maxStreakCurrent}</strong> 최고 연속(일)</span>
                            <span class="user-quick-stat"><strong>${petStrokes.toLocaleString()}</strong> 쓰담</span>
                            <span class="user-quick-stat"><strong>${totalChat}</strong> 채팅</span>
                            <span class="user-quick-stat"><strong>${totalImage}</strong> 이미지</span>
                        </div>
                    </div>
                </div>
            </div>`;
    }

    function buildUsage(container: HTMLElement): void {
        if (typeof window.DashboardBuild === 'function') {
            window.DashboardBuild(container);
        }
    }

    function buildAchievements(container: HTMLElement): void {
        Mdd.linePreset('achievement', { msg: '도전과제 보여줄게요~' });
        renderAchievements(container);
    }

    function renderAchievements(container: HTMLElement): void {
        const data = (Toolbox.getUserData?.() as UserData | undefined) ?? {};
        const achievements = data.achievements ?? [];
        const all = [...DEFS.achievements];

        container.innerHTML = `
            <div class="user-layout">
                <div class="user-section">
                    <h3>🏆 도전과제 (${achievements.length}/${all.length})</h3>
                    <div class="user-grid">
                        ${all.map(a => {
                            const unlocked = achievements.includes(a.id);
                            return `<div class="user-item ${unlocked ? '' : 'locked'}" title="${a.desc}">
                                <div class="user-item-icon">${unlocked ? a.icon : '🔒'}</div>
                                <div class="user-item-title">${a.title}</div>
                                <div class="user-item-desc">${a.desc}</div>
                            </div>`;
                        }).join('')}
                    </div>
                </div>
            </div>`;
    }

    function buildStreaks(container: HTMLElement): void {
        Mdd.linePreset('daily_start', { msg: '스트릭 현황이에요~' });
        renderStreaks(container);
    }

    function renderStreaks(container: HTMLElement): void {
        const data = (Toolbox.getUserData?.() as UserData | undefined) ?? {};
        const streaks = data.streaks ?? {};
        const ids = Object.keys(streaks);
        const labels = STREAK_TRACK_LABELS;

        container.innerHTML = `
            <div class="user-layout">
                <div class="user-section">
                    <h3>🔥 스트릭 (${ids.length} 트랙)</h3>
                    ${ids.length === 0 ? '<p style="font-size:var(--font-size-sm);color:var(--text-secondary);margin:0 0 12px 0;">아직 기록이 없어요. 플래너(React)에서 오늘 완료를 눌러보세요.</p>' : ''}
                    <div class="user-grid">
                        ${ids.map((id) => {
                            const s = streaks[id];
                            if (!s) return '';
                            const label = labels[id] || id;
                            const safeLabel = Toolbox.escapeHtml ? Toolbox.escapeHtml(label) : label;
                            const safeId = Toolbox.escapeHtml ? Toolbox.escapeHtml(id) : id;
                            return `<div class="user-item" title="${safeId}">
                                <div class="user-item-icon">🔥</div>
                                <div class="user-item-title">${safeLabel}</div>
                                <div class="user-item-desc">현재 ${s.current ?? 0}일 · 최장 ${s.longest ?? 0}일 · ${Toolbox.escapeHtml ? Toolbox.escapeHtml(s.lastActivityDate || '—') : (s.lastActivityDate || '—')}</div>
                            </div>`;
                        }).join('')}
                    </div>
                </div>
            </div>`;
    }

    function buildBadges(container: HTMLElement): void {
        Mdd.linePreset('tool_run', { msg: '뱃지 보여줄게요~' });
        renderBadges(container);
    }

    function renderBadges(container: HTMLElement): void {
        const data = (Toolbox.getUserData?.() as UserData | undefined) ?? {};
        const badges = data.badges ?? [];
        const all = [...DEFS.badges];

        container.innerHTML = `
            <div class="user-layout">
                <div class="user-section">
                    <h3>🎖️ 뱃지 (${badges.length}/${all.length})</h3>
                    <div class="user-grid">
                        ${all.map(b => {
                            const unlocked = badges.includes(b.id);
                            return `<div class="user-item ${unlocked ? '' : 'locked'}" title="${b.desc}">
                                <div class="user-item-icon">${unlocked ? b.icon : '🔒'}</div>
                                <div class="user-item-title">${b.title}</div>
                                <div class="user-item-desc">${b.desc}</div>
                            </div>`;
                        }).join('')}
                    </div>
                </div>
                <div class="user-actions">
                    <button class="btn btn-danger" id="userReset">🗑️ 유저 데이터 초기화</button>
                </div>
            </div>`;

        container.querySelector<HTMLButtonElement>('#userReset')?.addEventListener('click', () => {
            if (!confirm('모든 도전과제, 뱃지, 진행도를 초기화합니다. 계속할까요?')) return;
            localStorage.removeItem('toolbox_user_data');
            renderBadges(container);
            (Toolbox as any).showToast?.('유저 데이터 초기화 완료');
        });
    }

    /** 키별 용도 설명 (Toolbox 관련) */
    const STORAGE_DESC: Record<string, string> = {
        'toolbox_theme': '테마 (라이트/다크)',
        'toolbox_prism_theme': '코드 하이라이트 테마',
        'toolbox_last_page': '마지막 접속 페이지',
        'toolbox_widget_prefs': '위젯별 설정 (모델, 프리셋 등)',
        'toolbox_usage_stats': 'AI 사용량 통계 (채팅/이미지)',
        'toolbox_user_data': '유저 데이터 (도전과제, 뱃지, 진행도)',
        'toolbox_gemini_api_key': 'Gemini API 키 (구버전)',
        'toolbox_gemini_api_keys_v2': 'Gemini API 키 목록 (AI Studio)',
        'toolbox_vertex_api_key': 'Vertex AI (Google Cloud) API 키',
        'toolbox_memos': '메모 위젯',
        'toolbox_tierlists': '티어리스트',
        'toolbox_imagegen_custom_presets': '이미지 생성 커스텀 프리셋',
        'toolbox_ig_prompt_history': '이미지 생성 프롬프트 기록',
        'toolbox_chatbot_sessions_index': '챗봇 세션 인덱스',
        'karmolab_chatbot_characters_v1': '챗봇 캐릭터 카드 목록 (JSON 배열; karmochat_character_v1 내보내기와 별개로 localStorage에 저장)',
        'mdd_affection': '마스코트 호감도',
        'mdd_story_progress': '마스코트 스토리 진행',
    };

    function getStorageStats(storage: Storage): { totalBytes: number; items: StorageItemStat[] } {
        let totalBytes = 0;
        const items: StorageItemStat[] = [];
        try {
            for (let i = 0; i < storage.length; i++) {
                const key = storage.key(i);
                if (key == null) continue;
                const val = storage.getItem(key) ?? '';
                const bytes = (key.length + val.length) * 2;
                totalBytes += bytes;
                items.push({ key, bytes, valLen: val.length });
            }
        } catch (_) {}
        items.sort((a, b) => b.bytes - a.bytes);
        return { totalBytes, items };
    }

    function formatBytes(bytes: number): string {
        if (bytes >= 1048576) return (bytes / 1048576).toFixed(2) + ' MB';
        if (bytes >= 1024) return (bytes / 1024).toFixed(2) + ' KB';
        return bytes + ' B';
    }

    function buildStorage(container: HTMLElement): void {
        Mdd.linePreset('tool_run', { msg: '저장소 상태 보여줄게요~' });
        renderStorage(container);
    }

    function renderStorage(container: HTMLElement): void {
        const ls = getStorageStats(localStorage);
        const ss = getStorageStats(sessionStorage);
        const totalBytes = ls.totalBytes + ss.totalBytes;

        function getDesc(key: string): string {
            if (STORAGE_DESC[key]) return STORAGE_DESC[key] ?? '';
            if (key.startsWith('toolbox_chatbot_session')) return '챗봇 대화 내용';
            if (key.startsWith('toolbox_')) return 'KarmoLab';
            if (key.startsWith('mdd_')) return '마스코트';
            return '';
        }
        const lsRows = ls.items.map(({ key, bytes }) => {
            const desc = getDesc(key);
            return `<tr><td class="storage-key">${escapeHtml(key)}</td><td class="storage-desc">${escapeHtml(desc)}</td><td>${formatBytes(bytes)}</td></tr>`;
        }).join('');
        const ssRows = ss.items.map(({ key, bytes }) => {
            const desc = getDesc(key);
            return `<tr><td class="storage-key">${escapeHtml(key)}</td><td class="storage-desc">${escapeHtml(desc)}</td><td>${formatBytes(bytes)}</td></tr>`;
        }).join('');

        container.innerHTML = `
            <div class="user-layout">
                <div class="storage-summary">
                    <div class="storage-card">
                        <div class="storage-card-value">${formatBytes(totalBytes)}</div>
                        <div class="storage-card-label">총 저장 용량</div>
                    </div>
                    <div class="storage-card">
                        <div class="storage-card-value">${formatBytes(ls.totalBytes)}</div>
                        <div class="storage-card-label">localStorage (영구)</div>
                    </div>
                    <div class="storage-card">
                        <div class="storage-card-value">${formatBytes(ss.totalBytes)}</div>
                        <div class="storage-card-label">sessionStorage (탭 종료 시 삭제)</div>
                    </div>
                </div>
                <p style="font-size:var(--font-size-xs); color:var(--text-tertiary); margin-bottom:16px;">
                    브라우저별 localStorage 한도는 보통 5~10MB입니다. UTF-16 기준으로 키+값 길이×2 바이트로 계산합니다.
                </p>
                <div class="settings-section">
                    <h3>localStorage (${ls.items.length}개)</h3>
                    <div style="overflow-x:auto;">
                        <table class="storage-table">
                            <thead><tr><th>키</th><th>용도</th><th>크기</th></tr></thead>
                            <tbody>${lsRows || '<tr><td colspan="3" style="color:var(--text-tertiary);">비어 있음</td></tr>'}</tbody>
                        </table>
                    </div>
                </div>
                <div class="settings-section">
                    <h3>sessionStorage (${ss.items.length}개)</h3>
                    <div style="overflow-x:auto;">
                        <table class="storage-table">
                            <thead><tr><th>키</th><th>용도</th><th>크기</th></tr></thead>
                            <tbody>${ssRows || '<tr><td colspan="3" style="color:var(--text-tertiary);">비어 있음</td></tr>'}</tbody>
                        </table>
                    </div>
                </div>
                <div style="display:flex; justify-content:flex-end;">
                    <button type="button" class="btn-ghost" id="storageRefresh">🔄 새로고침</button>
                </div>
            </div>`;

        container.querySelector<HTMLButtonElement>('#storageRefresh')?.addEventListener('click', () => renderStorage(container));
    }

    function escapeHtml(s: string | null | undefined): string {
        if (!s) return '';
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function buildSettings(container: HTMLElement): void {
        Mdd.linePreset('tool_run', { msg: '설정 바꿀 거야?' });
        renderSettings(container);
    }

    function renderSettings(container: HTMLElement): void {
        const theme = Toolbox.getTheme?.() ?? 'dark';
        const prismTheme = Toolbox.getPrismTheme?.() ?? '';
        const prismThemes = Toolbox.getPrismThemes?.() ?? [];
        const bgTheme = Toolbox.getBgTheme?.() ?? '';
        const bgThemes = Toolbox.getBgThemes?.() ?? [];
        const apiUI = typeof Gemini !== 'undefined' ? Gemini.buildApiKeyUI('set') : { html: '' };

        container.innerHTML = `
            <div class="user-layout">
                <div class="settings-section">
                    <h3>🎨 표시</h3>
                    <div class="settings-row">
                        <label for="setTheme">테마</label>
                        <select id="setTheme" class="settings-control">
                            <option value="dark" ${theme === 'dark' ? 'selected' : ''}>다크</option>
                            <option value="light" ${theme === 'light' ? 'selected' : ''}>라이트</option>
                        </select>
                    </div>
                    <div class="settings-row">
                        <label for="setPrism">코드 하이라이트</label>
                        <select id="setPrism" class="settings-control">
                            ${prismThemes.map((t) => `<option value="${t.id}" ${t.id === prismTheme ? 'selected' : ''}>${t.label}</option>`).join('')}
                        </select>
                    </div>
                    <div class="settings-row">
                        <label for="setBgTheme">배경 테마</label>
                        <select id="setBgTheme" class="settings-control">
                            ${bgThemes.map((t) => `<option value="${t.id}" ${t.id === bgTheme ? 'selected' : ''}>${t.label}</option>`).join('')}
                        </select>
                    </div>
                    <div class="settings-code-preview">
                        <pre class="language-javascript"><code class="language-javascript">function hello() {
  const name = "World";
  return \`Hello, \${name}!\`;
}</code></pre>
                    </div>
                </div>
                <div class="settings-section">
                    <h3>🔑 API</h3>
                    ${apiUI.html}
                </div>
                <div class="settings-section">
                    <h3>⚠️ 위험 구역</h3>
                    <div class="settings-row settings-danger">
                        <label>유저 데이터 초기화</label>
                        <button type="button" class="btn btn-danger" id="setResetUser">🗑️ 초기화</button>
                    </div>
                    <div class="settings-row settings-danger">
                        <label>사용량 기록 초기화</label>
                        <button type="button" class="btn btn-danger" id="setResetUsage">🗑️ 초기화</button>
                    </div>
                </div>
            </div>`;

        container.querySelector<HTMLSelectElement>('#setTheme')?.addEventListener('change', (e: Event) => {
            const target = e.target as HTMLSelectElement | null;
            if (!target) return;
            (Toolbox as any).setTheme?.(target.value);
            (Toolbox as any).showToast?.('테마: ' + (target.value === 'dark' ? '다크' : '라이트'));
        });

        container.querySelector<HTMLSelectElement>('#setPrism')?.addEventListener('change', (e: Event) => {
            const target = e.target as HTMLSelectElement | null;
            if (!target) return;
            (Toolbox as any).setPrismTheme?.(target.value);
        });

        container.querySelector<HTMLSelectElement>('#setBgTheme')?.addEventListener('change', (e: Event) => {
            const target = e.target as HTMLSelectElement | null;
            if (!target) return;
            (Toolbox as any).setBgTheme?.(target.value);
            const label = bgThemes.find((t) => t.id === target.value)?.label || target.value;
            (Toolbox as any).showToast?.('배경: ' + label);
        });

        const previewCode = container.querySelector<HTMLElement>('.settings-code-preview code[class*="language-"]');
        if (previewCode && typeof Prism !== 'undefined') Prism.highlightElement(previewCode);

        if (typeof Gemini !== 'undefined') {
            Gemini.buildApiKeyUI('set').init(container);
        }

        container.querySelector<HTMLButtonElement>('#setResetUser')?.addEventListener('click', () => {
            if (!confirm('모든 도전과제, 뱃지, 진행도를 초기화합니다. 계속할까요?')) return;
            localStorage.removeItem('toolbox_user_data');
            (Toolbox as any).showToast?.('유저 데이터 초기화 완료');
            renderSettings(container);
        });

        container.querySelector<HTMLButtonElement>('#setResetUsage')?.addEventListener('click', () => {
            if (!confirm('모든 사용량 기록을 삭제합니다. 계속할까요?')) return;
            localStorage.removeItem('toolbox_usage_stats');
            (Toolbox as any).showToast?.('사용량 기록 초기화 완료');
        });
    }

    Toolbox.register({
        id: 'user',
        title: '내 정보',
        category: 'tool',
        desc: '사용량, 도전과제, 뱃지 등 내 정보를 확인합니다',
        hidden: true,
        layout: 'form',
        icon: '<circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M4 20c0-4 4-6 8-6s8 2 8 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" fill="none"/>',
        tabs: [
            { id: 'user-overview', label: '요약', build: buildOverview },
            { id: 'user-usage', label: '사용량', build: buildUsage },
            { id: 'user-achievements', label: '도전과제', build: buildAchievements },
            { id: 'user-streaks', label: '스트릭', build: buildStreaks },
            { id: 'user-badges', label: '뱃지', build: buildBadges },
            { id: 'user-storage', label: '저장소', build: buildStorage },
            { id: 'user-settings', label: '설정', build: buildSettings },
        ]
    });
})();
