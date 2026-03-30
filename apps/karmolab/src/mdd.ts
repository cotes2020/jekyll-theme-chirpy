/**
 * MDD (Moe Driven Development) — 마스코트 매니저 + 호감도/스토리 모듈
 *
 * 이미지 기반 마스코트 캐릭터, 12가지 감정 표현, 말풍선, 바운스,
 * 호감도 시스템, 스토리 이벤트를 관리합니다.
 * 티메토 대사는 `linePreset(id, { msg?, mood?, duration? })` + `LINE_PRESETS` 로 통일합니다.
 */
const Mdd = (() => {
    const POSES = ['idle','happy','sad','shock','think','sleep','angry','love','smug','eating','pointing','cheer'];
    const MASCOT_BASE = '/apps/karmolab/img/mascot';
    const IDLE_TIMEOUT = 30000;

    let currentMood = 'idle';
    let idleTimer: ReturnType<typeof setTimeout> | null = null;
    let container: HTMLDivElement | null = null;
    let charEl: HTMLDivElement | null = null;
    let bubbleEl: HTMLDivElement | null = null;
    let bubbleTimer: ReturnType<typeof setTimeout> | null = null;
    let _ready = false;

    /* ===== 이미지 마스코트 ===== */

    function getMascotImgSrc(mood: string): string {
        const valid = POSES.includes(mood) ? mood : 'idle';
        return `${MASCOT_BASE}/${valid}.png`;
    }

    /* ===== CSS 주입 ===== */

    function injectCSS(id: string, css: string): void {
        if (document.getElementById('mdd-css-' + id)) return;
        const style = document.createElement('style');
        style.id = 'mdd-css-' + id;
        style.textContent = css;
        (document.head || document.documentElement).appendChild(style);
    }

    /* ===== 감정/포즈 전환 ===== */

    function setMood(poseId: string): void {
        if (!POSES.includes(poseId)) poseId = 'idle';
        currentMood = poseId;
        if (!charEl) return;
        const img = charEl.querySelector('img');
        if (img) img.src = getMascotImgSrc(poseId);
        resetIdleTimer();
    }

    /* ===== 말풍선 ===== */

    function say(message: string, duration = 3000): void {
        const el = bubbleEl;
        if (!el) return;
        el.textContent = message;
        el.classList.add('visible');
        if (bubbleTimer !== null) clearTimeout(bubbleTimer);
        bubbleTimer = setTimeout(() => {
            el.classList.remove('visible');
        }, duration);
    }

    /* ===== 바운스 ===== */

    function bounce(): void {
        if (!charEl) return;
        charEl.classList.remove('mdd-bounce');
        void charEl.offsetWidth;
        charEl.classList.add('mdd-bounce');
    }

    /* ===== 티메토 감정별 대사 프리셋 (로드맵 기준) ===== */
    const LINE_PRESETS = {
        first_visit:   { mood: 'pointing', msg: '어서 오세요, 조수님! KarmoLab에 오신 걸 환영해요.' },
        daily_start:   { mood: 'happy',    msg: '조수님, 오늘의 실험 준비됐어요! 한 번 확인해볼래요?' },
        tool_run:      { mood: 'think',    msg: '측정 개시... 잠깐만요!' },
        success:       { mood: 'cheer',    msg: '샘플 확보! 연구 노트에 기록했어요.' },
        error:         { mood: 'sad',      msg: '장비가 잠깐 삐끗했어요... 다시 한 번만요!' },
        warn_data:     { mood: 'angry',    msg: '잠깐! 이건 중요한 데이터예요. 꼭 확인해주세요.' },
        idle_sleep:    { mood: 'sleep',    msg: 'zzZ... 조수님...?' },
        idle_wake:     { mood: 'shock',    msg: '앗! 돌아오셨군요!' },
        achievement:   { mood: 'love',     msg: '조수님 덕분에 연구소가 안정되고 있어요...!' },
        meme_done:     { mood: 'smug',     msg: '후후, 이건 명작이 될지도요?' },
        home_hub:      { mood: 'happy',    msg: '조수님, 연구소 허브예요! 자주 쓰는 장비는 즐겨찾기에 모아 두었어요.' },
        measure_done:  { mood: 'cheer',    msg: '측정 완료! 수치는 연구 노트에 반영했어요.' },
    };

    /**
     * 프리셋 대사 + 포즈. opts.msg / opts.mood / opts.duration 으로 덮어쓸 수 있음.
     * @returns {boolean} 알려진 id면 true
     */
    function linePreset(id: string, opts?: { msg?: string; mood?: string; duration?: number }): boolean {
        const base = (LINE_PRESETS as Record<string, { mood: string; msg: string } | undefined>)[id];
        if (!base) return false;
        const mood = (opts && opts.mood) || base.mood;
        const msg = (opts && opts.msg != null) ? opts.msg : base.msg;
        const duration = (opts && opts.duration != null) ? opts.duration : 3000;
        setMood(mood);
        say(msg, duration);
        return true;
    }

    const TAP_QUIPS = [
        '조수님, 저를 부르셨나요?',
        '실험 데이터... 아, 장난이에요.',
        '잠깐 쉬었다 갈게요~',
        '저도 측정 한번 해볼까요?',
        '조수님 손길, 기록해 둘게요.',
        '히히, 오늘 기분 좋아요.',
        '다음 실험은 뭘까요?',
        '연구소 안전 점검... 통과예요!',
    ];

    function resetIdleTimer(): void {
        if (currentMood === 'sleep') return;
        if (idleTimer !== null) clearTimeout(idleTimer);
        idleTimer = setTimeout(() => { linePreset('idle_sleep', { duration: 4000 }); }, IDLE_TIMEOUT);
    }

    /* ===== 호감도 시스템 ===== */

    const AFFECTION_KEY = 'mdd_affection';
    const STORY_KEY = 'mdd_story_progress';
    const STORY_LOG_KEY = 'mdd_story_log';
    const GUIDE_SEEN_KEY = 'mdd_guide_seen';
    const POSITION_KEY = 'mdd_position';

    function getAffection(): number {
        try { return parseInt(localStorage.getItem(AFFECTION_KEY) || '0', 10) || 0; } catch (_) { return 0; }
    }

    function addAffection(amount: number): number {
        const current = getAffection() + amount;
        try { localStorage.setItem(AFFECTION_KEY, String(current)); } catch (_) {}
        checkStoryMilestone(current);
        return current;
    }

    function getStoryProgress(): { seen: string[]; chapter: number } {
        try { return JSON.parse(localStorage.getItem(STORY_KEY) as string) || { seen: [], chapter: 0 }; }
        catch (_) { return { seen: [], chapter: 0 }; }
    }

    function saveStoryProgress(data: { seen: string[]; chapter: number }): void {
        try { localStorage.setItem(STORY_KEY, JSON.stringify(data)); } catch (_) {}
    }

    /* 티메토 온보딩 가이드 (짧은 순서) */
    const GUIDE_MESSAGES = [
        { id: 'welcome',  msg: LINE_PRESETS.first_visit.msg },
        { id: 'drag',     msg: '저를 드래그해서 연구소 구석구석, 편한 자리로 옮길 수 있어요.' },
        { id: 'click',    msg: '가끔 눌러 주시면 반응 샘플이 쌓여요. 우클릭은 스토리 로그예요!' },
    ];

    const STORY_EVENTS = [
        { threshold: 0,    id: 'intro',     mood: 'pointing',  msg: '처음 뵙겠어요, 조수님. 실험 참여 감사드려요!' },
        { threshold: 10,   id: 'curious',   mood: 'think',     msg: '자주 오시네요... 좋은 데이터가 쌓이고 있어요.' },
        { threshold: 30,   id: 'comfort',   mood: 'happy',     msg: '이제 조수님 손길이 익숙해졌어요. 안심하고 맡겨 주세요.' },
        { threshold: 50,   id: 'trust',     mood: 'smug',      msg: '다른 분들과는 뭔가 달라요... 인정할게요.' },
        { threshold: 100,  id: 'friend',    mood: 'love',      msg: '솔직히... 조수님 오시는 날이 기다려졌어요.' },
        { threshold: 200,  id: 'partner',   mood: 'cheer',     msg: '이제 공식 파트너예요! 앞으로도 실험 같이 해요!' },
        { threshold: 500,  id: 'soulmate',  mood: 'love',      msg: '연구소가 집 같아요. 조수님이 계셔서 그래요.' },
    ];

    function appendStoryLog(entry: { id: string; msg: string; mood: string; ts: number }): void {
        try {
            const log = getStoryLog();
            log.push(entry);
            localStorage.setItem(STORY_LOG_KEY, JSON.stringify(log));
        } catch (_) {}
    }

    function getStoryLog(): Array<{ id?: string; msg: string; mood?: string; ts?: number }> {
        try { return JSON.parse(localStorage.getItem(STORY_LOG_KEY) as string) || []; }
        catch (_) { return []; }
    }

    function checkStoryMilestone(affection: number): void {
        const progress = getStoryProgress();
        for (const event of STORY_EVENTS) {
            if (affection >= event.threshold && !progress.seen.includes(event.id)) {
                progress.seen.push(event.id);
                progress.chapter = Math.max(progress.chapter, STORY_EVENTS.indexOf(event));
                saveStoryProgress(progress);
                appendStoryLog({ id: event.id, msg: event.msg, mood: event.mood, ts: Date.now() });
                setTimeout(() => {
                    setMood(event.mood);
                    say(event.msg, 5000);
                    bounce();
                    if (event.id === 'intro') setTimeout(showNextGuide, 5500);
                }, 500);
                break;
            }
        }
    }

    /* 안내 대사 표시 (최소 세트) */
    function showGuide(id: string): void {
        const g = GUIDE_MESSAGES.find(x => x.id === id);
        if (!g) return;
        if (id === 'welcome') {
            linePreset('first_visit', { duration: 4000 });
            return;
        }
        linePreset('tool_run', { msg: g.msg, duration: 4000 });
    }

    function showNextGuide(): void {
        try {
            const seen = JSON.parse(localStorage.getItem(GUIDE_SEEN_KEY) as string) || [];
            const next = GUIDE_MESSAGES.find(g => !seen.includes(g.id));
            if (next) {
                seen.push(next.id);
                localStorage.setItem(GUIDE_SEEN_KEY, JSON.stringify(seen));
                showGuide(next.id);
            }
        } catch (_) {}
    }

    function getRelationshipTitle(): string {
        const affection = getAffection();
        if (affection >= 500) return '소울메이트';
        if (affection >= 200) return '파트너';
        if (affection >= 100) return '친구';
        if (affection >= 50) return '지인';
        if (affection >= 30) return '아는 사이';
        if (affection >= 10) return '관심';
        return '낯선 사람';
    }

    /* ===== 드래그 ===== */

    function loadPosition(): void {
        if (!container) return;
        try {
            const s = localStorage.getItem(POSITION_KEY);
            if (s) {
                const { left, top } = JSON.parse(s) as { left: number; top: number };
                if (typeof left === 'number' && typeof top === 'number') {
                    container.style.left = left + 'px';
                    container.style.top = top + 'px';
                    container.style.bottom = 'auto';
                    container.style.right = 'auto';
                }
            }
        } catch (_) {}
    }

    function savePosition(): void {
        if (!container) return;
        const rect = container.getBoundingClientRect();
        try {
            localStorage.setItem(POSITION_KEY, JSON.stringify({ left: rect.left, top: rect.top }));
        } catch (_) {}
    }

    function initDrag(): void {
        if (!charEl || !container) return;
        const el = charEl;
        const box = container;
        let dragStart: { x: number; y: number; left: number; top: number } | null = null;
        const DRAG_THRESHOLD = 8;

        const onDown = (e: PointerEvent) => {
            dragStart = { x: e.clientX, y: e.clientY, left: box.offsetLeft, top: box.offsetTop };
            const rect = box.getBoundingClientRect();
            if (box.style.left) {
                dragStart.left = rect.left;
                dragStart.top = rect.top;
            } else {
                dragStart.left = window.innerWidth - rect.width - 16;
                dragStart.top = window.innerHeight - rect.height - 16;
            }
            box.style.left = dragStart.left + 'px';
            box.style.top = dragStart.top + 'px';
            box.style.bottom = 'auto';
            box.style.right = 'auto';
            document.addEventListener('pointermove', onMove);
            document.addEventListener('pointerup', onUp, { once: true });
            document.addEventListener('pointercancel', onUp, { once: true });
        };

        const onMove = (e: PointerEvent) => {
            if (!dragStart) return;
            const dx = e.clientX - dragStart.x;
            const dy = e.clientY - dragStart.y;
            let left = dragStart.left + dx;
            let top = dragStart.top + dy;
            const maxLeft = window.innerWidth - box.offsetWidth;
            const maxTop = window.innerHeight - box.offsetHeight;
            left = Math.max(0, Math.min(left, maxLeft));
            top = Math.max(0, Math.min(top, maxTop));
            box.style.left = left + 'px';
            box.style.top = top + 'px';
        };

        const onUp = (e: PointerEvent) => {
            document.removeEventListener('pointermove', onMove);
            document.removeEventListener('pointerup', onUp);
            document.removeEventListener('pointercancel', onUp);
            if (!dragStart) return;
            const moved = Math.abs(e.clientX - dragStart.x) + Math.abs(e.clientY - dragStart.y);
            if (moved >= DRAG_THRESHOLD) {
                savePosition();
            } else {
                bounce();
                addAffection(1);
                const q = TAP_QUIPS[Math.floor(Math.random() * TAP_QUIPS.length)];
                say(q);
                setMood(['happy', 'smug', 'love', 'idle'][Math.floor(Math.random() * 4)]);
            }
            dragStart = null;
        };

        el.addEventListener('pointerdown', (e) => { e.preventDefault(); onDown(e); });
        el.addEventListener('dragstart', (e) => e.preventDefault());
        el.querySelector('img')?.addEventListener('dragstart', (e) => e.preventDefault());
        el.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            openStoryLog();
        });
    }

    /* ===== 스토리 로그 (미연시) ===== */
    function openStoryLog(): void {
        const log = getStoryLog();
        const overlay = document.createElement('div');
        overlay.className = 'mdd-log-overlay';
        overlay.innerHTML = `
            <div class="mdd-log-panel">
                <div class="mdd-log-header">
                    <h3>스토리 로그</h3>
                    <button class="mdd-log-close" type="button" aria-label="닫기">×</button>
                </div>
                <div class="mdd-log-body">
                    ${log.length ? log.map((e: { msg: string }) => `
                        <div class="mdd-log-entry">
                            <span class="mdd-log-msg">${escapeHtml(e.msg)}</span>
                        </div>
                    `).join('') : '<p class="mdd-log-empty">아직 기록된 스토리가 없어요, 조수님.</p>'}
                </div>
            </div>
        `;
        injectCSS('mdd-log', `
            .mdd-log-overlay { position:fixed; inset:0; z-index:9999; background:rgba(0,0,0,0.6); backdrop-filter:blur(4px); display:flex; align-items:center; justify-content:center; padding:16px; }
            .mdd-log-panel { background:var(--bg-secondary,#1a1a1e); border:1px solid var(--border,rgba(255,255,255,0.08)); max-width:360px; width:100%; max-height:70vh; display:flex; flex-direction:column; border-radius:8px; box-shadow:0 8px 32px rgba(0,0,0,0.4); }
            .mdd-log-header { display:flex; align-items:center; justify-content:space-between; padding:12px 16px; border-bottom:1px solid var(--border); }
            .mdd-log-header h3 { margin:0; font-size:14px; font-weight:600; color:var(--text-primary); }
            .mdd-log-close { background:none; border:none; color:var(--text-tertiary); font-size:24px; cursor:pointer; padding:0 4px; line-height:1; }
            .mdd-log-close:hover { color:var(--text-primary); }
            .mdd-log-body { overflow-y:auto; padding:12px; }

            .mdd-log-entry { padding:8px 0; border-bottom:1px solid var(--border); font-size:13px; line-height:1.5; color:var(--text-secondary); }
            .mdd-log-entry:last-child { border-bottom:none; }
            .mdd-log-empty { color:var(--text-tertiary); font-size:13px; text-align:center; padding:24px; margin:0; }
        `);
        const close = () => {
            overlay.remove();
            document.removeEventListener('keydown', onEsc);
        };
        const onEsc = (e: KeyboardEvent) => { if (e.key === 'Escape') close(); };
        const closeBtn = overlay.querySelector('.mdd-log-close');
        if (closeBtn) (closeBtn as HTMLElement).onclick = close;
        overlay.onclick = (e: MouseEvent) => { if (e.target === overlay) close(); };
        const panel = overlay.querySelector('.mdd-log-panel');
        if (panel) (panel as HTMLElement).onclick = (e: MouseEvent) => e.stopPropagation();
        document.addEventListener('keydown', onEsc);
        document.body.appendChild(overlay);
    }

    function escapeHtml(s: string): string {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    /* ===== DOM 초기화 ===== */

    function init(): void {
        if (_ready) return;
        _ready = true;

        injectCSS('mdd-core', `
            .mdd-container { position:fixed; bottom:16px; right:16px; z-index:900; display:flex; flex-direction:column; align-items:flex-end; pointer-events:none; }
            .mdd-bubble { background:rgba(8,16,30,0.85); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); border:1px solid rgba(0,229,255,0.12); color:var(--text-primary,#e4eaf6); padding:6px 10px; font-size:var(--font-size-xs); max-width:180px; text-align:left; line-height:1.4; margin-bottom:6px; opacity:0; transform:translateY(4px); transition:opacity 0.2s,transform 0.2s; pointer-events:auto; font-family:var(--font-sans,'Pretendard',sans-serif); box-shadow:0 0 12px rgba(0,229,255,0.06),0 4px 16px rgba(0,0,0,0.3); border-radius:var(--radius-md,8px); }
            .mdd-bubble.visible { opacity:1; transform:translateY(0); }
            .mdd-char { width:96px; height:105px; pointer-events:auto; cursor:grab; touch-action:none; transition:transform 0.15s,filter 0.3s; user-select:none; -webkit-user-select:none; opacity:0.85; }
            .mdd-char:active { cursor:grabbing; }
            .mdd-char:hover { transform:scale(1.05); opacity:1; filter:drop-shadow(0 0 8px rgba(0,229,255,0.25)); }
            .mdd-char img { width:100%; height:100%; object-fit:contain; display:block; pointer-events:none; -webkit-user-drag:none; user-drag:none; }
            .mdd-bounce { animation:mdd-bounce 0.3s ease; }
            @keyframes mdd-bounce { 0%,100%{transform:translateY(0)} 40%{transform:translateY(-10px)} 70%{transform:translateY(-3px)} }
            @media(max-width:768px){ .mdd-container{bottom:8px;right:8px} .mdd-char{width:72px;height:79px} .mdd-bubble{font-size:var(--font-size-2xs);max-width:140px;padding:5px 8px} }
        `);

        container = document.createElement('div');
        container.className = 'mdd-container';

        bubbleEl = document.createElement('div');
        bubbleEl.className = 'mdd-bubble';
        container.appendChild(bubbleEl);

        charEl = document.createElement('div');
        charEl.className = 'mdd-char';
        charEl.innerHTML = `<img src="${getMascotImgSrc('idle')}" alt="마스코트" draggable="false">`;
        container.appendChild(charEl);
        document.body.appendChild(container);

        loadPosition();
        initDrag();

        resetIdleTimer();
        document.addEventListener('mousemove', () => { if (currentMood === 'sleep') { linePreset('idle_wake', { duration: 3500 }); } resetIdleTimer(); });
        document.addEventListener('keydown', () => { if (currentMood === 'sleep') { linePreset('idle_wake', { duration: 3500 }); } resetIdleTimer(); });

        addAffection(1);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    return {
        setMood, say, bounce, injectCSS,
        linePreset, LINE_PRESETS,
        addAffection, getAffection, getRelationshipTitle,
        getStoryProgress, getStoryLog, STORY_EVENTS,
        showGuide, showNextGuide, openStoryLog, GUIDE_MESSAGES,
    };
})();
