/**
 * MDD (Moe Driven Development) — 마스코트 매니저 + 호감도/스토리 모듈
 *
 * 이미지 기반 마스코트 캐릭터, 12가지 감정 표현, 말풍선, 바운스,
 * 호감도 시스템, 스토리 이벤트를 관리합니다.
 */
const Mdd = (() => {
    const POSES = ['idle','happy','sad','shock','think','sleep','angry','love','smug','eating','pointing','cheer'];
    const MASCOT_BASE = '/apps/karmolab/img/mascot';
    const IDLE_TIMEOUT = 30000;

    let currentMood = 'idle';
    let idleTimer = null;
    let container = null;
    let charEl = null;
    let bubbleEl = null;
    let bubbleTimer = null;
    let _ready = false;

    /* ===== 이미지 마스코트 ===== */

    function getMascotImgSrc(mood) {
        const valid = POSES.includes(mood) ? mood : 'idle';
        return `${MASCOT_BASE}/${valid}.png`;
    }

    /* ===== CSS 주입 ===== */

    function injectCSS(id, css) {
        if (document.getElementById('mdd-css-' + id)) return;
        const style = document.createElement('style');
        style.id = 'mdd-css-' + id;
        style.textContent = css;
        (document.head || document.documentElement).appendChild(style);
    }

    /* ===== 감정/포즈 전환 ===== */

    function setMood(poseId) {
        if (!POSES.includes(poseId)) poseId = 'idle';
        currentMood = poseId;
        if (!charEl) return;
        const img = charEl.querySelector('img');
        if (img) img.src = getMascotImgSrc(poseId);
        resetIdleTimer();
    }

    /* ===== 말풍선 ===== */

    function say(message, duration = 3000) {
        if (!bubbleEl) return;
        bubbleEl.textContent = message;
        bubbleEl.classList.add('visible');
        clearTimeout(bubbleTimer);
        bubbleTimer = setTimeout(() => bubbleEl.classList.remove('visible'), duration);
    }

    /* ===== 바운스 ===== */

    function bounce() {
        if (!charEl) return;
        charEl.classList.remove('mdd-bounce');
        void charEl.offsetWidth;
        charEl.classList.add('mdd-bounce');
    }

    /* ===== 타이머 ===== */

    function resetIdleTimer() {
        if (currentMood === 'sleep') return;
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => { setMood('sleep'); say('zzz...'); }, IDLE_TIMEOUT);
    }

    /* ===== 호감도 시스템 ===== */

    const AFFECTION_KEY = 'mdd_affection';
    const STORY_KEY = 'mdd_story_progress';
    const STORY_LOG_KEY = 'mdd_story_log';
    const GUIDE_SEEN_KEY = 'mdd_guide_seen';
    const POSITION_KEY = 'mdd_position';

    function getAffection() {
        try { return parseInt(localStorage.getItem(AFFECTION_KEY)) || 0; } catch (_) { return 0; }
    }

    function addAffection(amount) {
        const current = getAffection() + amount;
        try { localStorage.setItem(AFFECTION_KEY, current); } catch (_) {}
        checkStoryMilestone(current);
        return current;
    }

    function getStoryProgress() {
        try { return JSON.parse(localStorage.getItem(STORY_KEY)) || { seen: [], chapter: 0 }; }
        catch (_) { return { seen: [], chapter: 0 }; }
    }

    function saveStoryProgress(data) {
        try { localStorage.setItem(STORY_KEY, JSON.stringify(data)); } catch (_) {}
    }

    /* 티메토 안내 대사 최소 세트 */
    const GUIDE_MESSAGES = [
        { id: 'welcome',  msg: '이곳은 KarmoLab이다.' },
        { id: 'drag',     msg: '드래그해서 옮길 수 있다.' },
        { id: 'click',    msg: '클릭하면 반응할지도.' },
    ];

    const STORY_EVENTS = [
        { threshold: 0,    id: 'intro',     mood: 'idle',     msg: '처음 보는 얼굴이다. 잘 부탁할지도.' },
        { threshold: 10,   id: 'curious',   mood: 'think',    msg: '자주 오네? 좋은 취향일지도~' },
        { threshold: 30,   id: 'comfort',   mood: 'happy',    msg: '이제 좀 편해졌다. 집사 자격 인정할지도!' },
        { threshold: 50,   id: 'trust',     mood: 'smug',     msg: '다른 인간들과는 다르네... 칭찬할지도.' },
        { threshold: 100,  id: 'friend',    mood: 'love',     msg: '사실... 여기 오는 거 기다리고 있었다. 비밀일지도!' },
        { threshold: 200,  id: 'partner',   mood: 'cheer',    msg: '우리 파트너다! 앞으로도 함께할지도!' },
        { threshold: 500,  id: 'soulmate',  mood: 'love',     msg: '여기가 내 집이다. 네가 있으니까 그럴지도.' },
    ];

    function appendStoryLog(entry) {
        try {
            const log = getStoryLog();
            log.push(entry);
            localStorage.setItem(STORY_LOG_KEY, JSON.stringify(log));
        } catch (_) {}
    }

    function getStoryLog() {
        try { return JSON.parse(localStorage.getItem(STORY_LOG_KEY)) || []; }
        catch (_) { return []; }
    }

    function checkStoryMilestone(affection) {
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
    function showGuide(id) {
        const g = GUIDE_MESSAGES.find(x => x.id === id);
        if (g) { setMood('pointing'); say(g.msg, 4000); }
    }

    function showNextGuide() {
        try {
            const seen = JSON.parse(localStorage.getItem(GUIDE_SEEN_KEY)) || [];
            const next = GUIDE_MESSAGES.find(g => !seen.includes(g.id));
            if (next) {
                seen.push(next.id);
                localStorage.setItem(GUIDE_SEEN_KEY, JSON.stringify(seen));
                showGuide(next.id);
            }
        } catch (_) {}
    }

    function getRelationshipTitle() {
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

    function loadPosition() {
        try {
            const s = localStorage.getItem(POSITION_KEY);
            if (s) {
                const { left, top } = JSON.parse(s);
                if (typeof left === 'number' && typeof top === 'number') {
                    container.style.left = left + 'px';
                    container.style.top = top + 'px';
                    container.style.bottom = 'auto';
                    container.style.right = 'auto';
                }
            }
        } catch (_) {}
    }

    function savePosition() {
        if (!container) return;
        const rect = container.getBoundingClientRect();
        try {
            localStorage.setItem(POSITION_KEY, JSON.stringify({ left: rect.left, top: rect.top }));
        } catch (_) {}
    }

    function initDrag() {
        let dragStart = null;
        const DRAG_THRESHOLD = 8;

        const onDown = (e) => {
            dragStart = { x: e.clientX, y: e.clientY, left: container.offsetLeft, top: container.offsetTop };
            const rect = container.getBoundingClientRect();
            if (container.style.left) {
                dragStart.left = rect.left;
                dragStart.top = rect.top;
            } else {
                dragStart.left = window.innerWidth - rect.width - 16;
                dragStart.top = window.innerHeight - rect.height - 16;
            }
            container.style.left = dragStart.left + 'px';
            container.style.top = dragStart.top + 'px';
            container.style.bottom = 'auto';
            container.style.right = 'auto';
            document.addEventListener('pointermove', onMove);
            document.addEventListener('pointerup', onUp, { once: true });
            document.addEventListener('pointercancel', onUp, { once: true });
        };

        const onMove = (e) => {
            if (!dragStart) return;
            const dx = e.clientX - dragStart.x;
            const dy = e.clientY - dragStart.y;
            let left = dragStart.left + dx;
            let top = dragStart.top + dy;
            const maxLeft = window.innerWidth - container.offsetWidth;
            const maxTop = window.innerHeight - container.offsetHeight;
            left = Math.max(0, Math.min(left, maxLeft));
            top = Math.max(0, Math.min(top, maxTop));
            container.style.left = left + 'px';
            container.style.top = top + 'px';
        };

        const onUp = (e) => {
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
                const quips = ['뭐지?','뭐 볼 거 없어.','만지지 마...','...(그르릉)','히힛... 그럴지도~','좋아...','뭐 하는 거지?','배고파...'];
                say(quips[Math.floor(Math.random() * quips.length)]);
                setMood(['happy','smug','love','idle'][Math.floor(Math.random() * 4)]);
            }
            dragStart = null;
        };

        charEl.addEventListener('pointerdown', (e) => { e.preventDefault(); onDown(e); });
        charEl.addEventListener('dragstart', (e) => e.preventDefault());
        charEl.querySelector('img')?.addEventListener('dragstart', (e) => e.preventDefault());
        charEl.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            openStoryLog();
        });
    }

    /* ===== 스토리 로그 (미연시) ===== */
    function openStoryLog() {
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
                    ${log.length ? log.map(e => `
                        <div class="mdd-log-entry">
                            <span class="mdd-log-msg">${escapeHtml(e.msg)}</span>
                        </div>
                    `).join('') : '<p class="mdd-log-empty">아직 기록이 없다.</p>'}
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
        const onEsc = (e) => { if (e.key === 'Escape') close(); };
        overlay.querySelector('.mdd-log-close').onclick = close;
        overlay.onclick = (e) => { if (e.target === overlay) close(); };
        overlay.querySelector('.mdd-log-panel').onclick = (e) => e.stopPropagation();
        document.addEventListener('keydown', onEsc);
        document.body.appendChild(overlay);
    }

    function escapeHtml(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    /* ===== DOM 초기화 ===== */

    function init() {
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
        document.addEventListener('mousemove', () => { if (currentMood === 'sleep') { setMood('idle'); } resetIdleTimer(); });
        document.addEventListener('keydown', () => { if (currentMood === 'sleep') { setMood('idle'); } resetIdleTimer(); });

        addAffection(1);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    return {
        setMood, say, bounce, injectCSS,
        addAffection, getAffection, getRelationshipTitle,
        getStoryProgress, getStoryLog, STORY_EVENTS,
        showGuide, showNextGuide, openStoryLog, GUIDE_MESSAGES,
    };
})();
