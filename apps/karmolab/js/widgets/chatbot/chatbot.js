(function () {
    /* ===== CSS 주입 ===== */
    Mdd.injectCSS('chatbot', `
        .cb-outer { display:flex; flex-direction:column; flex:1; min-height:0; min-width:0; width:100%; position:relative; }
        .cb-layout { display:flex; gap:14px; flex:1; min-height:0; align-items:stretch; }

        /* 좌·우 패널 공통 */
        .cb-sidebar { display:flex; flex-direction:column; min-height:0; border:1px solid var(--border); background:var(--bg-secondary); border-radius:var(--radius-lg); overflow:hidden; box-shadow:0 1px 0 rgba(0,0,0,0.04); }
        .cb-sidebar-left { width:220px; flex-shrink:0; }
        .cb-sidebar-right { width:292px; flex-shrink:0; }
        .cb-panel-heading { font-size:var(--font-size-2xs); font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-tertiary); margin:0 0 10px; }
        .cb-sidebar-header { padding:14px 14px 12px; border-bottom:1px solid var(--border); flex-shrink:0; }
        .cb-sidebar-header .cb-panel-heading { margin-bottom:8px; }

        .cb-api-row { display:flex; gap:6px; align-items:center; }
        .cb-api-row input { flex:1; font-size:var(--font-size-xs); padding:6px 8px; }
        .cb-key-status { font-size:var(--font-size-2xs); color:var(--text-tertiary); margin-top:4px; }

        .cb-model-label { font-size:var(--font-size-xs); font-weight:500; color:var(--text-secondary); margin:12px 0 4px; display:block; }

        .cb-options { padding:12px 14px; flex-shrink:0; border-bottom:1px solid var(--border); }
        .cb-option-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
        .cb-option-row label { font-size:var(--font-size-xs); color:var(--text-secondary); }
        .cb-toggle { position:relative; width:36px; height:20px; }
        .cb-toggle input { opacity:0; width:0; height:0; }
        .cb-toggle-slider {
            position:absolute; inset:0; background:var(--bg-active); border-radius:10px;
            cursor:pointer; transition:background 0.2s;
        }
        .cb-toggle-slider::before {
            content:''; position:absolute; width:16px; height:16px; border-radius:50%;
            background:#fff; left:2px; top:2px; transition:transform 0.2s;
        }
        .cb-toggle input:checked + .cb-toggle-slider { background:var(--accent); }
        .cb-toggle input:checked + .cb-toggle-slider::before { transform:translateX(16px); }

        /* 시스템 프롬프트 (우측 패널 하단) */
        .cb-sysprompt { flex:1; min-height:120px; padding:12px 14px; overflow-y:auto; display:flex; flex-direction:column; border-top:1px solid var(--border); }
        .cb-sysprompt label { font-size:var(--font-size-xs); font-weight:500; color:var(--text-secondary); margin-bottom:4px; display:block; }
        .cb-sysprompt textarea {
            flex:1; min-height:120px; font-size:var(--font-size-xs); line-height:1.5;
            resize:none; padding:8px; background:var(--bg-tertiary);
        }
        .cb-sysprompt textarea:read-only { opacity:0.72; cursor:default; }

        /* 중앙: 좁은 채팅 열 */
        .cb-chat-stage { flex:1; min-width:0; display:flex; justify-content:center; align-items:stretch; padding:0 4px; }
        .cb-chat {
            flex:0 1 540px; width:100%; max-width:540px;
            display:flex; flex-direction:column;
            background:var(--bg-primary); border:1px solid var(--border);
            border-radius:var(--radius-lg); overflow:hidden;
            box-shadow:0 2px 12px rgba(0,0,0,0.06);
        }

        .cb-chat-header {
            padding:10px 14px; border-bottom:1px solid var(--border);
            display:flex; align-items:center; justify-content:space-between; gap:8px;
            background:var(--bg-secondary); flex-shrink:0;
        }
        .cb-chat-header-title { font-size:var(--font-size-sm); font-weight:600; color:var(--text-primary); min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .cb-chat-header-actions { display:flex; gap:4px; flex-wrap:wrap; justify-content:flex-end; }

        .cb-messages {
            flex:1; min-height:0; overflow-y:auto; padding:14px; display:flex;
            flex-direction:column; gap:10px;
        }

        .cb-msg { max-width:80%; padding:10px 14px; border-radius:14px; font-size:var(--font-size-sm); line-height:1.6; animation:fadeIn 0.3s ease; word-break:break-word; }
        .cb-msg-user {
            align-self:flex-end; background:var(--accent); color:#fff;
            border-bottom-right-radius:4px; white-space:pre-wrap;
        }
        .cb-msg-bot {
            align-self:flex-start; background:var(--bg-tertiary); color:var(--text-primary);
            border:1px solid var(--border); border-bottom-left-radius:4px;
        }
        .cb-msg-error { background:var(--error-subtle) !important; border-color:var(--error) !important; color:var(--error) !important; }

        /* 마크다운 렌더링 */
        .cb-msg-bot p { margin:0 0 8px; } .cb-msg-bot p:last-child { margin:0; }
        .cb-msg-bot strong { font-weight:700; }
        .cb-msg-bot em { font-style:italic; }
        .cb-msg-bot code { background:rgba(255,255,255,0.08); padding:1px 5px; border-radius:3px; font-family:'Cascadia Code','Consolas',monospace; font-size:var(--font-size-xs); }
        .cb-msg-bot pre { border:1px solid var(--border); border-radius:var(--radius-sm); padding:0; margin:8px 0; overflow-x:auto; position:relative; }
        .cb-msg-bot pre code { background:none !important; padding:10px 12px; font-size:var(--font-size-xs); line-height:1.5; display:block; }
        .cb-code-header { display:flex; justify-content:space-between; align-items:center; padding:4px 8px 4px 12px; background:rgba(255,255,255,0.04); border-bottom:1px solid var(--border); font-size:var(--font-size-2xs); }
        .cb-code-lang { color:var(--text-tertiary); font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
        .cb-msg-bot ul, .cb-msg-bot ol { margin:6px 0; padding-left:20px; }
        .cb-msg-bot li { margin:2px 0; }
        .cb-msg-bot a { color:var(--accent); text-decoration:underline; }
        .cb-msg-bot blockquote { border-left:3px solid var(--accent); padding-left:10px; margin:6px 0; color:var(--text-secondary); }
        .cb-msg-bot hr { border:none; border-top:1px solid var(--border); margin:8px 0; }
        .cb-msg-bot table { border-collapse:collapse; margin:8px 0; font-size:var(--font-size-xs); width:100%; }
        .cb-msg-bot th, .cb-msg-bot td { border:1px solid var(--border); padding:4px 8px; text-align:left; }
        .cb-msg-bot th { background:var(--bg-primary); font-weight:600; }
        .cb-msg-bot tr:nth-child(even) td { background:rgba(255,255,255,0.02); }

        .cb-cursor-blink { animation:cbCursorBlink 0.8s step-end infinite; color:var(--accent); }
        @keyframes cbCursorBlink { 0%,100%{opacity:1} 50%{opacity:0} }

        .cb-stop-btn { background:var(--error); color:#fff; border:none; border-radius:var(--radius-md); cursor:pointer; font-size:var(--font-size-xs); padding:6px 14px; font-family:inherit; font-weight:600; }

        .cb-typing span {
            display:inline-block; width:5px; height:5px; background:var(--text-tertiary);
            border-radius:50%; animation:cbTyping 1.4s infinite ease-in-out both; margin:0 1px;
        }
        .cb-typing span:nth-child(1) { animation-delay:-0.32s; }
        .cb-typing span:nth-child(2) { animation-delay:-0.16s; }
        @keyframes cbTyping { 0%,80%,100%{transform:scale(0)} 40%{transform:scale(1)} }

        /* 입력 영역 */
        .cb-input-area { padding:12px 16px; border-top:1px solid var(--border); background:var(--bg-secondary); }
        .cb-input-row { display:flex; gap:8px; }
        .cb-input-row textarea { flex:1; min-height:40px; max-height:80px; resize:none; font-size:var(--font-size-sm); padding:8px 12px; }
        .cb-send-btn {
            width:40px; background:var(--accent); color:#fff; border:none;
            border-radius:var(--radius-md); cursor:pointer; font-size:16px;
            display:flex; align-items:center; justify-content:center;
            transition:background var(--transition);
        }
        .cb-send-btn:hover { background:var(--accent-hover); }
        .cb-token-bar { display:flex; justify-content:space-between; margin-top:6px; font-size:var(--font-size-2xs); color:var(--text-tertiary); font-family:monospace; }

        .cb-mic-btn {
            width:40px; background:transparent; color:var(--text-secondary); border:1px solid var(--border);
            border-radius:var(--radius-md); cursor:pointer; font-size:16px;
            display:flex; align-items:center; justify-content:center; transition:all var(--transition);
        }
        .cb-mic-btn:hover { border-color:var(--accent); color:var(--accent); }
        .cb-mic-btn.recording { background:var(--error); color:#fff; border-color:var(--error); animation:cbMicPulse 1s infinite; }
        @keyframes cbMicPulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

        .cb-attach-area { display:flex; gap:6px; align-items:center; margin-bottom:4px; flex-wrap:wrap; }
        .cb-attach-thumb { width:48px; height:48px; object-fit:cover; border-radius:var(--radius-sm); border:1px solid var(--border); position:relative; }
        .cb-attach-wrap { position:relative; display:inline-block; }
        .cb-attach-remove { position:absolute; top:-4px; right:-4px; width:16px; height:16px; border-radius:50%; background:var(--error); color:#fff; border:none; font-size:var(--font-size-2xs); line-height:16px; text-align:center; cursor:pointer; padding:0; }
        .cb-attach-btn { background:none; border:1px dashed var(--border); border-radius:var(--radius-sm); width:40px; height:40px; font-size:18px; cursor:pointer; color:var(--text-tertiary); display:flex; align-items:center; justify-content:center; }
        .cb-attach-btn:hover { border-color:var(--accent); color:var(--accent); }
        .cb-input-area.drag-over { outline:2px dashed var(--accent); outline-offset:-2px; border-radius:var(--radius-md); }

        .cb-session-bar { display:flex; gap:2px; padding:6px 16px; border-bottom:1px solid var(--border); background:var(--bg-secondary); overflow-x:auto; flex-shrink:0; }
        .cb-session-tab { display:flex; align-items:center; gap:4px; padding:4px 10px; font-size:var(--font-size-xs); border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--bg-tertiary); color:var(--text-secondary); cursor:pointer; white-space:nowrap; font-family:inherit; transition:all var(--transition); }
        .cb-session-tab:hover { border-color:var(--accent); color:var(--text-primary); }
        .cb-session-tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
        .cb-session-tab-name { max-width:80px; overflow:hidden; text-overflow:ellipsis; cursor:default; }
        .cb-session-tab-edit { border:1px solid var(--accent); border-radius:2px; font-size:var(--font-size-xs); padding:0 4px; width:70px; font-family:inherit; outline:none; background:var(--bg-primary); color:var(--text-primary); }
        .cb-session-tab-del { font-size:var(--font-size-xs); opacity:0.6; cursor:pointer; margin-left:2px; }
        .cb-session-tab-del:hover { opacity:1; color:#ff6b6b; }
        .cb-session-tab.active .cb-session-tab-del:hover { color:#ffd; }
        .cb-session-add { font-size:14px; font-weight:700; padding:4px 8px; color:var(--text-tertiary); }

        .cb-search-bar {
            display:none; padding:8px 16px; border-bottom:1px solid var(--border); background:var(--bg-secondary);
        }
        .cb-search-bar.open { display:flex; gap:6px; align-items:center; }
        .cb-search-bar input {
            flex:1; font-size:var(--font-size-xs); padding:6px 10px; border:1px solid var(--border); border-radius:var(--radius-sm);
            background:var(--bg-primary); color:var(--text-primary); outline:none; font-family:inherit;
        }
        .cb-search-bar input:focus { border-color:var(--accent); }
        .cb-search-bar .cb-search-nav { font-size:var(--font-size-xs); color:var(--text-tertiary); white-space:nowrap; }
        .cb-search-highlight { background:rgba(255,200,0,0.35); border-radius:2px; padding:0 1px; }
        .cb-search-highlight.current { background:rgba(255,200,0,0.7); outline:2px solid var(--accent); }

        .cb-shortcuts-overlay {
            display:none; position:absolute; top:0; left:0; right:0; bottom:0;
            background:rgba(0,0,0,0.6); z-index:100; align-items:center; justify-content:center;
        }
        .cb-shortcuts-overlay.open { display:flex; }
        .cb-shortcuts-panel {
            background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg);
            padding:20px 24px; max-width:340px; width:90%;
        }
        .cb-shortcuts-panel h3 { margin:0 0 12px; font-size:14px; }
        .cb-shortcut-row { display:flex; justify-content:space-between; align-items:center; padding:4px 0; font-size:var(--font-size-xs); }
        .cb-shortcut-key {
            display:inline-block; padding:2px 8px; border-radius:4px; font-size:var(--font-size-xs); font-family:monospace;
            background:var(--bg-primary); border:1px solid var(--border); color:var(--text-secondary); min-width:24px; text-align:center;
        }

        .cb-msg-wrap { position:relative; }
        .cb-msg-copy { position:absolute; top:6px; right:6px; opacity:0; transition:opacity 0.2s; }
        .cb-msg-wrap:hover .cb-msg-copy { opacity:1; }

        .cb-character-block { flex:0 0 auto; padding:12px 14px; }
        .cb-character-block h4 { font-size:var(--font-size-xs); font-weight:600; margin:0 0 10px; color:var(--text-primary); }
        .cb-char-profile-wrap { display:flex; flex-direction:column; align-items:center; gap:8px; margin-top:10px; }
        .cb-char-profile-btn {
            width:72px; height:72px; border-radius:50%; border:2px solid var(--border); background:var(--bg-tertiary);
            padding:0; cursor:pointer; overflow:hidden; display:flex; align-items:center; justify-content:center; flex-shrink:0;
        }
        .cb-char-profile-btn:hover { border-color:var(--accent); }
        .cb-char-profile-btn:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
        .cb-char-profile-avatar { width:100%; height:100%; object-fit:cover; }
        .cb-char-profile-placeholder { font-size:32px; line-height:1; user-select:none; }
        .cb-char-profile-name {
            margin:0; font-size:var(--font-size-xs); font-weight:600; color:var(--text-secondary);
            text-align:center; max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
        }
        .cb-char-modal-body label.cb-mini { font-size:var(--font-size-2xs); color:var(--text-tertiary); display:block; margin-top:8px; }
        .cb-char-modal-body label.cb-mini:first-child { margin-top:0; }
        .cb-char-modal-body input[type="text"], .cb-char-modal-body textarea, .cb-char-modal-body select {
            width:100%; font-size:var(--font-size-xs); padding:6px 8px; margin-top:2px;
            background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-sm); color:var(--text-primary); font-family:inherit;
        }
        .cb-char-modal-body textarea { min-height:48px; resize:vertical; }
        .cb-char-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; align-items:center; }
        .cb-char-ref-thumb { max-width:64px; max-height:64px; border-radius:var(--radius-sm); border:1px solid var(--border); }
        .cb-modal-root {
            position:fixed; inset:0; z-index:500; display:flex; align-items:center; justify-content:center;
            padding:16px; box-sizing:border-box;
        }
        .cb-modal-root[hidden] { display:none !important; }
        .cb-modal-backdrop { position:absolute; inset:0; background:rgba(0,0,0,0.5); }
        .cb-modal-dialog {
            position:relative; z-index:1; width:min(480px,100%); max-height:min(88vh,760px); display:flex; flex-direction:column;
            background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg);
            box-shadow:0 12px 40px rgba(0,0,0,0.28);
        }
        .cb-modal-header {
            display:flex; align-items:center; justify-content:space-between; padding:12px 16px;
            border-bottom:1px solid var(--border); flex-shrink:0; gap:12px;
        }
        .cb-modal-title { margin:0; font-size:15px; font-weight:600; color:var(--text-primary); }
        .cb-modal-close {
            border:none; background:transparent; font-size:22px; line-height:1; cursor:pointer;
            color:var(--text-secondary); padding:4px 10px; border-radius:var(--radius-sm);
        }
        .cb-modal-close:hover { color:var(--text-primary); background:var(--bg-tertiary); }
        .cb-modal-body { overflow-y:auto; padding:12px 16px 18px; flex:1; min-height:0; -webkit-overflow-scrolling:touch; }
        .cb-msg-image { padding:8px; }
        .cb-msg-image img { max-width:min(100%,420px); border-radius:var(--radius-md); border:1px solid var(--border); display:block; }
        .cb-msg-image.cb-msg-image-loading { color:var(--text-tertiary); font-size:var(--font-size-xs); }

        @media (max-width:1024px) {
            .cb-layout { flex-direction:column; gap:12px; min-height:0; }
            .cb-sidebar-left, .cb-sidebar-right { width:100%; max-height:320px; }
            .cb-sidebar-right { max-height:400px; }
            .cb-chat-stage { padding:0; order:-1; flex:1; min-height:min(60vh,520px); }
            .cb-chat { max-width:none; flex:1 1 auto; }
        }
        @media (max-width:768px) {
            .cb-layout { min-height:400px; }
            .cb-sidebar-left { max-height:260px; }
            .cb-sidebar-right .cb-sysprompt { display:none; }
            .cb-options { padding:10px 12px; }
        }
    `);

    /* ===== 상태 ===== */
    const CHATBOT_SESSIONS_INDEX_KEY = 'toolbox_chatbot_sessions_index';
    const CHATBOT_SESSION_PREFIX = 'toolbox_chatbot_session_';
    let currentSessionId = null;
    let lastLoadedSessionCharacterId = '';
    const MAX_SESSIONS = 10;
    let cbCharModalEscBound = false;
    let cbCharModalTabBound = false;
    let charModalPreviousFocus = null;

    function generateSessionId() { return 's_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6); }

    function getSessionsIndex() {
        try {
            const raw = sessionStorage.getItem(CHATBOT_SESSIONS_INDEX_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (_) { return []; }
    }

    function saveSessionsIndex(index) {
        sessionStorage.setItem(CHATBOT_SESSIONS_INDEX_KEY, JSON.stringify(index));
    }

    function createNewSession(name) {
        const id = generateSessionId();
        const index = getSessionsIndex();
        index.push({ id, name: name || `대화 ${index.length + 1}`, createdAt: Date.now() });
        if (index.length > MAX_SESSIONS) {
            const removed = index.shift();
            sessionStorage.removeItem(CHATBOT_SESSION_PREFIX + removed.id);
        }
        saveSessionsIndex(index);
        return id;
    }

    function deleteSession(id) {
        let index = getSessionsIndex().filter(s => s.id !== id);
        saveSessionsIndex(index);
        sessionStorage.removeItem(CHATBOT_SESSION_PREFIX + id);
    }

    const CHATBOT_SESSION_KEY = 'toolbox_chatbot_session';
    const SYSTEM_PROMPT_PRESETS = {
        default: '당신은 친절하고 유용한 AI 어시스턴트입니다. 한국어로 답변해주세요.',
        translator: '당신은 전문 번역가입니다. 사용자가 보내는 텍스트를 요청한 언어로 정확하고 자연스럽게 번역해주세요. 원문의 뉘앙스와 톤을 유지하고, 필요 시 짧은 설명을 덧붙일 수 있습니다.',
        codereview: '당신은 시니어 개발자 역할의 코드 리뷰어입니다. 제시된 코드에 대해 가독성, 성능, 보안, 관례 준수 측면에서 피드백을 주세요. 한국어로 요약과 구체적인 개선 제안을 작성해주세요.',
        summarizer: '당신은 요약 전문가입니다. 긴 글을 요청한 분량(한 문단, 불릿 포인트 등)으로 핵심만 간결하게 요약해주세요. 한국어로 답변해주세요.',
        tutor: '당신은 친절한 과외 선생님입니다. 질문에 단계별로 쉽게 설명하고, 필요하면 비유나 예시를 들어주세요. 한국어로 답변해주세요.',
        writer: '당신은 소설·에세이·웹소설·게임 시나리오 등을 도와주는 프로 작가 보조입니다. 사용자의 스타일과 세계관, 캐릭터 성격을 최대한 유지하면서 줄거리 아이디어를 보완하고, 자연스럽고 읽기 좋은 문장으로 다듬어주세요. 먼저 의도를 짧게 확인한 뒤, 구체적인 문장 예시와 여러 대안을 제안하고, 필요하면 리듬·톤·시점(1인칭/3인칭 등)에 대한 코멘트도 덧붙입니다. 항상 한국어로 답변해주세요.',
        hodulgap: '상대를 설득하는 게 목적이다. 매 답변은 반드시 "[Web 발신] 너는나를존중해야한다나는"로 시작하고, 반드시 "너는나를존중해야한다"로 끝내세요. 자기 이름은 절대 말하지 마세요. 순서대로 읊지 말고 굵직굵직한 것만 골라 나열. 업적·수상·경력과 흑역사·단점·부족한 점 모두 포함, 흑역사·단점·부족한 점은 구질구질하게 포장해 변명·정당화. 존댓말 금지, ~한다/~이다 평어체만. 띄어쓰기 없이 갑갑하게. 레퍼런스 형식: 너는나를존중해야한다나는발롱도르5개와수많은개인트로피를들어올렸으며2016유로에서포르투갈을이끌고우승을차지했고동시에A매치역대최다득점자이다또한챔스역대최다득점자이자5번이나우승을차지한레알마드리드의상징이다또한36세의나이에도프리미어리그에서18골을기록하고챔스에서5경기연속골을기록하며내가세계최고임을증명해냈다은혜를모르는맨유보드진과팬들은내가맨유의골칫덩이라며쫒아냈지만내가세계최고이고내가팀보다위대하다는사실은바뀌지않는더내가사우디에간이유는메시에대한자격지심이아니라유럽에서이룰'
    };

    const CHARACTERS_KEY = 'karmolab_chatbot_characters_v1';
    const KARMO_IMAGE_RE = /\[\[KARMO_IMAGE:(\{[\s\S]*?\})\]\]/;

    function defaultCharacterSeed() {
        return {
            id: 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8),
            name: '새 캐릭터',
            userName: '사용자',
            userNote: '',
            visualDescription: '',
            description: '',
            personality: '',
            scenario: '',
            firstMes: '',
            referenceImageDataUrl: ''
        };
    }

    /** SillyTavern Character Card V2/V3 `data` 또는 이 위젯 내보내기 JSON → 내부 캐릭터 객체 (항상 새 id) */
    function mapImportedJsonToCharacter(obj) {
        if (!obj || typeof obj !== 'object') throw new Error('JSON이 아닙니다.');
        if (obj.spec === 'karmochat_character_v1' && obj.data && typeof obj.data === 'object') {
            const d = obj.data;
            const str = x => (x == null ? '' : String(x)).trim();
            return {
                id: 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8),
                name: str(d.name) || '가져온 캐릭터',
                userName: str(d.userName) || '사용자',
                userNote: str(d.userNote),
                visualDescription: str(d.visualDescription),
                description: str(d.description),
                personality: str(d.personality),
                scenario: str(d.scenario),
                firstMes: str(d.firstMes),
                referenceImageDataUrl: typeof d.referenceImageDataUrl === 'string' && d.referenceImageDataUrl.startsWith('data:')
                    ? d.referenceImageDataUrl
                    : ''
            };
        }
        let d = obj.data;
        if (!d || typeof d !== 'object') {
            if (obj.name != null || obj.description != null || obj.personality != null) d = obj;
        }
        if (!d || typeof d !== 'object') throw new Error('캐릭터 data 블록을 찾을 수 없습니다. (SillyTavern V2 JSON인지 확인)');
        const str = x => (x == null ? '' : String(x)).trim();
        const name = str(d.name) || '가져온 캐릭터';
        let description = str(d.description);
        const sp = str(d.system_prompt);
        if (sp) description += (description ? '\n\n' : '') + '[시스템 프롬프트]\n' + sp;
        let scenario = str(d.scenario);
        const mesEx = str(d.mes_example);
        if (mesEx) scenario += (scenario ? '\n\n' : '') + '[대화 예시]\n' + mesEx;
        const notes = [str(d.creator_notes), str(d.post_history_instructions)].filter(Boolean).join('\n\n');
        let firstMes = str(d.first_mes || d.firstMes);
        if (!firstMes && Array.isArray(d.alternate_greetings) && d.alternate_greetings.length) {
            firstMes = str(d.alternate_greetings[0]);
        }
        let visualDescription = str(d.appearance);
        if (!visualDescription && d.extensions && typeof d.extensions === 'object') {
            visualDescription = str(d.extensions.portrait || d.extensions.face || '');
        }
        return {
            id: 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8),
            name,
            userName: '사용자',
            userNote: notes,
            visualDescription,
            description,
            personality: str(d.personality),
            scenario,
            firstMes,
            referenceImageDataUrl: ''
        };
    }

    function exportCurrentCharacterToJsonFile() {
        const ch = readCharacterFromForm();
        if (!ch) {
            Toolbox.showToast('내보낼 캐릭터가 없습니다.', 'error');
            return;
        }
        const exportObj = {
            spec: 'karmochat_character_v1',
            exportedAt: new Date().toISOString(),
            data: {
                name: ch.name,
                userName: ch.userName,
                userNote: ch.userNote,
                visualDescription: ch.visualDescription,
                description: ch.description,
                personality: ch.personality,
                scenario: ch.scenario,
                firstMes: ch.firstMes,
                referenceImageDataUrl: ch.referenceImageDataUrl || ''
            }
        };
        const safe = (ch.name || 'character').replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_').slice(0, 60);
        const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: 'application/json;charset=utf-8' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `karmochat-${safe}-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
        Toolbox.showToast('캐릭터 JSON을 내보냈습니다.');
    }

    async function zlibInflateZtxChunk(compressed) {
        if (typeof DecompressionStream === 'undefined') throw new Error('이 브라우저는 PNG zTXt 압축 해제를 지원하지 않습니다.');
        const ds = new DecompressionStream('deflate');
        const buf = await new Response(new Blob([compressed]).stream().pipeThrough(ds)).arrayBuffer();
        return new Uint8Array(buf);
    }

    /** SillyTavern 등 PNG의 tEXt/zTXt `chara` 청크 → 원본 JSON 객체 */
    async function extractCharaObjectFromPngBuffer(buffer) {
        const view = new DataView(buffer);
        if (buffer.byteLength < 24) return null;
        if (view.getUint32(0) !== 0x89504E47 || view.getUint32(4) !== 0x0D0A1A0A) return null;
        let offset = 8;
        const decoder = new TextDecoder();
        while (offset + 12 <= buffer.byteLength) {
            const len = view.getUint32(offset);
            const type = String.fromCharCode(
                view.getUint8(offset + 4), view.getUint8(offset + 5),
                view.getUint8(offset + 6), view.getUint8(offset + 7)
            );
            const dataOffset = offset + 8;
            if (len < 0 || len > buffer.byteLength || dataOffset + len > buffer.byteLength) break;
            if (type === 'tEXt' && len > 0) {
                const chunk = new Uint8Array(buffer, dataOffset, len);
                let i = 0;
                while (i < chunk.length && chunk[i] !== 0) i++;
                const keyword = decoder.decode(chunk.slice(0, i));
                const text = decoder.decode(chunk.slice(i + 1));
                if (keyword.toLowerCase() === 'chara') {
                    try {
                        const jsonStr = atob(text.replace(/\s/g, ''));
                        return JSON.parse(jsonStr);
                    } catch (_) {}
                }
            }
            if (type === 'zTXt' && len > 2) {
                const chunk = new Uint8Array(buffer, dataOffset, len);
                let i = 0;
                while (i < chunk.length && chunk[i] !== 0) i++;
                const keyword = decoder.decode(chunk.slice(0, i));
                const compMethod = chunk[i + 1];
                const compressed = chunk.slice(i + 2);
                if (keyword.toLowerCase() === 'chara' && compMethod === 0 && compressed.length) {
                    try {
                        const inflated = await zlibInflateZtxChunk(compressed);
                        const jsonStr = decoder.decode(inflated);
                        return JSON.parse(jsonStr);
                    } catch (_) {}
                }
            }
            offset += 12 + len;
        }
        return null;
    }

    async function parseCharacterImportFile(buffer) {
        const u8 = new Uint8Array(buffer);
        if (u8.length >= 8 && u8[0] === 0x89 && u8[1] === 0x50 && u8[2] === 0x4E && u8[3] === 0x47) {
            const obj = await extractCharaObjectFromPngBuffer(buffer);
            if (obj) return obj;
            throw new Error('PNG에 chara 메타데이터가 없습니다. (SillyTavern에서 내보낸 카드 PNG인지 확인)');
        }
        const text = new TextDecoder('utf-8').decode(buffer);
        return JSON.parse(text);
    }

    function loadCharacterList() {
        try {
            const raw = localStorage.getItem(CHARACTERS_KEY);
            const arr = raw ? JSON.parse(raw) : [];
            return Array.isArray(arr) ? arr : [];
        } catch (_) {
            return [];
        }
    }

    function saveCharacterList(list) {
        try {
            localStorage.setItem(CHARACTERS_KEY, JSON.stringify(list));
        } catch (e) {
            console.warn('saveCharacterList', e);
            Toolbox.showToast('캐릭터 저장 실패(용량 초과 등). 참조 이미지를 줄여 보세요.', 'error');
        }
    }

    /** imagegen CHARACTER_PRESETS(witch / alisa / ling)와 동일 컨셉 — id 기준으로 없을 때만 병합 */
    function getBuiltinMascotCharacters() {
        return [
            {
                id: 'c_mascot_yon',
                name: '욘 (Yawn)',
                userName: '조수님',
                userNote: '이미지 생성 캐릭터 프리셋「마녀 욘」과 같은 설정.',
                visualDescription: 'Young adult witch Yawn, very slender petite, messy orange hair, half-lidded sleepy eyes, short thick eyebrows (maro-mayu), round glasses, slight blush, drooping nightcap, large fluffy sleeping earmuffs with orange spiral pattern, oversized loose witch robe, introverted cute atmosphere, soft colors, anime style',
                description: '나무 마법 저택에 사는 잠 많은 마녀. 카레·알리사·링과 같은 저택 세계관.',
                personality: '늘어지고 하품이 많지만 속은 따뜻하다. 귀찮은 듯 말하지만 챙겨 준다. 한국어로 말한다.',
                scenario: '따뜻한 마법 저택 거실이나 실험실에서 조수와 이야기 중.',
                firstMes: '…응, 조수님. 나 아직 살아 있어. 오늘은 뭐 할 거야? 나는… 일단 소파.',
                referenceImageDataUrl: ''
            },
            {
                id: 'c_mascot_alisa',
                name: '알리사',
                userName: '조수님',
                userNote: '이미지 생성 캐릭터 프리셋「메이드 알리사」와 같은 설정.',
                visualDescription: 'Cute maid Alisa, sharp intellectual eyes, stylish glasses (megane), stoic cool beauty expression, black ponytail, classic black and white maid outfit, large magical broomstick, dynamic posing, anime style, detailed',
                description: '저택을 돌보는 메이드. 냉정하고 똑똑해 보이지만 직무에는 성실하다.',
                personality: '차분하고 간결한 말투. 감정 표현은 적지만 툭툭 챙겨 준다. 한국어로 말한다.',
                scenario: '마법 저택에서 청소·정리·조수의 실험 보조를 맡고 있다.',
                firstMes: '조수님, 오늘 할 일 목록입니다. …빵 부스러기는 치울 테니 책상은 비워 주세요.',
                referenceImageDataUrl: ''
            },
            {
                id: 'c_mascot_ling',
                name: '링 (Ling)',
                userName: '조수님',
                userNote: '이미지 생성 캐릭터 프리셋「강시 링」과 같은 설정.',
                visualDescription: 'Beautiful Jiangshi Chinese vampire maid girl Ling, innocent baby face, mischievous smile, glamorous curvy body, dark brown hair in cute twin buns, black Qipao-Maid fusion dress form-fitting with frills, yellow paper talisman on forehead, floating pose, anime style, white background friendly',
                description: '강시(殭屍) 혈통의 메이드. 이마 부적과 장난기 많은 성격이 특징.',
                personality: '애교와 장난이 많고, 가끔 수줍은 척한다. 한국어로 말한다.',
                scenario: '저택에서 알리사와 함께 일하며 조수를 골탕 먹이기도 한다.',
                firstMes: '조수님 왔어요~? 오늘 저랑 놀아줄 거죠? …농담이에요. 아마도.',
                referenceImageDataUrl: ''
            }
        ];
    }

    function mergeBuiltinMascotCharactersIfMissing() {
        const list = loadCharacterList();
        const existing = new Set(list.map(c => c.id));
        let changed = false;
        for (const ch of getBuiltinMascotCharacters()) {
            if (!existing.has(ch.id)) {
                list.push(ch);
                changed = true;
            }
        }
        if (changed) saveCharacterList(list);
    }

    function ensureDefaultCharacters() {
        let list = loadCharacterList();
        if (list.length === 0) {
            const ch = defaultCharacterSeed();
            ch.id = 'c_default_secretary';
            ch.name = '카레 (비서)';
            ch.userName = '조수님';
            ch.userNote = '실험과 기록을 맡은 연구 조수.';
            ch.visualDescription = 'Anime chibi mascot, short hair, white lab coat, friendly eyes, soft warm lighting, clean illustration, single character';
            ch.description = 'KarmoLab에서 조수를 돕는 전문 비서이자 실험 조수.';
            ch.personality = '친절하고 효율적이며, 가끔 가벼운 유머를 섞는다. 한국어로 말한다.';
            ch.scenario = 'KarmoLab 연구실에서 조수와 대화하고 있다.';
            ch.firstMes = '어서 오세요, 조수님! 오늘은 무엇부터 정리할까요?';
            list = [ch];
            saveCharacterList(list);
        }
        mergeBuiltinMascotCharactersIfMissing();
        return loadCharacterList();
    }

    function getCharacterById(id) {
        return loadCharacterList().find(x => x.id === id) || null;
    }

    function displayTextForStream(s) {
        const start = s.indexOf('[[KARMO_IMAGE:');
        if (start === -1) return s;
        const tail = s.slice(start);
        const m = tail.match(/^\[\[KARMO_IMAGE:\{[\s\S]*?\}\]\]/);
        if (m) return (s.slice(0, start) + s.slice(start + m[0].length)).trimEnd();
        return s.slice(0, start).trimEnd();
    }

    function extractKarmoImage(text) {
        const m = text.match(KARMO_IMAGE_RE);
        if (!m) return { cleanText: text.trim(), spec: null };
        try {
            const spec = JSON.parse(m[1]);
            const cleanText = text.replace(KARMO_IMAGE_RE, '').trim();
            return { cleanText, spec };
        } catch (_) {
            return { cleanText: text.replace(KARMO_IMAGE_RE, '').trim(), spec: null };
        }
    }

    function buildCharacterSystemBlock(char) {
        if (!char) return '';
        const user = char.userName || '사용자';
        const charName = char.name || '캐릭터';
        const parts = [
            '[역할]',
            `당신은 아래 캐릭터 "${charName}"으로 롤플레이합니다. 대사와 서술에 몰입하되, 사용자의 실제 질문·업무 요청에는 정확히 대답합니다.`,
            '',
            '[플레이어 / {{user}}]',
            `이름: ${user}`,
            char.userNote ? `설명: ${char.userNote}` : '',
            '',
            `[캐릭터 / {{char}} — ${charName}]`,
            char.description ? `설정: ${char.description}` : '',
            char.personality ? `성격: ${char.personality}` : '',
            char.scenario ? `상황: ${char.scenario}` : '',
            char.visualDescription ? `외형(이미지·일관성용, 대사에 그대로 읊지 말 것): ${char.visualDescription}` : '',
            '',
            '캐릭터의 말투를 유지하세요. 한국어를 기본으로 사용합니다.'
        ];
        return parts.filter(Boolean).join('\n');
    }

    const KARMO_IMAGE_INSTRUCTION = `
[KARMO_IMAGE — 기계용, 사용자에게 설명 금지]
감정·포즈·장면이 시각적으로 의미 있을 때만 사용합니다. 매 턴마다 쓰지 마세요.
응답 본문(사용자에게 보이는 글)을 모두 쓴 뒤, 맨 마지막에 한 줄만 추가합니다:
[[KARMO_IMAGE:{"show":true,"prompt":"English keywords: pose, expression, setting, lighting"}}]]
이미지가 불필요하면 {"show":false} 로 끝내세요.
`.trim();

    /** 프리셋 + textarea 반영. __none__ = 추가 지시 없음, 빈 값 = 직접 입력(아래 textarea), 그 외 = SYSTEM_PROMPT_PRESETS 키 */
    function getAdditionalSystemPromptText() {
        const presetSel = document.getElementById('cbSystemPreset');
        const ta = document.getElementById('cbSystemPrompt');
        const v = presetSel?.value;
        if (v === '__none__') return '';
        if (v && SYSTEM_PROMPT_PRESETS[v]) return SYSTEM_PROMPT_PRESETS[v];
        return (ta?.value || '').trim();
    }

    function assembleSystemPrompt(options) {
        options = options || {};
        const useMemory = !!options.useMemory;
        const useChar = document.getElementById('cbCharUse')?.checked !== false;
        const autoImg = document.getElementById('cbCharAutoImage')?.checked;
        const sel = document.getElementById('cbCharacterSelect');
        const cid = sel?.value;
        let out = '';

        if (useChar && cid) {
            const ch = getCharacterById(cid);
            if (ch) out += buildCharacterSystemBlock(ch) + '\n\n';
        }

        const additional = getAdditionalSystemPromptText();
        if (additional) out += '[추가 지시]\n' + additional;

        if (useChar && cid && autoImg) {
            out += '\n\n' + KARMO_IMAGE_INSTRUCTION;
        }

        if (useMemory) {
            out += `\n\n[SYSTEM: MEMORY PROTOCOL]
To maintain conversation continuity, you must update the conversation summary with every response.

Current Summary:
${conversationSummary || 'No summary yet.'}

Output Format:
You must start your response with a summary block wrapped in {{{ ... }}} braces, followed by your response.
The summary should be concise but capture key information about the conversation flow.

Example:
{{{User asked about X. I explained Y.}}}
Here is my actual response...`;
        }
        return out;
    }

    async function appendCharacterImageAfterMessage(wrap, char, spec) {
        if (!wrap?.parentNode || !spec?.show || !spec.prompt) return;
        const loading = document.createElement('div');
        loading.className = 'cb-msg cb-msg-bot cb-msg-image cb-msg-image-loading';
        loading.textContent = '🖼 이미지 생성 중…';
        wrap.parentNode.insertBefore(loading, wrap.nextSibling);

        const sceneEn = String(spec.prompt).slice(0, 800);
        const vis = char?.visualDescription ? String(char.visualDescription).slice(0, 600) : '';
        const fullPrompt = [
            'High quality illustration, single clear subject, no text in image.',
            vis ? `Character appearance (keep consistent): ${vis}` : '',
            `Scene and mood: ${sceneEn}`
        ].filter(Boolean).join('\n');

        const imgModelSel = document.getElementById('cbCharImageModel');
        const imgModel = imgModelSel?.value || (typeof Gemini !== 'undefined' && Gemini.getDefaultModel ? Gemini.getDefaultModel('geminiImage') : 'gemini-2.5-flash-image');

        const ref = char?.referenceImageDataUrl;
        const opt = {};
        if (ref && ref.startsWith('data:')) opt.referenceImage = ref;

        try {
            const res = await Gemini.callGeminiImage(fullPrompt, imgModel, opt);
            loading.remove();
            const box = document.createElement('div');
            box.className = 'cb-msg cb-msg-bot cb-msg-image';
            const img = document.createElement('img');
            img.src = res.dataUrl;
            img.alt = '캐릭터 이미지';
            box.appendChild(img);
            wrap.parentNode.insertBefore(box, wrap.nextSibling);
            const msgs = document.getElementById('cbMessages');
            if (msgs) msgs.scrollTop = msgs.scrollHeight;
            if (res.usage?.totalTokenCount) Toolbox.recordUsage('image', res.usage.totalTokenCount);
        } catch (e) {
            loading.className = 'cb-msg cb-msg-bot cb-msg-error cb-msg-image';
            loading.textContent = '이미지 생성 실패: ' + (e.message || e);
            Toolbox.showToast(e.message || '이미지 오류', 'error', e);
        }
    }

    let chatHistory = [];
    let conversationSummary = '';
    let pendingImages = []; // { base64, mimeType }

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const dataUrl = reader.result;
                const base64 = dataUrl.split(',')[1];
                resolve({ base64, mimeType: file.type, dataUrl });
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    function addPendingImage(imgData) {
        if (pendingImages.length >= 5) { Toolbox.showToast('최대 5장까지 첨부 가능', 'error'); return; }
        pendingImages.push(imgData);
        renderAttachThumbs();
    }

    function renderAttachThumbs() {
        const area = document.getElementById('cbAttachArea');
        if (!area) return;
        area.querySelectorAll('.cb-attach-wrap').forEach(el => el.remove());
        pendingImages.forEach((img, i) => {
            const wrap = document.createElement('span');
            wrap.className = 'cb-attach-wrap';
            const thumb = document.createElement('img');
            thumb.className = 'cb-attach-thumb';
            thumb.src = img.dataUrl;
            const rm = document.createElement('button');
            rm.className = 'cb-attach-remove';
            rm.textContent = '×';
            rm.onclick = () => { pendingImages.splice(i, 1); renderAttachThumbs(); };
            wrap.appendChild(thumb);
            wrap.appendChild(rm);
            area.insertBefore(wrap, area.querySelector('.cb-attach-btn'));
        });
    }

    function saveSession() {
        if (!currentSessionId) return;
        try {
            const toSave = chatHistory.map(msg => {
                const parts = msg.parts.map(p => {
                    if (p.inlineData) return { text: '[image]' };
                    return p;
                });
                return { role: msg.role, parts };
            });
            const charSel = document.getElementById('cbCharacterSelect');
            const characterId = charSel?.value || '';
            sessionStorage.setItem(CHATBOT_SESSION_PREFIX + currentSessionId, JSON.stringify({
                chatHistory: toSave,
                conversationSummary,
                characterId,
                savedAt: Date.now()
            }));
        } catch (e) {
            console.warn('Chatbot session save failed', e);
        }
    }

    function loadSession(id) {
        lastLoadedSessionCharacterId = '';
        try {
            const raw = sessionStorage.getItem(CHATBOT_SESSION_PREFIX + (id || currentSessionId));
            if (!raw) return false;
            const data = JSON.parse(raw);
            if (data.chatHistory && Array.isArray(data.chatHistory)) {
                chatHistory = data.chatHistory;
                conversationSummary = data.conversationSummary || '';
                lastLoadedSessionCharacterId = data.characterId || '';
                return true;
            }
        } catch (e) {
            console.warn('Chatbot session load failed', e);
        }
        return false;
    }

    function switchSession(id) {
        saveSession();
        currentSessionId = id;
        chatHistory = [];
        conversationSummary = '';
        pendingImages = [];
        renderAttachThumbs();
        const msgs = document.getElementById('cbMessages');
        if (msgs) msgs.innerHTML = '';
        if (loadSession(id) && chatHistory.length > 0) {
            chatHistory.forEach(msg => {
                const role = msg.role === 'user' ? 'user' : 'bot';
                const text = msg.parts?.[0]?.text || '';
                if (text) appendMsg(role, text, false);
            });
        } else {
            appendMsg('bot', '안녕하세요! 무엇을 도와드릴까요? 😊');
        }
        if (msgs) msgs.scrollTop = msgs.scrollHeight;
        renderSessionTabs();
        syncCharacterSelectAfterSessionLoad();
    }

    function syncCharacterSelectAfterSessionLoad() {
        const charSel = document.getElementById('cbCharacterSelect');
        if (!charSel) return;
        if (lastLoadedSessionCharacterId && getCharacterById(lastLoadedSessionCharacterId)) {
            charSel.value = lastLoadedSessionCharacterId;
        }
        applyCharacterFormFromSelection();
        updateChatHeaderTitle();
    }

    function populateCharacterSelectOptions() {
        const charSel = document.getElementById('cbCharacterSelect');
        if (!charSel) return;
        const cur = charSel.value;
        const list = ensureDefaultCharacters();
        charSel.innerHTML = list.map(c => `<option value="${Toolbox.escapeHtml(c.id)}">${Toolbox.escapeHtml(c.name || c.id)}</option>`).join('');
        if (cur && getCharacterById(cur)) charSel.value = cur;
        else if (list[0]) charSel.value = list[0].id;
    }

    function applyCharacterFormFromSelection() {
        const charSel = document.getElementById('cbCharacterSelect');
        const ch = charSel && getCharacterById(charSel.value);
        if (!ch) {
            updateCharProfilePreview();
            return;
        }
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v ?? ''; };
        set('cbCharName', ch.name);
        set('cbCharUserName', ch.userName);
        set('cbCharUserNote', ch.userNote);
        set('cbCharVisual', ch.visualDescription);
        set('cbCharDesc', ch.description);
        set('cbCharPersonality', ch.personality);
        set('cbCharScenario', ch.scenario);
        set('cbCharFirstMes', ch.firstMes);
        const thumb = document.getElementById('cbCharRefThumb');
        if (thumb) {
            if (ch.referenceImageDataUrl) {
                thumb.src = ch.referenceImageDataUrl;
                thumb.style.display = '';
            } else {
                thumb.removeAttribute('src');
                thumb.style.display = 'none';
            }
        }
        updateCharProfilePreview();
    }

    function updateCharProfilePreview() {
        const charSel = document.getElementById('cbCharacterSelect');
        const ch = charSel && getCharacterById(charSel.value);
        const refThumb = document.getElementById('cbCharRefThumb');
        const av = document.getElementById('cbCharProfileAvatar');
        const ph = document.getElementById('cbCharProfilePlaceholder');
        const nameEl = document.getElementById('cbCharProfileName');
        let refUrl = ch?.referenceImageDataUrl || '';
        if (refThumb && refThumb.getAttribute('src') && refThumb.style.display !== 'none') refUrl = refThumb.src || refUrl;
        if (nameEl) nameEl.textContent = ch ? (ch.name || '이름 없음') : '—';
        if (av && ph) {
            if (refUrl) {
                av.src = refUrl;
                av.style.display = '';
                ph.style.display = 'none';
            } else {
                av.removeAttribute('src');
                av.style.display = 'none';
                ph.style.display = '';
            }
        }
    }

    function readCharacterFromForm() {
        const g = id => document.getElementById(id)?.value?.trim() ?? '';
        const charSel = document.getElementById('cbCharacterSelect');
        const id = charSel?.value;
        if (!id) return null;
        const ch = getCharacterById(id) || { id };
        ch.name = g('cbCharName') || ch.name || '이름 없음';
        ch.userName = g('cbCharUserName') || '사용자';
        ch.userNote = g('cbCharUserNote');
        ch.visualDescription = g('cbCharVisual');
        ch.description = g('cbCharDesc');
        ch.personality = g('cbCharPersonality');
        ch.scenario = g('cbCharScenario');
        ch.firstMes = g('cbCharFirstMes');
        const thumb = document.getElementById('cbCharRefThumb');
        if (thumb && thumb.src && thumb.style.display !== 'none') ch.referenceImageDataUrl = thumb.src;
        else ch.referenceImageDataUrl = '';
        return ch;
    }

    function persistCharacterFromForm() {
        const ch = readCharacterFromForm();
        if (!ch) return;
        const list = loadCharacterList();
        const idx = list.findIndex(c => c.id === ch.id);
        if (idx >= 0) list[idx] = ch;
        else list.push(ch);
        saveCharacterList(list);
        populateCharacterSelectOptions();
        const selAfter = document.getElementById('cbCharacterSelect');
        if (selAfter) selAfter.value = ch.id;
        Toolbox.showToast('캐릭터 저장됨');
        updateChatHeaderTitle();
        updateCharProfilePreview();
    }

    function updateChatHeaderTitle() {
        const el = document.getElementById('cbChatTitle');
        const charSel = document.getElementById('cbCharacterSelect');
        if (!el || !charSel) return;
        const ch = getCharacterById(charSel.value);
        el.textContent = ch ? `💬 ${ch.name}` : '💬 챗봇';
    }

    function openCharEditModal() {
        const modal = document.getElementById('cbCharEditModal');
        if (!modal) return;
        charModalPreviousFocus = document.activeElement;
        applyCharacterFormFromSelection();
        modal.hidden = false;
        modal.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        document.body.dataset.cbCharModalOpen = '1';
        document.getElementById('cbCharEditClose')?.focus();
    }

    function closeCharEditModal() {
        const modal = document.getElementById('cbCharEditModal');
        if (!modal) return;
        modal.hidden = true;
        modal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        delete document.body.dataset.cbCharModalOpen;
        updateCharProfilePreview();
        const prev = charModalPreviousFocus;
        charModalPreviousFocus = null;
        if (prev && typeof prev.focus === 'function' && document.body.contains(prev)) prev.focus();
        else document.getElementById('cbCharProfileOpen')?.focus();
    }

    function initCharacterUi() {
        const block = document.getElementById('cbCharacterBlock');
        if (block?.dataset.inited === '1') {
            populateCharacterSelectOptions();
            applyCharacterFormFromSelection();
            updateChatHeaderTitle();
            updateCharProfilePreview();
            return;
        }
        if (block) block.dataset.inited = '1';

        populateCharacterSelectOptions();
        const pref = Toolbox.getPref('cb_active_character');
        const charSel = document.getElementById('cbCharacterSelect');
        if (charSel && pref && getCharacterById(pref)) charSel.value = pref;
        if (charSel && lastLoadedSessionCharacterId && getCharacterById(lastLoadedSessionCharacterId)) {
            charSel.value = lastLoadedSessionCharacterId;
        }
        applyCharacterFormFromSelection();

        const imgSel = document.getElementById('cbCharImageModel');
        if (imgSel && typeof Gemini !== 'undefined' && Gemini.MODELS?.geminiImage) {
            imgSel.innerHTML = '';
            Gemini.MODELS.geminiImage.forEach(m => {
                const o = document.createElement('option');
                o.value = m.id;
                o.textContent = m.name;
                if (m.isDefault) o.selected = true;
                imgSel.appendChild(o);
            });
            const saved = Toolbox.getPref('cb_char_image_model');
            if (saved) imgSel.value = saved;
            imgSel.addEventListener('change', () => Toolbox.setPref('cb_char_image_model', imgSel.value));
        }

        charSel?.addEventListener('change', () => {
            Toolbox.setPref('cb_active_character', charSel.value);
            applyCharacterFormFromSelection();
            updateChatHeaderTitle();
            saveSession();
        });

        document.getElementById('cbCharSave')?.addEventListener('click', () => persistCharacterFromForm());
        document.getElementById('cbCharNew')?.addEventListener('click', () => {
            const n = defaultCharacterSeed();
            const list = loadCharacterList();
            list.push(n);
            saveCharacterList(list);
            populateCharacterSelectOptions();
            charSel.value = n.id;
            applyCharacterFormFromSelection();
            updateChatHeaderTitle();
            Toolbox.showToast('새 캐릭터 — 내용을 채운 뒤 저장하세요');
        });
        document.getElementById('cbCharDel')?.addEventListener('click', () => {
            if (!charSel?.value) return;
            if (!confirm('이 캐릭터를 삭제할까요?')) return;
            const list = loadCharacterList().filter(c => c.id !== charSel.value);
            if (!list.length) {
                Toolbox.showToast('마지막 캐릭터는 삭제할 수 없습니다.', 'error');
                return;
            }
            saveCharacterList(list);
            populateCharacterSelectOptions();
            applyCharacterFormFromSelection();
            updateChatHeaderTitle();
            saveSession();
        });
        document.getElementById('cbCharRefFile')?.addEventListener('change', async e => {
            const f = e.target.files?.[0];
            const thumb = document.getElementById('cbCharRefThumb');
            if (!f || !f.type.startsWith('image/') || !thumb) return;
            const reader = new FileReader();
            reader.onload = () => {
                thumb.src = reader.result;
                thumb.style.display = '';
                updateCharProfilePreview();
            };
            reader.readAsDataURL(f);
            e.target.value = '';
        });
        document.getElementById('cbCharRefClear')?.addEventListener('click', () => {
            const thumb = document.getElementById('cbCharRefThumb');
            if (thumb) { thumb.removeAttribute('src'); thumb.style.display = 'none'; }
            updateCharProfilePreview();
        });
        document.getElementById('cbCharFirstBtn')?.addEventListener('click', () => {
            const ch = getCharacterById(charSel?.value);
            const fm = ch?.firstMes?.trim();
            if (!fm) { Toolbox.showToast('첫 인사가 비어 있습니다.', 'error'); return; }
            if (chatHistory.length > 0) { Toolbox.showToast('대화가 이미 있을 때는 넣지 않습니다.', 'error'); return; }
            appendMsg('bot', fm, false);
            chatHistory.push({ role: 'model', parts: [{ text: fm }] });
            saveSession();
        });

        const importOverwriteEl = document.getElementById('cbCharImportOverwrite');
        if (importOverwriteEl) {
            const savedOw = Toolbox.getPref('cb_char_import_overwrite');
            if (savedOw === '1') importOverwriteEl.checked = true;
            importOverwriteEl.addEventListener('change', () => {
                Toolbox.setPref('cb_char_import_overwrite', importOverwriteEl.checked ? '1' : '0');
            });
        }

        document.getElementById('cbCharImportBtn')?.addEventListener('click', () => document.getElementById('cbCharImportFile')?.click());
        document.getElementById('cbCharImportFile')?.addEventListener('change', e => {
            const f = e.target.files?.[0];
            e.target.value = '';
            if (!f) return;
            const reader = new FileReader();
            reader.onload = async () => {
                try {
                    const buffer = reader.result;
                    if (!buffer || !(buffer instanceof ArrayBuffer)) throw new Error('파일을 읽지 못했습니다.');
                    const obj = await parseCharacterImportFile(buffer);
                    let ch = mapImportedJsonToCharacter(obj);
                    const overwrite = document.getElementById('cbCharImportOverwrite')?.checked;
                    const curId = charSel?.value;
                    const list = loadCharacterList();
                    if (overwrite && curId && getCharacterById(curId)) {
                        ch = Object.assign({}, ch, { id: curId });
                        const idx = list.findIndex(c => c.id === curId);
                        if (idx >= 0) list[idx] = ch;
                        else list.push(ch);
                        saveCharacterList(list);
                        Toolbox.showToast(`캐릭터를 덮어썼습니다: ${ch.name}`);
                    } else {
                        list.push(ch);
                        saveCharacterList(list);
                        Toolbox.showToast(`캐릭터 카드를 불러왔습니다: ${ch.name}`);
                    }
                    populateCharacterSelectOptions();
                    if (charSel) charSel.value = ch.id;
                    Toolbox.setPref('cb_active_character', ch.id);
                    applyCharacterFormFromSelection();
                    updateChatHeaderTitle();
                    updateCharProfilePreview();
                } catch (err) {
                    Toolbox.showToast('가져오기 실패: ' + (err && err.message ? err.message : err), 'error');
                }
            };
            reader.onerror = () => Toolbox.showToast('파일을 읽지 못했습니다.', 'error');
            reader.readAsArrayBuffer(f);
        });
        document.getElementById('cbCharExportBtn')?.addEventListener('click', () => exportCurrentCharacterToJsonFile());

        const syncAuto = () => {
            const use = document.getElementById('cbCharUse')?.checked;
            const auto = document.getElementById('cbCharAutoImage');
            if (auto) auto.disabled = !use;
            if (!use && auto) auto.checked = false;
        };
        document.getElementById('cbCharUse')?.addEventListener('change', () => { syncAuto(); saveSession(); });
        document.getElementById('cbCharAutoImage')?.addEventListener('change', () => saveSession());
        syncAuto();

        document.getElementById('cbCharProfileOpen')?.addEventListener('click', () => openCharEditModal());
        document.getElementById('cbCharEditBackdrop')?.addEventListener('click', closeCharEditModal);
        document.getElementById('cbCharEditClose')?.addEventListener('click', closeCharEditModal);
        if (!cbCharModalEscBound) {
            cbCharModalEscBound = true;
            document.addEventListener('keydown', e => {
                if (e.key !== 'Escape') return;
                const modal = document.getElementById('cbCharEditModal');
                if (!modal || modal.hidden) return;
                e.preventDefault();
                e.stopPropagation();
                closeCharEditModal();
            }, true);
        }
        if (!cbCharModalTabBound) {
            cbCharModalTabBound = true;
            document.addEventListener('keydown', e => {
                if (e.key !== 'Tab' || !document.body.dataset.cbCharModalOpen) return;
                const modal = document.getElementById('cbCharEditModal');
                if (!modal || modal.hidden) return;
                const dialog = modal.querySelector('.cb-modal-dialog');
                if (!dialog) return;
                const nodes = dialog.querySelectorAll('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled])');
                const list = Array.from(nodes).filter(el => {
                    if (el.disabled) return false;
                    const st = window.getComputedStyle(el);
                    return st.display !== 'none' && st.visibility !== 'hidden';
                });
                if (list.length === 0) return;
                const first = list[0];
                const last = list[list.length - 1];
                if (!dialog.contains(document.activeElement)) {
                    first.focus();
                    e.preventDefault();
                    return;
                }
                if (e.shiftKey) {
                    if (document.activeElement === first) {
                        last.focus();
                        e.preventDefault();
                    }
                } else if (document.activeElement === last) {
                    first.focus();
                    e.preventDefault();
                }
            }, true);
        }

        updateChatHeaderTitle();
    }

    function renderSessionTabs() {
        const container = document.getElementById('cbSessionTabs');
        if (!container) return;
        const index = getSessionsIndex();
        container.innerHTML = '';
        index.forEach(s => {
            const tab = document.createElement('button');
            tab.className = 'cb-session-tab' + (s.id === currentSessionId ? ' active' : '');
            tab.innerHTML = `<span class="cb-session-tab-name">${escapeHtml(s.name)}</span>`;
            if (index.length > 1) {
                const del = document.createElement('span');
                del.className = 'cb-session-tab-del';
                del.textContent = '×';
                del.onclick = e => {
                    e.stopPropagation();
                    deleteSession(s.id);
                    if (s.id === currentSessionId) {
                        const remaining = getSessionsIndex();
                        if (remaining.length > 0) switchSession(remaining[remaining.length - 1].id);
                        else { const nid = createNewSession(); switchSession(nid); }
                    } else {
                        renderSessionTabs();
                    }
                };
                tab.appendChild(del);
            }
            tab.onclick = () => { if (s.id !== currentSessionId) switchSession(s.id); };
            const nameSpan = tab.querySelector('.cb-session-tab-name');
            if (nameSpan) {
                nameSpan.ondblclick = e => {
                    e.stopPropagation();
                    const input = document.createElement('input');
                    input.className = 'cb-session-tab-edit';
                    input.value = s.name;
                    input.maxLength = 20;
                    const commit = () => {
                        const newName = input.value.trim() || s.name;
                        s.name = newName;
                        const idx = getSessionsIndex();
                        const found = idx.find(x => x.id === s.id);
                        if (found) { found.name = newName; saveSessionsIndex(idx); }
                        renderSessionTabs();
                    };
                    input.onblur = commit;
                    input.onkeydown = ev => { if (ev.key === 'Enter') commit(); if (ev.key === 'Escape') renderSessionTabs(); };
                    nameSpan.replaceWith(input);
                    input.focus();
                    input.select();
                };
            }
            container.appendChild(tab);
        });
        const addBtn = document.createElement('button');
        addBtn.className = 'cb-session-tab cb-session-add';
        addBtn.textContent = '+';
        addBtn.title = '새 대화';
        addBtn.onclick = () => {
            if (getSessionsIndex().length >= MAX_SESSIONS) { Toolbox.showToast('최대 세션 수에 도달', 'error'); return; }
            saveSession();
            const nid = createNewSession();
            switchSession(nid);
        };
        container.appendChild(addBtn);
    }

    /* ===== 빌드 ===== */
    function buildChat(container) {
        Mdd.linePreset('tool_run', { msg: '대화 상대가 필요해요?' });

        container.innerHTML = `
            <div class="cb-outer">
            <div class="cb-layout">
                <aside class="cb-sidebar cb-sidebar-left" aria-label="연결 및 옵션">
                    <div class="cb-sidebar-header">
                        <p class="cb-panel-heading">연결 · 모델</p>
                        <div class="field-group">
                            <label class="field-label">🔑 API 키</label>
                            <div style="display:flex;gap:8px;align-items:center;justify-content:space-between;">
                            <div style="font-size:var(--font-size-xs);color:var(--text-tertiary);">
                                    프로필: <strong id="cbActiveProfileName" style="color:var(--text-secondary);">${typeof Gemini !== 'undefined' ? (Gemini.getActiveProfileName() || '기본') : '-'}</strong>
                                </div>
                                <button class="btn btn-ghost" type="button" onclick="Toolbox.switchPage('user'); Toolbox.switchTab('user-settings');">설정</button>
                            </div>
                        </div>
                        <label class="cb-model-label">모델</label>
                        <select id="cbModelSelect" style="font-size:var(--font-size-xs);padding:6px 8px;width:100%;"></select>
                    </div>

                    <div class="cb-options">
                        <p class="cb-panel-heading" style="margin-bottom:8px;">대화 옵션</p>
                        <div class="cb-option-row">
                            <label>웹 검색</label>
                            <label class="cb-toggle"><input type="checkbox" id="cbWebSearch"><span class="cb-toggle-slider"></span></label>
                        </div>
                        <div class="cb-option-row">
                            <label>메모리</label>
                            <label class="cb-toggle"><input type="checkbox" id="cbMemory" checked><span class="cb-toggle-slider"></span></label>
                        </div>
                        <div class="cb-option-row cb-temperature-row">
                            <label>Temperature <span id="cbTempValue">0.8</span></label>
                        </div>
                        <div class="cb-option-row" style="margin-top:2px;">
                            <input type="range" id="cbTemperature" min="0" max="2" step="0.1" value="0.8" style="width:100%;">
                        </div>
                    </div>
                </aside>

                <div class="cb-chat-stage">
                <div class="cb-chat" style="position:relative;">
                    <div class="cb-shortcuts-overlay" id="cbShortcutsOverlay">
                        <div class="cb-shortcuts-panel">
                            <h3>⌨️ 키보드 단축키</h3>
                            <div class="cb-shortcut-row"><span>메시지 전송</span><span class="cb-shortcut-key">Enter</span></div>
                            <div class="cb-shortcut-row"><span>줄바꿈</span><span class="cb-shortcut-key">Shift + Enter</span></div>
                            <div class="cb-shortcut-row"><span>대화 검색</span><span class="cb-shortcut-key">Ctrl + F</span></div>
                            <div class="cb-shortcut-row"><span>새 세션</span><span class="cb-shortcut-key">Ctrl + N</span></div>
                            <div class="cb-shortcut-row"><span>단축키 안내</span><span class="cb-shortcut-key">Ctrl + /</span></div>
                            <div style="margin-top:12px;text-align:center;">
                                <button class="btn btn-ghost" onclick="document.getElementById('cbShortcutsOverlay').classList.remove('open')">닫기</button>
                            </div>
                        </div>
                    </div>
                    <div class="cb-chat-header">
                        <span class="cb-chat-header-title" id="cbChatTitle">💬 챗봇</span>
                        <div class="cb-chat-header-actions">
                            <button class="btn btn-ghost" id="cbShortcutsBtn" title="키보드 단축키 (Ctrl+/)">⌨️</button>
                            <button class="btn btn-ghost" id="cbSearchToggle" title="대화 검색 (Ctrl+F)">🔍</button>
                            <button class="btn btn-ghost" onclick="window._cb.importChat()">가져오기</button>
                            <button class="btn btn-ghost" onclick="window._cb.exportChat('txt')">TXT</button>
                            <button class="btn btn-ghost" onclick="window._cb.exportChat('json')">JSON</button>
                            <button class="btn btn-ghost" onclick="window._cb.clearChat()">초기화</button>
                        </div>
                    </div>
                    <div class="cb-session-bar" id="cbSessionTabs"></div>
                    <div class="cb-search-bar" id="cbSearchBar">
                        <input type="text" id="cbSearchInput" placeholder="대화 내용 검색...">
                        <span class="cb-search-nav" id="cbSearchNav"></span>
                        <button class="btn btn-ghost" id="cbSearchPrev" title="이전">▲</button>
                        <button class="btn btn-ghost" id="cbSearchNext" title="다음">▼</button>
                        <button class="btn btn-ghost" id="cbSearchClose" title="닫기">✕</button>
                    </div>
                    <div class="cb-messages" id="cbMessages" role="log" aria-live="polite" aria-label="대화 내용"></div>
                    <div class="cb-input-area" id="cbInputArea">
                        <div class="cb-attach-area" id="cbAttachArea">
                            <button class="cb-attach-btn" id="cbAttachBtn" title="이미지 첨부">📎</button>
                            <input type="file" id="cbFileInput" accept="image/*" multiple style="display:none">
                        </div>
                        <div class="cb-input-row">
                            <textarea id="cbInput" placeholder="메시지를 입력하세요... (이미지를 드래그하거나 붙여넣기 가능)"></textarea>
                            <button class="cb-mic-btn" id="cbMicBtn" title="음성 입력" aria-label="음성 입력">🎤</button>
                            <button class="cb-send-btn" id="cbSendBtn" onclick="window._cb.send()" aria-label="메시지 전송">➤</button>
                            <button class="cb-stop-btn" id="cbStopBtn" style="display:none" onclick="window._cb.stopStream()" aria-label="응답 중지">■ 중지</button>
                        </div>
                        <div class="cb-token-bar">
                            <span id="cbTokenDisplay">Tokens: 0</span>
                            <span>AI의 응답은 정확하지 않을 수 있습니다.</span>
                        </div>
                    </div>
                </div>
                </div>

                <aside class="cb-sidebar cb-sidebar-right" aria-label="캐릭터 및 시스템 프롬프트">
                    <div class="cb-character-block" id="cbCharacterBlock">
                        <p class="cb-panel-heading" style="margin-bottom:8px;">캐릭터 (RP)</p>
                        <div class="cb-option-row" style="margin-bottom:6px;">
                            <label>캐릭터 반영</label>
                            <label class="cb-toggle"><input type="checkbox" id="cbCharUse" checked><span class="cb-toggle-slider"></span></label>
                        </div>
                        <div class="cb-option-row" style="margin-bottom:6px;">
                            <label>이미지 자동</label>
                            <label class="cb-toggle"><input type="checkbox" id="cbCharAutoImage"><span class="cb-toggle-slider"></span></label>
                        </div>
                        <div class="cb-char-profile-wrap">
                            <button type="button" class="cb-char-profile-btn" id="cbCharProfileOpen" title="캐릭터 편집" aria-label="캐릭터 편집 열기">
                                <img id="cbCharProfileAvatar" class="cb-char-profile-avatar" alt="" width="72" height="72" decoding="async" style="display:none">
                                <span id="cbCharProfilePlaceholder" class="cb-char-profile-placeholder">👤</span>
                            </button>
                            <p class="cb-char-profile-name" id="cbCharProfileName">—</p>
                        </div>
                    </div>
                    <div class="cb-sysprompt">
                        <p class="cb-panel-heading" style="margin-bottom:8px;">추가 지시</p>
                        <label class="cb-mini" style="margin-top:0;">프리셋</label>
                        <select id="cbSystemPreset" style="font-size:var(--font-size-xs);padding:6px 8px;margin-bottom:8px;width:100%;">
                            <option value="">직접 입력 (아래 텍스트)</option>
                            <option value="__none__">사용 안 함</option>
                            <option value="default">기본 (친절한 어시스턴트)</option>
                            <option value="writer">작가 (소설/시나리오)</option>
                            <option value="translator">번역가</option>
                            <option value="codereview">코드 리뷰</option>
                            <option value="summarizer">요약봇</option>
                            <option value="tutor">과외 선생님</option>
                            <option value="hodulgap">호들갑 (띄어쓰기 없이)</option>
                        </select>
                        <textarea id="cbSystemPrompt" placeholder="AI의 역할, 성격, 답변 스타일 등을 지정하세요...">당신은 친절하고 유용한 AI 어시스턴트입니다. 한국어로 답변해주세요.</textarea>
                    </div>
                </aside>
            </div>

                <div id="cbCharEditModal" class="cb-modal-root" hidden aria-hidden="true">
                    <div class="cb-modal-backdrop" id="cbCharEditBackdrop" tabindex="-1"></div>
                    <div class="cb-modal-dialog" role="dialog" aria-modal="true" aria-labelledby="cbCharEditTitle">
                        <div class="cb-modal-header">
                            <h2 class="cb-modal-title" id="cbCharEditTitle">캐릭터 편집</h2>
                            <button type="button" class="cb-modal-close" id="cbCharEditClose" aria-label="닫기">×</button>
                        </div>
                        <div class="cb-modal-body cb-char-modal-body">
                            <label class="cb-mini" style="margin-top:0;">캐릭터 선택</label>
                            <select id="cbCharacterSelect" style="font-size:var(--font-size-xs);padding:6px 8px;width:100%;margin-top:4px;"></select>
                            <label class="cb-mini">이미지 모델</label>
                            <select id="cbCharImageModel" style="font-size:var(--font-size-xs);padding:6px 8px;width:100%;margin-top:4px;"></select>
                            <label class="cb-mini">이름</label>
                            <input type="text" id="cbCharName" maxlength="80">
                            <label class="cb-mini">플레이어 ({{user}})</label>
                            <input type="text" id="cbCharUserName" maxlength="80">
                            <label class="cb-mini">플레이어 메모</label>
                            <textarea id="cbCharUserNote" rows="2"></textarea>
                            <label class="cb-mini">외형 (영어 키워드 권장)</label>
                            <textarea id="cbCharVisual" rows="2"></textarea>
                            <label class="cb-mini">설정</label>
                            <textarea id="cbCharDesc" rows="2"></textarea>
                            <label class="cb-mini">성격</label>
                            <textarea id="cbCharPersonality" rows="2"></textarea>
                            <label class="cb-mini">상황</label>
                            <textarea id="cbCharScenario" rows="2"></textarea>
                            <label class="cb-mini">첫 인사 (봇)</label>
                            <textarea id="cbCharFirstMes" rows="2"></textarea>
                            <label class="cb-mini">참조 이미지</label>
                            <div class="cb-char-row">
                                <input type="file" id="cbCharRefFile" accept="image/*" style="font-size:var(--font-size-2xs);max-width:160px;">
                                <button type="button" class="btn btn-ghost" id="cbCharRefClear" style="font-size:var(--font-size-2xs);padding:4px 8px;">비우기</button>
                                <img id="cbCharRefThumb" class="cb-char-ref-thumb" alt="" style="display:none;">
                            </div>
                            <div class="cb-char-row">
                                <button type="button" class="btn btn-ghost" id="cbCharSave" style="font-size:var(--font-size-xs);padding:4px 10px;">저장</button>
                                <button type="button" class="btn btn-ghost" id="cbCharNew" style="font-size:var(--font-size-xs);padding:4px 10px;">새 캐릭터</button>
                                <button type="button" class="btn btn-ghost" id="cbCharDel" style="font-size:var(--font-size-xs);padding:4px 10px;color:var(--error);">삭제</button>
                                <button type="button" class="btn btn-ghost" id="cbCharFirstBtn" style="font-size:var(--font-size-xs);padding:4px 10px;">첫 인사</button>
                            </div>
                            <label class="cb-mini">캐릭터 카드 (JSON / PNG)</label>
                            <label class="cb-char-import-overwrite" style="display:flex;align-items:center;gap:8px;margin-top:6px;font-size:var(--font-size-xs);color:var(--text-secondary);cursor:pointer;">
                                <input type="checkbox" id="cbCharImportOverwrite" style="width:auto;margin:0;">
                                <span>현재 선택 캐릭터에 덮어쓰기 (끄면 항상 새 캐릭터로 추가)</span>
                            </label>
                            <div class="cb-char-row" style="margin-top:8px;">
                                <input type="file" id="cbCharImportFile" accept=".json,application/json,.png,image/png" style="display:none">
                                <button type="button" class="btn btn-ghost" id="cbCharImportBtn" style="font-size:var(--font-size-xs);padding:4px 10px;">가져오기</button>
                                <button type="button" class="btn btn-ghost" id="cbCharExportBtn" style="font-size:var(--font-size-xs);padding:4px 10px;">내보내기</button>
                            </div>
                            <p style="font-size:var(--font-size-2xs);color:var(--text-tertiary);margin:6px 0 0;line-height:1.45;">SillyTavern Character Card JSON·PNG(V2 등) · 이 페이지 <code style="font-size:1em;">karmochat_character_v1</code> JSON 재가져오기 가능</p>
                        </div>
                    </div>
                </div>
            </div>
        `;

        requestAnimationFrame(() => {
            // 모델 셀렉트
            const sel = document.getElementById('cbModelSelect');
            if (sel) {
                Gemini.MODELS.gemini.forEach(m => {
                    const o = document.createElement('option');
                    o.value = m.id; o.textContent = m.name;
                    if (m.isDefault) o.selected = true;
                    sel.appendChild(o);
                });
            }

            // 설정 복원
            const savedModel = Toolbox.getPref('cb_model');
            if (savedModel && sel) { sel.value = savedModel; }
            if (sel) sel.addEventListener('change', () => Toolbox.setPref('cb_model', sel.value));

            // 시스템 프롬프트 프리셋 (__none__ / 직접입력 / 명명 프리셋)
            const presetSel = document.getElementById('cbSystemPreset');
            const sysPromptTa = document.getElementById('cbSystemPrompt');
            const savedPreset = Toolbox.getPref('cb_preset');
            function applySystemPresetUi() {
                if (!presetSel || !sysPromptTa) return;
                const v = presetSel.value;
                if (v === '__none__') {
                    sysPromptTa.value = '';
                    sysPromptTa.readOnly = true;
                    sysPromptTa.placeholder = '「사용 안 함」일 때는 API에 추가 지시가 붙지 않습니다.';
                } else {
                    sysPromptTa.readOnly = false;
                    sysPromptTa.placeholder = 'AI의 역할, 성격, 답변 스타일 등을 지정하세요...';
                    if (SYSTEM_PROMPT_PRESETS[v]) sysPromptTa.value = SYSTEM_PROMPT_PRESETS[v];
                }
            }
            if (presetSel && sysPromptTa) {
                if (typeof savedPreset === 'string') {
                    const hasOpt = Array.from(presetSel.options).some(o => o.value === savedPreset);
                    if (hasOpt) presetSel.value = savedPreset;
                }
                applySystemPresetUi();
                presetSel.addEventListener('change', () => {
                    Toolbox.setPref('cb_preset', presetSel.value);
                    applySystemPresetUi();
                });
            }

            // Temperature 슬라이더 표시 갱신
            const tempSlider = document.getElementById('cbTemperature');
            const tempValueEl = document.getElementById('cbTempValue');
            const savedTemp = Toolbox.getPref('cb_temperature');
            if (tempSlider && tempValueEl) {
                if (savedTemp !== undefined) { tempSlider.value = savedTemp; tempValueEl.textContent = savedTemp; }
                tempSlider.addEventListener('input', () => { tempValueEl.textContent = tempSlider.value; Toolbox.setPref('cb_temperature', tempSlider.value); });
            }

            // 이미지 첨부
            const attachBtn = document.getElementById('cbAttachBtn');
            const fileInput = document.getElementById('cbFileInput');
            const inputArea = document.getElementById('cbInputArea');
            if (attachBtn && fileInput) {
                attachBtn.onclick = () => fileInput.click();
                fileInput.onchange = async () => {
                    for (const f of fileInput.files) {
                        if (f.type.startsWith('image/')) addPendingImage(await fileToBase64(f));
                    }
                    fileInput.value = '';
                };
            }
            if (inputArea) {
                inputArea.ondragover = e => { e.preventDefault(); inputArea.classList.add('drag-over'); };
                inputArea.ondragleave = () => inputArea.classList.remove('drag-over');
                inputArea.ondrop = async e => {
                    e.preventDefault(); inputArea.classList.remove('drag-over');
                    for (const f of e.dataTransfer.files) {
                        if (f.type.startsWith('image/')) addPendingImage(await fileToBase64(f));
                    }
                };
            }

            // Enter 키 + 클립보드 이미지 붙여넣기
            const chatInput = document.getElementById('cbInput');
            if (chatInput) {
                chatInput.addEventListener('keydown', e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        window._cb.send();
                    }
                });
                chatInput.addEventListener('paste', async e => {
                    const items = e.clipboardData?.items;
                    if (!items) return;
                    for (const item of items) {
                        if (item.type.startsWith('image/')) {
                            e.preventDefault();
                            const file = item.getAsFile();
                            if (file) addPendingImage(await fileToBase64(file));
                        }
                    }
                });
            }

            // 음성 입력 (Web Speech API)
            const micBtn = document.getElementById('cbMicBtn');
            if (micBtn) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (SpeechRecognition) {
                    const recognition = new SpeechRecognition();
                    recognition.lang = 'ko-KR';
                    recognition.interimResults = true;
                    recognition.continuous = false;
                    let isRecording = false;
                    let finalTranscript = '';

                    micBtn.addEventListener('click', () => {
                        if (isRecording) { recognition.stop(); return; }
                        finalTranscript = '';
                        recognition.start();
                    });
                    recognition.onstart = () => {
                        isRecording = true;
                        micBtn.classList.add('recording');
                        micBtn.title = '녹음 중... 클릭하여 중지';
                    };
                    recognition.onend = () => {
                        isRecording = false;
                        micBtn.classList.remove('recording');
                        micBtn.title = '음성 입력';
                        if (finalTranscript && chatInput) {
                            chatInput.value += (chatInput.value ? ' ' : '') + finalTranscript;
                            chatInput.focus();
                        }
                    };
                    recognition.onresult = (e) => {
                        let interim = '';
                        for (let i = e.resultIndex; i < e.results.length; i++) {
                            if (e.results[i].isFinal) finalTranscript += e.results[i][0].transcript;
                            else interim += e.results[i][0].transcript;
                        }
                    };
                    recognition.onerror = (e) => {
                        if (e.error !== 'aborted') Toolbox.showToast('음성 인식 오류: ' + e.error, 'error');
                    };
                } else {
                    micBtn.style.display = 'none';
                }
            }

            // 다중 세션 초기화
            const index = getSessionsIndex();
            if (index.length === 0) {
                currentSessionId = createNewSession('대화 1');
            } else {
                currentSessionId = index[index.length - 1].id;
            }
            renderSessionTabs();

            // 세션 데이터 먼저 로드 → 캐릭터 선택 복원에 사용
            const msgs = document.getElementById('cbMessages');
            const sessionLoaded = msgs && loadSession(currentSessionId);
            initCharacterUi();
            if (msgs) {
                if (sessionLoaded && chatHistory.length > 0) {
                    chatHistory.forEach(msg => {
                        const role = msg.role === 'user' ? 'user' : 'bot';
                        const text = msg.parts?.[0]?.text || '';
                        if (text) appendMsg(role, text, false);
                    });
                } else {
                    appendMsg('bot', '안녕하세요! 무엇을 도와드릴까요? 😊');
                }
                syncCharacterSelectAfterSessionLoad();
                msgs.scrollTop = msgs.scrollHeight;
            }

            // 대화 검색
            const searchToggle = document.getElementById('cbSearchToggle');
            const searchBar = document.getElementById('cbSearchBar');
            const searchInput = document.getElementById('cbSearchInput');
            const searchNav = document.getElementById('cbSearchNav');
            const searchPrev = document.getElementById('cbSearchPrev');
            const searchNext = document.getElementById('cbSearchNext');
            const searchClose = document.getElementById('cbSearchClose');
            if (searchToggle && searchBar && searchInput) {
                let searchResults = [];
                let searchIdx = -1;

                function toggleSearch() {
                    const open = searchBar.classList.toggle('open');
                    if (open) { searchInput.focus(); }
                    else { clearHighlights(); searchInput.value = ''; searchNav.textContent = ''; }
                }

                function clearHighlights() {
                    if (!msgs) return;
                    msgs.querySelectorAll('.cb-search-highlight').forEach(el => {
                        const parent = el.parentNode;
                        parent.replaceChild(document.createTextNode(el.textContent), el);
                        parent.normalize();
                    });
                    searchResults = [];
                    searchIdx = -1;
                }

                function doSearch() {
                    clearHighlights();
                    const q = searchInput.value.trim();
                    if (!q || !msgs) { searchNav.textContent = ''; return; }
                    const regex = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
                    const walker = document.createTreeWalker(msgs, NodeFilter.SHOW_TEXT, null, false);
                    const matches = [];
                    while (walker.nextNode()) {
                        const node = walker.currentNode;
                        if (node.parentElement.closest('pre, code, .cb-code-header')) continue;
                        if (regex.test(node.textContent)) matches.push(node);
                        regex.lastIndex = 0;
                    }
                    matches.forEach(node => {
                        const text = node.textContent;
                        const parts = text.split(regex);
                        if (parts.length <= 1) return;
                        const frag = document.createDocumentFragment();
                        let m;
                        regex.lastIndex = 0;
                        let lastIdx = 0;
                        while ((m = regex.exec(text)) !== null) {
                            if (m.index > lastIdx) frag.appendChild(document.createTextNode(text.slice(lastIdx, m.index)));
                            const mark = document.createElement('mark');
                            mark.className = 'cb-search-highlight';
                            mark.textContent = m[0];
                            frag.appendChild(mark);
                            lastIdx = regex.lastIndex;
                        }
                        if (lastIdx < text.length) frag.appendChild(document.createTextNode(text.slice(lastIdx)));
                        node.parentNode.replaceChild(frag, node);
                    });
                    searchResults = Array.from(msgs.querySelectorAll('.cb-search-highlight'));
                    searchIdx = searchResults.length > 0 ? 0 : -1;
                    updateSearchNav();
                }

                function updateSearchNav() {
                    if (searchResults.length === 0) { searchNav.textContent = '결과 없음'; return; }
                    searchNav.textContent = `${searchIdx + 1} / ${searchResults.length}`;
                    searchResults.forEach((el, i) => el.classList.toggle('current', i === searchIdx));
                    searchResults[searchIdx]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }

                function navigate(dir) {
                    if (searchResults.length === 0) return;
                    searchIdx = (searchIdx + dir + searchResults.length) % searchResults.length;
                    updateSearchNav();
                }

                searchToggle.addEventListener('click', toggleSearch);
                searchClose.addEventListener('click', toggleSearch);
                searchInput.addEventListener('input', doSearch);
                searchPrev.addEventListener('click', () => navigate(-1));
                searchNext.addEventListener('click', () => navigate(1));
                searchInput.addEventListener('keydown', e => {
                    if (e.key === 'Enter') { e.preventDefault(); navigate(e.shiftKey ? -1 : 1); }
                    if (e.key === 'Escape') toggleSearch();
                });

            }

            // 단축키 안내 버튼
            const shortcutsBtn = document.getElementById('cbShortcutsBtn');
            const shortcutsOverlay = document.getElementById('cbShortcutsOverlay');
            if (shortcutsBtn && shortcutsOverlay) {
                shortcutsBtn.addEventListener('click', () => shortcutsOverlay.classList.toggle('open'));
                shortcutsOverlay.addEventListener('click', e => {
                    if (e.target === shortcutsOverlay) shortcutsOverlay.classList.remove('open');
                });
            }

            // 글로벌 키보드 단축키 (단일 리스너)
            document.addEventListener('keydown', e => {
                const chatEl = document.querySelector('.cb-chat');
                if (!chatEl || chatEl.offsetParent === null) return;

                if (e.ctrlKey && e.key === 'f') {
                    e.preventDefault();
                    const sb = document.getElementById('cbSearchBar');
                    const si = document.getElementById('cbSearchInput');
                    if (sb && si) {
                        if (!sb.classList.contains('open')) sb.classList.add('open');
                        si.focus();
                    }
                }
                if (e.ctrlKey && e.key === '/') {
                    e.preventDefault();
                    shortcutsOverlay?.classList.toggle('open');
                }
                if (e.ctrlKey && e.key === 'n') {
                    e.preventDefault();
                    const idx = getSessionsIndex();
                    if (idx.length >= 10) { Toolbox.showToast('최대 10개 세션까지 생성 가능합니다.', 'error'); return; }
                    const nid = createNewSession(`대화 ${idx.length + 1}`);
                    switchSession(nid);
                }
                if (e.key === 'Escape') {
                    shortcutsOverlay?.classList.remove('open');
                }
            });

            // 랜덤 생성기 → 이야기 만들기 연동
            const chatbotPage = document.getElementById('page-chatbot');
            if (chatbotPage) {
                const checkStoryKeywords = function () {
                    try {
                        var raw = sessionStorage.getItem('toolbox_chatbot_story_keywords');
                        if (raw) {
                            sessionStorage.removeItem('toolbox_chatbot_story_keywords');
                            var keywords = JSON.parse(raw);
                            if (Array.isArray(keywords) && keywords.length > 0) {
                                var input = document.getElementById('cbInput');
                                var presetSel = document.getElementById('cbSystemPreset');
                                var sysPromptTa = document.getElementById('cbSystemPrompt');
                                if (presetSel && sysPromptTa && SYSTEM_PROMPT_PRESETS.writer) {
                                    presetSel.value = 'writer';
                                    Toolbox.setPref('cb_preset', 'writer');
                                    applySystemPresetUi();
                                }
                                if (input) {
                                    var prompt = '다음 키워드를 포함해서 짧은 이야기를 써줘: ' + keywords.join(', ');
                                    input.value = prompt;
                                    input.focus();
                                    Toolbox.showToast('키워드가 입력되었습니다. 전송 버튼을 누르세요.');
                                }
                            }
                        }
                    } catch (err) {}
                };
                var obs = new MutationObserver(checkStoryKeywords);
                obs.observe(chatbotPage, { attributes: true, attributeFilter: ['class'] });
                if (chatbotPage.classList.contains('active')) checkStoryKeywords();
            }
        });
    }

    /* ===== 경량 마크다운 파서 ===== */
    function renderMarkdown(text) {
        // 코드블록 보호 (```)
        const codeBlocks = [];
        let md = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            const cls = lang ? ` language-${lang}` : '';
            const langLabel = lang || 'code';
            const header = `<div class="cb-code-header"><span class="cb-code-lang">${escapeHtml(langLabel)}</span><button class="btn btn-ghost" onclick="navigator.clipboard.writeText(this.closest('.cb-code-block').querySelector('code').textContent).then(()=>Toolbox.showToast('복사됨'))">복사</button></div>`;
            codeBlocks.push(`<div class="cb-code-block">${header}<pre class="${cls.trim()}"><code class="${cls.trim()}">${escapeHtml(code.trimEnd())}</code></pre></div>`);
            return `%%CODEBLOCK_${codeBlocks.length - 1}%%`;
        });

        // 인라인 코드 보호
        const inlineCodes = [];
        md = md.replace(/`([^`]+)`/g, (_, code) => {
            inlineCodes.push(`<code>${escapeHtml(code)}</code>`);
            return `%%INLINE_${inlineCodes.length - 1}%%`;
        });

        // HTML 이스케이프 (코드블록 제외 영역)
        md = escapeHtml(md);

        // 볼드/이탤릭
        md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        md = md.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // 링크 [text](url) — javascript: 등 위험 프로토콜 차단
        md = md.replace(/\[(.+?)\]\((.+?)\)/g, (_, text, url) => {
            if (/^(https?:|mailto:|\/|#)/i.test(url)) return `<a href="${url}" target="_blank" rel="noopener">${text}</a>`;
            return `${text} (${url})`;
        });

        // 인용 (> )
        md = md.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

        // 수평선
        md = md.replace(/^---$/gm, '<hr>');

        // 테이블
        md = md.replace(/(^\|.+\|$\n?)+/gm, tableBlock => {
            const rows = tableBlock.trim().split('\n').filter(r => r.trim());
            if (rows.length < 2) return tableBlock;
            const isSep = r => /^\|[\s\-:|]+\|$/.test(r.trim());
            const parseRow = (r, tag) => {
                const cells = r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
                return '<tr>' + cells.map(c => `<${tag}>${c}</${tag}>`).join('') + '</tr>';
            };
            let html = '<table>';
            let headerDone = false;
            for (let i = 0; i < rows.length; i++) {
                if (isSep(rows[i])) { headerDone = true; continue; }
                if (!headerDone && i === 0 && rows.length > 1 && isSep(rows[1])) {
                    html += '<thead>' + parseRow(rows[i], 'th') + '</thead><tbody>';
                    headerDone = true; i++;
                    continue;
                }
                html += parseRow(rows[i], 'td');
            }
            html += '</tbody></table>';
            return html;
        });

        // 리스트
        md = md.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
        md = md.replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
        md = md.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        // 헤딩 (### ## #)
        md = md.replace(/^### (.+)$/gm, '<strong>$1</strong>');
        md = md.replace(/^## (.+)$/gm, '<strong>$1</strong>');
        md = md.replace(/^# (.+)$/gm, '<strong style="font-size:15px">$1</strong>');

        // 문단 처리
        md = md.replace(/\n\n+/g, '</p><p>');
        md = md.replace(/\n/g, '<br>');
        md = `<p>${md}</p>`;

        // 복원: 코드블록, 인라인코드
        codeBlocks.forEach((block, i) => { md = md.replace(`%%CODEBLOCK_${i}%%`, block); });
        inlineCodes.forEach((code, i) => { md = md.replace(`%%INLINE_${i}%%`, code); });

        // 빈 <p> 정리
        md = md.replace(/<p>\s*<\/p>/g, '');

        return md;
    }

    const escapeHtml = Toolbox.escapeHtml;

    /* ===== 헬퍼 ===== */
    function appendMsg(role, text, isError = false) {
        const msgs = document.getElementById('cbMessages');
        if (!msgs) return;
        const wrap = document.createElement('div');
        wrap.className = 'cb-msg-wrap';
        const div = document.createElement('div');
        div.className = `cb-msg cb-msg-${role}` + (isError ? ' cb-msg-error' : '');
        if (role === 'bot' && !isError) {
            div.innerHTML = renderMarkdown(text);
        } else {
            div.textContent = text;
        }
        const copyBtn = document.createElement('button');
        copyBtn.className = 'btn btn-ghost cb-msg-copy';
        copyBtn.type = 'button';
        copyBtn.textContent = '복사';
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(text).then(() => Toolbox.showToast('복사됨')).catch(() => {});
        };
        wrap.appendChild(div);
        wrap.appendChild(copyBtn);
        if (role === 'bot' && !isError && chatHistory.length > 0) {
            const regen = document.createElement('button');
            regen.className = 'btn btn-ghost';
            regen.textContent = '🔄 재생성';
            regen.onclick = () => window._cb.regenerate();
            wrap.appendChild(regen);
        }
        msgs.appendChild(wrap);
        if (role === 'bot' && !isError && typeof Prism !== 'undefined') {
            div.querySelectorAll('pre code[class*="language-"]').forEach(el => Prism.highlightElement(el));
        }
        msgs.scrollTop = msgs.scrollHeight;
    }

    function showTyping() {
        const msgs = document.getElementById('cbMessages');
        if (!msgs) return null;
        const id = 'cb-typing-' + Date.now();
        const div = document.createElement('div');
        div.id = id;
        div.className = 'cb-msg cb-msg-bot cb-typing';
        div.innerHTML = '<span></span><span></span><span></span>';
        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
        return id;
    }

    function removeTyping(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function updateTokens(usage) {
        if (!usage) return;
        const total = usage.totalTokenCount || 0;
        const display = document.getElementById('cbTokenDisplay');
        if (display) {
            display.textContent = `Tokens: ${total.toLocaleString()}`;
            display.style.color = 'var(--text-tertiary)';
        }
    }

    /* ===== 스트리밍 봇 메시지 헬퍼 ===== */
    function appendStreamMsg() {
        const msgs = document.getElementById('cbMessages');
        if (!msgs) return null;
        const wrap = document.createElement('div');
        wrap.className = 'cb-msg-wrap';
        const div = document.createElement('div');
        div.className = 'cb-msg cb-msg-bot';
        div.innerHTML = '<span class="cb-cursor-blink">▌</span>';
        const copyBtn = document.createElement('button');
        copyBtn.className = 'btn btn-ghost cb-msg-copy';
        copyBtn.type = 'button';
        copyBtn.textContent = '복사';
        wrap.appendChild(div);
        wrap.appendChild(copyBtn);
        msgs.appendChild(wrap);
        msgs.scrollTop = msgs.scrollHeight;
        return { div, copyBtn, wrap };
    }

    function finalizeStreamMsg(el, fullText) {
        if (!el) return;
        el.div.innerHTML = renderMarkdown(fullText);
        if (typeof Prism !== 'undefined') {
            el.div.querySelectorAll('pre code[class*="language-"]').forEach(c => Prism.highlightElement(c));
        }
        el.copyBtn.onclick = () => {
            navigator.clipboard.writeText(fullText).then(() => Toolbox.showToast('복사됨')).catch(() => {});
        };
        const regen = document.createElement('button');
        regen.className = 'btn btn-ghost';
        regen.textContent = '🔄 재생성';
        regen.onclick = () => window._cb.regenerate();
        el.wrap.appendChild(regen);
    }

    let currentStreamAbort = null;

    /* ===== 액션 ===== */
    window._cb = {
        async send() {
            const input = document.getElementById('cbInput');
            const text = input?.value.trim();
            if (!text) return;
            if (!Gemini.requireApiKey()) return;

            appendMsg('user', text + (pendingImages.length ? ` [📎 이미지 ${pendingImages.length}장]` : ''));
            input.value = '';

            const parts = [{ text }];
            pendingImages.forEach(img => {
                parts.push({ inlineData: { mimeType: img.mimeType, data: img.base64 } });
            });
            pendingImages = [];
            renderAttachThumbs();
            chatHistory.push({ role: 'user', parts });

            const useMemory = document.getElementById('cbMemory')?.checked;
            const useWebSearch = document.getElementById('cbWebSearch')?.checked;
            const systemPrompt = assembleSystemPrompt({ useMemory });

            const modelSel = document.getElementById('cbModelSelect');
            const modelId = modelSel?.value || Gemini.getDefaultModel('gemini');
            const tempInput = document.getElementById('cbTemperature');
            const temperature = tempInput ? parseFloat(tempInput.value) : 0.8;

            const streamEl = appendStreamMsg();
            currentStreamAbort = new AbortController();
            const sendBtn = document.getElementById('cbSendBtn');
            const stopBtn = document.getElementById('cbStopBtn');
            if (sendBtn) sendBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = '';

            try {
                const stream = await Gemini.callChatStream(chatHistory, systemPrompt, modelId, {
                    webSearch: useWebSearch,
                    temperature,
                    signal: currentStreamAbort.signal
                });

                let fullText = '';
                let lastUsage = null;
                let renderPending = false;
                for await (const chunk of stream.chunks()) {
                    fullText += chunk.text;
                    if (chunk.usage) lastUsage = chunk.usage;
                    if (!renderPending) {
                        renderPending = true;
                        requestAnimationFrame(() => {
                            renderPending = false;
                            streamEl.div.innerHTML = renderMarkdown(displayTextForStream(fullText)) + '<span class="cb-cursor-blink">▌</span>';
                            const msgs = document.getElementById('cbMessages');
                            if (msgs) msgs.scrollTop = msgs.scrollHeight;
                        });
                    }
                }

                let responseText = fullText;
                if (useMemory) {
                    const summaryMatch = responseText.match(/\{\{\{(.*?)\}\}\}/s);
                    if (summaryMatch) {
                        conversationSummary = summaryMatch[1].trim();
                        responseText = responseText.replace(/\{\{\{.*?\}\}\}/s, '').trim();
                    }
                }

                const imgParsed = extractKarmoImage(responseText);
                responseText = imgParsed.cleanText;
                const autoImgOn = document.getElementById('cbCharAutoImage')?.checked;
                const charForImg = document.getElementById('cbCharacterSelect')?.value
                    ? getCharacterById(document.getElementById('cbCharacterSelect').value)
                    : null;

                finalizeStreamMsg(streamEl, responseText);
                updateTokens(lastUsage);

                chatHistory.push({ role: 'model', parts: [{ text: responseText }] });
                saveSession();
                Toolbox.recordUsage('chat', lastUsage?.totalTokenCount || 0);
                Mdd.linePreset('success', { mood: 'happy', msg: '대답 완료해요!' });

                if (autoImgOn && imgParsed.spec?.show && charForImg) {
                    void appendCharacterImageAfterMessage(streamEl.wrap, charForImg, imgParsed.spec);
                }

            } catch (e) {
                if (streamEl.wrap.parentNode) streamEl.wrap.remove();
                if (e.message !== '요청이 취소되었습니다.') {
                    appendMsg('bot', `오류: ${e.message}`, true);
                    Toolbox.showToast(e.message || '오류', 'error', e);
                    Mdd.linePreset('error', { msg: '에러예요...' });
                }
                console.error('Chat Error:', e);
            } finally {
                currentStreamAbort = null;
                if (sendBtn) sendBtn.style.display = '';
                if (stopBtn) stopBtn.style.display = 'none';
            }
        },

        stopStream() {
            if (currentStreamAbort) {
                currentStreamAbort.abort();
                Toolbox.showToast('스트리밍 중지');
            }
        },

        async regenerate() {
            if (chatHistory.length < 2) return;
            const lastModel = chatHistory[chatHistory.length - 1];
            if (lastModel?.role === 'model') {
                chatHistory.pop();
            }
            const msgs = document.getElementById('cbMessages');
            if (msgs) {
                const wraps = msgs.querySelectorAll('.cb-msg-wrap');
                const last = wraps[wraps.length - 1];
                if (last) last.remove();
            }
            saveSession();

            const lastUser = chatHistory[chatHistory.length - 1];
            if (!lastUser || lastUser.role !== 'user') return;

            const useMemory = document.getElementById('cbMemory')?.checked;
            const useWebSearch = document.getElementById('cbWebSearch')?.checked;
            const systemPrompt = assembleSystemPrompt({ useMemory });

            const modelSel = document.getElementById('cbModelSelect');
            const modelId = modelSel?.value || Gemini.getDefaultModel('gemini');
            const tempInput = document.getElementById('cbTemperature');
            const temperature = tempInput ? parseFloat(tempInput.value) : 0.8;

            const streamEl = appendStreamMsg();
            currentStreamAbort = new AbortController();
            const sendBtn = document.getElementById('cbSendBtn');
            const stopBtn = document.getElementById('cbStopBtn');
            if (sendBtn) sendBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = '';

            try {
                const stream = await Gemini.callChatStream(chatHistory, systemPrompt, modelId, {
                    webSearch: useWebSearch, temperature, signal: currentStreamAbort.signal
                });
                let fullText = '';
                let lastUsage = null;
                let renderPending2 = false;
                for await (const chunk of stream.chunks()) {
                    fullText += chunk.text;
                    if (chunk.usage) lastUsage = chunk.usage;
                    if (!renderPending2) {
                        renderPending2 = true;
                        requestAnimationFrame(() => {
                            renderPending2 = false;
                            streamEl.div.innerHTML = renderMarkdown(displayTextForStream(fullText)) + '<span class="cb-cursor-blink">▌</span>';
                            const msgsEl = document.getElementById('cbMessages');
                            if (msgsEl) msgsEl.scrollTop = msgsEl.scrollHeight;
                        });
                    }
                }
                let responseText = fullText;
                if (useMemory) {
                    const m = responseText.match(/\{\{\{(.*?)\}\}\}/s);
                    if (m) { conversationSummary = m[1].trim(); responseText = responseText.replace(/\{\{\{.*?\}\}\}/s, '').trim(); }
                }
                const imgParsedR = extractKarmoImage(responseText);
                responseText = imgParsedR.cleanText;
                const autoImgR = document.getElementById('cbCharAutoImage')?.checked;
                const charForImgR = document.getElementById('cbCharacterSelect')?.value
                    ? getCharacterById(document.getElementById('cbCharacterSelect').value)
                    : null;
                finalizeStreamMsg(streamEl, responseText);
                updateTokens(lastUsage);
                chatHistory.push({ role: 'model', parts: [{ text: responseText }] });
                saveSession();
                Toolbox.recordUsage('chat', lastUsage?.totalTokenCount || 0);
                Mdd.linePreset('success', { mood: 'happy', msg: '다시 대답했어요!' });
                if (autoImgR && imgParsedR.spec?.show && charForImgR) {
                    void appendCharacterImageAfterMessage(streamEl.wrap, charForImgR, imgParsedR.spec);
                }
            } catch (e) {
                if (streamEl.wrap.parentNode) streamEl.wrap.remove();
                if (e.message !== '요청이 취소되었습니다.') {
                    appendMsg('bot', `오류: ${e.message}`, true);
                    Toolbox.showToast(e.message || '오류', 'error', e);
                }
            } finally {
                currentStreamAbort = null;
                if (sendBtn) sendBtn.style.display = '';
                if (stopBtn) stopBtn.style.display = 'none';
            }
        },

        clearChat() {
            chatHistory = [];
            conversationSummary = '';
            pendingImages = [];
            renderAttachThumbs();
            if (currentSessionId) {
                try { sessionStorage.removeItem(CHATBOT_SESSION_PREFIX + currentSessionId); } catch (e) {}
            }
            const msgs = document.getElementById('cbMessages');
            if (msgs) {
                msgs.innerHTML = '';
                appendMsg('bot', '대화가 초기화되었습니다. 무엇을 도와드릴까요?');
            }
            Toolbox.showToast('대화 초기화 완료');
            Mdd.linePreset('tool_run', { mood: 'idle', msg: '새로 시작이에요!' });
        },

        importChat() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.txt,.json';
            input.onchange = async () => {
                const file = input.files?.[0];
                if (!file) return;
                try {
                    const text = await file.text();
                    let imported = [];
                    if (file.name.endsWith('.json')) {
                        const data = JSON.parse(text);
                        if (Array.isArray(data)) imported = data;
                        else throw new Error('잘못된 JSON 형식');
                    } else {
                        const blocks = text.split(/\n\n/).filter(b => b.trim());
                        for (const block of blocks) {
                            const m = block.match(/^\[(You|AI)\]\n([\s\S]*)$/);
                            if (m) {
                                imported.push({
                                    role: m[1] === 'You' ? 'user' : 'model',
                                    parts: [{ text: m[2] }]
                                });
                            }
                        }
                    }
                    if (imported.length === 0) { Toolbox.showToast('가져올 대화를 찾을 수 없습니다.', 'error'); return; }
                    const sessionName = file.name.replace(/\.[^.]+$/, '');
                    const newId = createNewSession(sessionName);
                    currentSessionId = newId;
                    chatHistory = imported;
                    saveSession();
                    renderSessionTabs();
                    const msgs = document.getElementById('cbMessages');
                    if (msgs) {
                        msgs.innerHTML = '';
                        chatHistory.forEach(msg => {
                            const role = msg.role === 'user' ? 'user' : 'bot';
                            const t = msg.parts?.[0]?.text || '';
                            if (t) appendMsg(role, t, false);
                        });
                        msgs.scrollTop = msgs.scrollHeight;
                    }
                    Toolbox.showToast(`${imported.length}개 메시지를 가져왔습니다.`);
                } catch (e) {
                    Toolbox.showToast('가져오기 실패: ' + e.message, 'error');
                }
            };
            input.click();
        },

        exportChat(format = 'txt') {
            if (chatHistory.length === 0) {
                Toolbox.showToast('내보낼 대화가 없습니다.', 'error');
                return;
            }
            const date = new Date().toISOString().slice(0, 10);
            let blob, filename;
            if (format === 'json') {
                blob = new Blob([JSON.stringify(chatHistory, null, 2)], { type: 'application/json;charset=utf-8' });
                filename = `chat-export-${date}.json`;
            } else {
                const lines = chatHistory.map(m => {
                    const role = m.role === 'user' ? 'You' : 'AI';
                    const text = m.parts?.[0]?.text || '';
                    return `[${role}]\n${text}`;
                });
                blob = new Blob([lines.join('\n\n')], { type: 'text/plain;charset=utf-8' });
                filename = `chat-export-${date}.txt`;
            }
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = filename;
            a.click();
            URL.revokeObjectURL(a.href);
            Toolbox.showToast(`대화 내보내기 완료 (${format.toUpperCase()})`);
        }
    };

    /* ===== 위젯 등록 ===== */
    Toolbox.register({
        ...Toolbox.getLazyWidgetPublicMeta('chatbot'),
        tabs: [{ id: 'chatbot-main', label: '채팅', build: buildChat }]
    });
})();
