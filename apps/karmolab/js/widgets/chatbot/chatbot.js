(function () {
    /* ===== CSS 주입 ===== */
    Mdd.injectCSS('chatbot', `
        .cb-layout { display:flex; gap:20px; flex:1; min-height:0; }

        /* 사이드바 */
        .cb-sidebar { width:260px; flex-shrink:0; display:flex; flex-direction:column; border:1px solid var(--border); background:var(--bg-secondary); overflow:hidden; }
        .cb-sidebar-header { padding:16px; border-bottom:1px solid var(--border); }
        .cb-sidebar-header h3 { font-size:var(--font-size-sm); font-weight:600; color:var(--text-primary); margin-bottom:12px; }

        .cb-api-row { display:flex; gap:6px; align-items:center; }
        .cb-api-row input { flex:1; font-size:var(--font-size-xs); padding:6px 8px; }
        .cb-key-status { font-size:var(--font-size-2xs); color:var(--text-tertiary); margin-top:4px; }

        .cb-model-label { font-size:var(--font-size-xs); font-weight:500; color:var(--text-secondary); margin:12px 0 4px; display:block; }

        .cb-options { padding:12px 16px; border-bottom:1px solid var(--border); }
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

        /* 시스템 프롬프트 */
        .cb-sysprompt { flex:1; padding:12px 16px; overflow-y:auto; display:flex; flex-direction:column; }
        .cb-sysprompt label { font-size:var(--font-size-xs); font-weight:500; color:var(--text-secondary); margin-bottom:4px; display:block; }
        .cb-sysprompt textarea {
            flex:1; min-height:120px; font-size:var(--font-size-xs); line-height:1.5;
            resize:none; padding:8px; background:var(--bg-tertiary);
        }

        /* 채팅 영역 */
        .cb-chat { flex:1; display:flex; flex-direction:column; background:var(--bg-primary); border:1px solid var(--border); overflow:hidden; }

        .cb-chat-header {
            padding:12px 16px; border-bottom:1px solid var(--border);
            display:flex; align-items:center; justify-content:space-between;
            background:var(--bg-secondary);
        }
        .cb-chat-header-title { font-size:var(--font-size-sm); font-weight:600; color:var(--text-primary); }

        .cb-messages {
            flex:1; overflow-y:auto; padding:16px; display:flex;
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

        @media (max-width:768px) {
            .cb-layout { flex-direction:column; min-height:400px; }
            .cb-sidebar { width:100%; max-height:200px; flex-direction:row; }
            .cb-sidebar-header { flex:1; }
            .cb-sysprompt { display:none; }
            .cb-options { border-bottom:none; border-right:1px solid var(--border); padding:8px; }
        }
    `);

    /* ===== 상태 ===== */
    const CHATBOT_SESSIONS_INDEX_KEY = 'toolbox_chatbot_sessions_index';
    const CHATBOT_SESSION_PREFIX = 'toolbox_chatbot_session_';
    let currentSessionId = null;
    const MAX_SESSIONS = 10;

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
            sessionStorage.setItem(CHATBOT_SESSION_PREFIX + currentSessionId, JSON.stringify({
                chatHistory: toSave,
                conversationSummary,
                savedAt: Date.now()
            }));
        } catch (e) {
            console.warn('Chatbot session save failed', e);
        }
    }

    function loadSession(id) {
        try {
            const raw = sessionStorage.getItem(CHATBOT_SESSION_PREFIX + (id || currentSessionId));
            if (!raw) return false;
            const data = JSON.parse(raw);
            if (data.chatHistory && Array.isArray(data.chatHistory)) {
                chatHistory = data.chatHistory;
                conversationSummary = data.conversationSummary || '';
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
        Mdd.setMood('think'); Mdd.say('대화 상대가 필요해요?');

        container.innerHTML = `
            <div class="cb-layout">
                <div class="cb-sidebar">
                    <div class="cb-sidebar-header">
                        <h3>⚙️ 설정</h3>
                        <div class="field-group">
                            <label class="field-label">🔑 API 키</label>
                            <div style="display:flex;gap:8px;align-items:center;justify-content:space-between;">
                            <div style="font-size:var(--font-size-xs);color:var(--text-tertiary);">
                                    프로필: <strong id="cbActiveProfileName" style="color:var(--text-secondary);">${typeof Gemini !== 'undefined' ? (Gemini.getActiveProfileName() || '기본') : '-'}</strong>
                                </div>
                                <button class="btn btn-ghost" type="button" onclick="Toolbox.switchPage('user'); Toolbox.switchTab('user-settings');">설정에서 변경</button>
                            </div>
                        </div>
                        <label class="cb-model-label">🤖 모델</label>
                        <select id="cbModelSelect" style="font-size:var(--font-size-xs);padding:6px 8px;"></select>
                    </div>

                    <div class="cb-options">
                        <div class="cb-option-row">
                            <label>🔍 웹 검색</label>
                            <label class="cb-toggle"><input type="checkbox" id="cbWebSearch"><span class="cb-toggle-slider"></span></label>
                        </div>
                        <div class="cb-option-row">
                            <label>📝 메모리</label>
                            <label class="cb-toggle"><input type="checkbox" id="cbMemory" checked><span class="cb-toggle-slider"></span></label>
                        </div>
                        <div class="cb-option-row cb-temperature-row">
                            <label>🌡️ Temperature <span id="cbTempValue">0.8</span></label>
                        </div>
                        <div class="cb-option-row" style="margin-top:2px;">
                            <input type="range" id="cbTemperature" min="0" max="2" step="0.1" value="0.8" style="width:100%;">
                        </div>
                    </div>

                    <div class="cb-sysprompt">
                        <label>📋 시스템 프롬프트</label>
                        <select id="cbSystemPreset" style="font-size:var(--font-size-xs);padding:6px 8px;margin-bottom:8px;width:100%;">
                            <option value="">직접 입력</option>
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
                </div>

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
                        <span class="cb-chat-header-title">💬 챗봇</span>
                        <div style="display:flex;gap:6px;">
                            <button class="btn btn-ghost" id="cbShortcutsBtn" title="키보드 단축키 (Ctrl+/)">⌨️</button>
                            <button class="btn btn-ghost" id="cbSearchToggle" title="대화 검색 (Ctrl+F)">🔍</button>
                            <button class="btn btn-ghost" onclick="window._cb.importChat()">📥 가져오기</button>
                            <button class="btn btn-ghost" onclick="window._cb.exportChat('txt')">📤 TXT</button>
                            <button class="btn btn-ghost" onclick="window._cb.exportChat('json')">📤 JSON</button>
                            <button class="btn btn-ghost" onclick="window._cb.clearChat()">🗑️ 초기화</button>
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

            // 시스템 프롬프트 프리셋
            const presetSel = document.getElementById('cbSystemPreset');
            const sysPromptTa = document.getElementById('cbSystemPrompt');
            const savedPreset = Toolbox.getPref('cb_preset');
            if (presetSel && sysPromptTa) {
                if (savedPreset) { presetSel.value = savedPreset; if (SYSTEM_PROMPT_PRESETS[savedPreset]) sysPromptTa.value = SYSTEM_PROMPT_PRESETS[savedPreset]; }
                presetSel.addEventListener('change', () => {
                    const v = presetSel.value;
                    Toolbox.setPref('cb_preset', v);
                    if (v && SYSTEM_PROMPT_PRESETS[v]) sysPromptTa.value = SYSTEM_PROMPT_PRESETS[v];
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

            // 세션 복원 또는 초기 인사
            const msgs = document.getElementById('cbMessages');
            if (msgs) {
                if (loadSession(currentSessionId) && chatHistory.length > 0) {
                    chatHistory.forEach(msg => {
                        const role = msg.role === 'user' ? 'user' : 'bot';
                        const text = msg.parts?.[0]?.text || '';
                        if (text) appendMsg(role, text, false);
                    });
                } else {
                    appendMsg('bot', '안녕하세요! 무엇을 도와드릴까요? 😊');
                }
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
                                    sysPromptTa.value = SYSTEM_PROMPT_PRESETS.writer;
                                    Toolbox.setPref('cb_preset', 'writer');
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

            let systemPrompt = document.getElementById('cbSystemPrompt')?.value || '';
            const useMemory = document.getElementById('cbMemory')?.checked;
            const useWebSearch = document.getElementById('cbWebSearch')?.checked;

            if (useMemory) {
                systemPrompt += `\n\n[SYSTEM: MEMORY PROTOCOL]
To maintain conversation continuity, you must update the conversation summary with every response.

Current Summary:
${conversationSummary || "No summary yet."}

Output Format:
You must start your response with a summary block wrapped in {{{ ... }}} braces, followed by your response.
The summary should be concise but capture key information about the conversation flow.

Example:
{{{User asked about X. I explained Y.}}}
Here is my actual response...`;
            }

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
                            streamEl.div.innerHTML = renderMarkdown(fullText) + '<span class="cb-cursor-blink">▌</span>';
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

                finalizeStreamMsg(streamEl, responseText);
                updateTokens(lastUsage);

                chatHistory.push({ role: 'model', parts: [{ text: responseText }] });
                saveSession();
                Toolbox.recordUsage('chat', lastUsage?.totalTokenCount || 0);
                Mdd.setMood('happy'); Mdd.say('대답 완료해요!');

            } catch (e) {
                if (streamEl.wrap.parentNode) streamEl.wrap.remove();
                if (e.message !== '요청이 취소되었습니다.') {
                    appendMsg('bot', `오류: ${e.message}`, true);
                    Toolbox.showToast(e.message || '오류', 'error', e);
                    Mdd.setMood('sad'); Mdd.say('에러예요...');
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

            let systemPrompt = document.getElementById('cbSystemPrompt')?.value || '';
            const useMemory = document.getElementById('cbMemory')?.checked;
            const useWebSearch = document.getElementById('cbWebSearch')?.checked;
            if (useMemory) {
                systemPrompt += `\n\n[SYSTEM: MEMORY PROTOCOL]\nTo maintain conversation continuity, you must update the conversation summary with every response.\n\nCurrent Summary:\n${conversationSummary || "No summary yet."}\n\nOutput Format:\nYou must start your response with a summary block wrapped in {{{ ... }}} braces, followed by your response.\nThe summary should be concise but capture key information about the conversation flow.\n\nExample:\n{{{User asked about X. I explained Y.}}}\nHere is my actual response...`;
            }

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
                            streamEl.div.innerHTML = renderMarkdown(fullText) + '<span class="cb-cursor-blink">▌</span>';
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
                finalizeStreamMsg(streamEl, responseText);
                updateTokens(lastUsage);
                chatHistory.push({ role: 'model', parts: [{ text: responseText }] });
                saveSession();
                Toolbox.recordUsage('chat', lastUsage?.totalTokenCount || 0);
                Mdd.setMood('happy'); Mdd.say('다시 대답했어요!');
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
            Mdd.setMood('idle'); Mdd.say('새로 시작이에요!');
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
