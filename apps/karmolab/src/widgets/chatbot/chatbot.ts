// @ts-nocheck
(function () {
    /* ===== 상태 ===== */
    const CHATBOT_SESSIONS_INDEX_KEY = 'toolbox_chatbot_sessions_index';
    const CHATBOT_SESSION_PREFIX = 'toolbox_chatbot_session_';
    let currentSessionId = null;
    let lastLoadedSessionCharacterId = '';
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
        window.ChatbotCharacters.syncAfterSessionLoad(lastLoadedSessionCharacterId);
    }

    function renderSessionTabs() {
        const container = document.getElementById('cbSessionTabs');
        if (!container) return;
        const index = getSessionsIndex();
        container.innerHTML = '';
        index.forEach(s => {
            const tab = document.createElement('button');
            tab.className = 'cb-session-tab' + (s.id === currentSessionId ? ' active' : '');
            tab.innerHTML = `<span class="cb-session-tab-name">${Toolbox.escapeHtml(s.name)}</span>`;
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
                    if (window.ChatbotPrompt.SYSTEM_PROMPT_PRESETS[v]) sysPromptTa.value = window.ChatbotPrompt.SYSTEM_PROMPT_PRESETS[v];
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
            window.ChatbotCharacters.initCharacterUi({
                saveSession,
                getChatHistoryLength: () => chatHistory.length,
                appendBotFirstMes: (fm) => {
                    appendMsg('bot', fm, false);
                    chatHistory.push({ role: 'model', parts: [{ text: fm }] });
                    saveSession();
                },
                getLastLoadedSessionCharacterId: () => lastLoadedSessionCharacterId
            });
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
                        const raw = sessionStorage.getItem('toolbox_chatbot_story_keywords');
                        if (raw) {
                            sessionStorage.removeItem('toolbox_chatbot_story_keywords');
                            const keywords = JSON.parse(raw);
                            if (Array.isArray(keywords) && keywords.length > 0) {
                                const input = document.getElementById('cbInput');
                                const presetSel = document.getElementById('cbSystemPreset');
                                const sysPromptTa = document.getElementById('cbSystemPrompt');
                                if (presetSel && sysPromptTa && window.ChatbotPrompt.SYSTEM_PROMPT_PRESETS.writer) {
                                    presetSel.value = 'writer';
                                    Toolbox.setPref('cb_preset', 'writer');
                                    applySystemPresetUi();
                                }
                                if (input) {
                                    input.value = '다음 키워드를 포함해서 짧은 이야기를 써줘: ' + keywords.join(', ');
                                    input.focus();
                                    Toolbox.showToast('키워드가 입력되었습니다. 전송 버튼을 누르세요.');
                                }
                            }
                        }
                    } catch (err) {}
                };
                const obs = new MutationObserver(checkStoryKeywords);
                obs.observe(chatbotPage, { attributes: true, attributeFilter: ['class'] });
                if (chatbotPage.classList.contains('active')) checkStoryKeywords();
            }
        });
    }

    const { displayTextForStream, extractKarmoImage, appendCharacterImageAfterMessage } = window.ChatbotKarmoImage;

    /** 메모리 요약 블록 제거 후 KARMO_IMAGE 파싱 (스트리밍 send / regenerate 공통) */
    function parseStreamResponseText(fullText, useMemory) {
        let t = fullText;
        let newSummary;
        if (useMemory) {
            const m = t.match(/\{\{\{(.*?)\}\}\}/s);
            if (m) {
                newSummary = m[1].trim();
                t = t.replace(/\{\{\{.*?\}\}\}/s, '').trim();
            }
        }
        const imgParsed = extractKarmoImage(t);
        return { responseText: imgParsed.cleanText, imgParsed, newSummary };
    }

    const renderMarkdown = window.ChatbotMarkdown.renderMarkdown;

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
            const systemPrompt = window.ChatbotPrompt.assembleSystemPrompt({ useMemory, conversationSummary });

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

                const { responseText, imgParsed, newSummary } = parseStreamResponseText(fullText, useMemory);
                if (newSummary !== undefined) conversationSummary = newSummary;

                const autoImgOn = document.getElementById('cbCharAutoImage')?.checked;
                const charSel = document.getElementById('cbCharacterSelect');
                const charForImg = charSel?.value
                    ? window.ChatbotCharacters.getCharacterById(charSel.value)
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
            const systemPrompt = window.ChatbotPrompt.assembleSystemPrompt({ useMemory, conversationSummary });

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
                const { responseText, imgParsed: imgParsedR, newSummary: newSummaryR } = parseStreamResponseText(fullText, useMemory);
                if (newSummaryR !== undefined) conversationSummary = newSummaryR;

                const autoImgR = document.getElementById('cbCharAutoImage')?.checked;
                const charSelR = document.getElementById('cbCharacterSelect');
                const charForImgR = charSelR?.value
                    ? window.ChatbotCharacters.getCharacterById(charSelR.value)
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
