/** Chatbot UI: buildChat, appendMsg, renderSessionTabs, 첨부/검색/단축키 등 */
(function () {
    var Session = window.Chatbot.Session;
    var Markdown = window.Chatbot.Markdown;
    var state = window.Chatbot.state;
    var escapeHtml = function (s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    };

    function fileToBase64(file) {
        return new Promise(function (resolve, reject) {
            var reader = new FileReader();
            reader.onload = function () {
                var dataUrl = reader.result;
                var base64 = dataUrl.split(',')[1];
                resolve({ base64: base64, mimeType: file.type, dataUrl: dataUrl });
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    function addPendingImage(imgData) {
        if (state.pendingImages.length >= 5) {
            if (typeof Toolbox !== 'undefined') Toolbox.showToast('최대 5장까지 첨부 가능', 'error');
            return;
        }
        state.pendingImages.push(imgData);
        renderAttachThumbs();
    }

    function renderAttachThumbs() {
        var area = document.getElementById('cbAttachArea');
        if (!area) return;
        area.querySelectorAll('.cb-attach-wrap').forEach(function (el) { el.remove(); });
        state.pendingImages.forEach(function (img, i) {
            var wrap = document.createElement('span');
            wrap.className = 'cb-attach-wrap';
            var thumb = document.createElement('img');
            thumb.className = 'cb-attach-thumb';
            thumb.src = img.dataUrl;
            var rm = document.createElement('button');
            rm.className = 'cb-attach-remove';
            rm.textContent = '×';
            rm.onclick = function () { state.pendingImages.splice(i, 1); renderAttachThumbs(); };
            wrap.appendChild(thumb);
            wrap.appendChild(rm);
            area.insertBefore(wrap, area.querySelector('.cb-attach-btn'));
        });
    }

    function appendMsg(role, text, isError) {
        if (isError === undefined) isError = false;
        var msgs = document.getElementById('cbMessages');
        if (!msgs) return;
        var wrap = document.createElement('div');
        wrap.className = 'cb-msg-wrap';
        var div = document.createElement('div');
        div.className = 'cb-msg cb-msg-' + role + (isError ? ' cb-msg-error' : '');
        if (role === 'bot' && !isError) div.innerHTML = Markdown.render(text);
        else div.textContent = text;
        var copyBtn = document.createElement('button');
        copyBtn.className = 'btn btn-ghost cb-msg-copy';
        copyBtn.type = 'button';
        copyBtn.textContent = '복사';
        copyBtn.onclick = function () { navigator.clipboard.writeText(text).then(function () { Toolbox.showToast('복사됨'); }).catch(function () {}); };
        wrap.appendChild(div);
        wrap.appendChild(copyBtn);
        if (role === 'bot' && !isError && state.chatHistory.length > 0) {
            var regen = document.createElement('button');
            regen.className = 'btn btn-ghost';
            regen.textContent = '🔄 재생성';
            regen.onclick = function () { window._cb.regenerate(); };
            wrap.appendChild(regen);
        }
        msgs.appendChild(wrap);
        if (role === 'bot' && !isError && typeof Prism !== 'undefined') {
            div.querySelectorAll('pre code[class*="language-"]').forEach(function (el) { Prism.highlightElement(el); });
        }
        msgs.scrollTop = msgs.scrollHeight;
    }

    function appendStreamMsg() {
        var msgs = document.getElementById('cbMessages');
        if (!msgs) return null;
        var wrap = document.createElement('div');
        wrap.className = 'cb-msg-wrap';
        var div = document.createElement('div');
        div.className = 'cb-msg cb-msg-bot';
        div.innerHTML = '<span class="cb-cursor-blink">▌</span>';
        var copyBtn = document.createElement('button');
        copyBtn.className = 'btn btn-ghost cb-msg-copy';
        copyBtn.type = 'button';
        copyBtn.textContent = '복사';
        wrap.appendChild(div);
        wrap.appendChild(copyBtn);
        msgs.appendChild(wrap);
        msgs.scrollTop = msgs.scrollHeight;
        return { div: div, copyBtn: copyBtn, wrap: wrap };
    }

    function finalizeStreamMsg(el, fullText) {
        if (!el) return;
        el.div.innerHTML = Markdown.render(fullText);
        if (typeof Prism !== 'undefined') {
            el.div.querySelectorAll('pre code[class*="language-"]').forEach(function (c) { Prism.highlightElement(c); });
        }
        el.copyBtn.onclick = function () { navigator.clipboard.writeText(fullText).then(function () { Toolbox.showToast('복사됨'); }).catch(function () {}); };
        var regen = document.createElement('button');
        regen.className = 'btn btn-ghost';
        regen.textContent = '🔄 재생성';
        regen.onclick = function () { window._cb.regenerate(); };
        el.wrap.appendChild(regen);
    }

    function updateTokens(usage) {
        if (!usage) return;
        var total = usage.totalTokenCount || 0;
        var display = document.getElementById('cbTokenDisplay');
        if (display) {
            display.textContent = 'Tokens: ' + total.toLocaleString();
            display.style.color = 'var(--text-tertiary)';
        }
    }

    function renderSessionTabs() {
        var container = document.getElementById('cbSessionTabs');
        if (!container) return;
        var index = Session.getSessionsIndex();
        var switchSession = window.Chatbot.Actions && window.Chatbot.Actions.switchSession;
        container.innerHTML = '';
        index.forEach(function (s) {
            var tab = document.createElement('button');
            tab.className = 'cb-session-tab' + (s.id === state.currentSessionId ? ' active' : '');
            tab.innerHTML = '<span class="cb-session-tab-name">' + escapeHtml(s.name) + '</span>';
            if (index.length > 1) {
                var del = document.createElement('span');
                del.className = 'cb-session-tab-del';
                del.textContent = '×';
                del.onclick = function (e) {
                    e.stopPropagation();
                    Session.deleteSession(s.id);
                    if (s.id === state.currentSessionId) {
                        var remaining = Session.getSessionsIndex();
                        if (remaining.length > 0) switchSession(remaining[remaining.length - 1].id);
                        else { var nid = Session.createNewSession(); switchSession(nid); }
                    } else {
                        renderSessionTabs();
                    }
                };
                tab.appendChild(del);
            }
            tab.onclick = function () { if (s.id !== state.currentSessionId) switchSession(s.id); };
            var nameSpan = tab.querySelector('.cb-session-tab-name');
            if (nameSpan) {
                nameSpan.ondblclick = function (e) {
                    e.stopPropagation();
                    var input = document.createElement('input');
                    input.className = 'cb-session-tab-edit';
                    input.value = s.name;
                    input.maxLength = 20;
                    var commit = function () {
                        var newName = (input.value.trim() || s.name);
                        s.name = newName;
                        var idx = Session.getSessionsIndex();
                        for (var i = 0; i < idx.length; i++) { if (idx[i].id === s.id) { idx[i].name = newName; break; } }
                        Session.saveSessionsIndex(idx);
                        renderSessionTabs();
                    };
                    input.onblur = commit;
                    input.onkeydown = function (ev) { if (ev.key === 'Enter') commit(); if (ev.key === 'Escape') renderSessionTabs(); };
                    nameSpan.replaceWith(input);
                    input.focus();
                    input.select();
                };
            }
            container.appendChild(tab);
        });
        var addBtn = document.createElement('button');
        addBtn.className = 'cb-session-tab cb-session-add';
        addBtn.textContent = '+';
        addBtn.title = '새 대화';
        addBtn.onclick = function () {
            var idx = Session.getSessionsIndex();
            if (idx.length >= Session.MAX_SESSIONS) {
                Toolbox.showToast('최대 세션 수에 도달', 'error');
                return;
            }
            var nid = Session.createNewSession('대화 ' + (idx.length + 1));
            switchSession(nid);
        };
        container.appendChild(addBtn);
    }

    function buildPresetOptions(presets) {
        var html = '';
        for (var i = 0; i < (presets || []).length; i++) {
            html += '<option value="' + escapeHtml(presets[i].id) + '">' + escapeHtml(presets[i].label) + '</option>';
        }
        return html;
    }

    function buildChat(container) {
        if (typeof Mdd !== 'undefined') { Mdd.linePreset('tool_run', { msg: '대화 상대가 필요해요?' }); }

        var presets = window.Chatbot.presets || [];
        var defaultText = '당신은 친절하고 유용한 AI 어시스턴트입니다. 한국어로 답변해주세요.';
        for (var i = 0; i < presets.length; i++) {
            if (presets[i].id === 'default') { defaultText = presets[i].text; break; }
        }

        var profileName = (typeof Gemini !== 'undefined' && Gemini.getActiveProfileName) ? (Gemini.getActiveProfileName() || '기본') : '-';

        container.innerHTML = '<div class="cb-layout">' +
            '<div class="cb-sidebar">' +
                '<div class="cb-sidebar-header">' +
                    '<h3>⚙️ 설정</h3>' +
                    '<div class="field-group"><label class="field-label">🔑 API 키</label>' +
                    '<div style="display:flex;gap:8px;align-items:center;justify-content:space-between;">' +
                    '<div style="font-size:var(--font-size-xs);color:var(--text-tertiary);">프로필: <strong id="cbActiveProfileName" style="color:var(--text-secondary);">' + profileName + '</strong></div>' +
                    '<button class="btn btn-ghost" type="button" onclick="Toolbox.switchPage(\'user\'); Toolbox.switchTab(\'user-settings\');">설정에서 변경</button></div></div>' +
                    '<label class="cb-model-label">🤖 모델</label>' +
                    '<select id="cbModelSelect" style="font-size:var(--font-size-xs);padding:6px 8px;"></select>' +
                '</div>' +
                '<div class="cb-options">' +
                    '<div class="cb-option-row"><label>🔍 웹 검색</label><label class="cb-toggle"><input type="checkbox" id="cbWebSearch"><span class="cb-toggle-slider"></span></label></div>' +
                    '<div class="cb-option-row"><label>📝 메모리</label><label class="cb-toggle"><input type="checkbox" id="cbMemory" checked><span class="cb-toggle-slider"></span></label></div>' +
                    '<div class="cb-option-row cb-temperature-row"><label>🌡️ Temperature <span id="cbTempValue">0.8</span><button type="button" class="cb-info-btn" id="cbTempInfoBtn" title="Temperature 설명" aria-label="설명">i</button></label></div>' +
                    '<div class="cb-info-card" id="cbTempInfoCard"><strong>Temperature</strong>는 AI가 답변할 때 <strong>다양성(무작위성)</strong>을 얼마나 허용할지 정합니다.<table><tr><td>낮음 (0~0.3)</td><td>일관적·정확한 답변 (번역, 요약, 코드)</td></tr><tr><td>중간 (0.7~0.9)</td><td>자연스러운 대화 (기본 권장)</td></tr><tr><td>높음 (1.1~2)</td><td>창의적인 답변 (시, 소설, 아이디어)</td></tr></table></div>' +
                    '<div class="cb-option-row" style="margin-top:2px;"><input type="range" id="cbTemperature" min="0" max="2" step="0.1" value="0.8" style="width:100%;"></div>' +
                '</div>' +
                '<div class="cb-sysprompt">' +
                    '<label>📋 시스템 프롬프트</label>' +
                    '<select id="cbSystemPreset" style="font-size:var(--font-size-xs);padding:6px 8px;margin-bottom:8px;width:100%;"><option value="">직접 입력</option>' + buildPresetOptions(presets) + '</select>' +
                    '<textarea id="cbSystemPrompt" placeholder="AI의 역할, 성격, 답변 스타일 등을 지정하세요...">' + escapeHtml(defaultText) + '</textarea>' +
                '</div>' +
            '</div>' +
            '<div class="cb-chat" style="position:relative;">' +
                '<div class="cb-shortcuts-overlay" id="cbShortcutsOverlay">' +
                    '<div class="cb-shortcuts-panel">' +
                        '<h3>⌨️ 키보드 단축키</h3>' +
                        '<div class="cb-shortcut-row"><span>메시지 전송</span><span class="cb-shortcut-key">Enter</span></div>' +
                        '<div class="cb-shortcut-row"><span>줄바꿈</span><span class="cb-shortcut-key">Shift + Enter</span></div>' +
                        '<div class="cb-shortcut-row"><span>대화 검색</span><span class="cb-shortcut-key">Ctrl + F</span></div>' +
                        '<div class="cb-shortcut-row"><span>새 세션</span><span class="cb-shortcut-key">Ctrl + N</span></div>' +
                        '<div class="cb-shortcut-row"><span>단축키 안내</span><span class="cb-shortcut-key">Ctrl + /</span></div>' +
                        '<div style="margin-top:12px;text-align:center;"><button class="btn btn-ghost" onclick="document.getElementById(\'cbShortcutsOverlay\').classList.remove(\'open\')">닫기</button></div>' +
                    '</div>' +
                '</div>' +
                '<div class="cb-chat-header">' +
                    '<span class="cb-chat-header-title">💬 챗봇</span>' +
                    '<div style="display:flex;gap:6px;">' +
                        '<button class="btn btn-ghost" id="cbShortcutsBtn" title="키보드 단축키 (Ctrl+/)">⌨️</button>' +
                        '<button class="btn btn-ghost" id="cbSearchToggle" title="대화 검색 (Ctrl+F)">🔍</button>' +
                        '<button class="btn btn-ghost" onclick="window._cb.importChat()">📥 가져오기</button>' +
                        '<button class="btn btn-ghost" onclick="window._cb.exportChat(\'txt\')">📤 TXT</button>' +
                        '<button class="btn btn-ghost" onclick="window._cb.exportChat(\'json\')">📤 JSON</button>' +
                        '<button class="btn btn-ghost" onclick="window._cb.clearChat()">🗑️ 초기화</button>' +
                    '</div>' +
                '</div>' +
                '<div class="cb-session-bar" id="cbSessionTabs"></div>' +
                '<div class="cb-search-bar" id="cbSearchBar">' +
                    '<input type="text" id="cbSearchInput" placeholder="대화 내용 검색...">' +
                    '<span class="cb-search-nav" id="cbSearchNav"></span>' +
                    '<button class="btn btn-ghost" id="cbSearchPrev" title="이전">▲</button>' +
                    '<button class="btn btn-ghost" id="cbSearchNext" title="다음">▼</button>' +
                    '<button class="btn btn-ghost" id="cbSearchClose" title="닫기">✕</button>' +
                '</div>' +
                '<div class="cb-messages" id="cbMessages" role="log" aria-live="polite" aria-label="대화 내용"></div>' +
                '<div class="cb-input-area" id="cbInputArea">' +
                    '<div class="cb-attach-area" id="cbAttachArea">' +
                        '<button class="cb-attach-btn" id="cbAttachBtn" title="이미지 첨부">📎</button>' +
                        '<input type="file" id="cbFileInput" accept="image/*" multiple style="display:none">' +
                    '</div>' +
                    '<div class="cb-input-row">' +
                        '<textarea id="cbInput" placeholder="메시지를 입력하세요... (이미지를 드래그하거나 붙여넣기 가능)"></textarea>' +
                        '<button class="cb-mic-btn" id="cbMicBtn" title="음성 입력" aria-label="음성 입력">🎤</button>' +
                        '<button class="cb-send-btn" id="cbSendBtn" onclick="window._cb.send()" aria-label="메시지 전송">➤</button>' +
                        '<button class="cb-stop-btn" id="cbStopBtn" style="display:none" onclick="window._cb.stopStream()" aria-label="응답 중지">■ 중지</button>' +
                    '</div>' +
                    '<div class="cb-token-bar"><span id="cbTokenDisplay">Tokens: 0</span><span>AI의 응답은 정확하지 않을 수 있습니다.</span></div>' +
                '</div>' +
            '</div>' +
        '</div>';

        requestAnimationFrame(function () {
            var sel = document.getElementById('cbModelSelect');
            if (sel && typeof Gemini !== 'undefined' && Gemini.MODELS && Gemini.MODELS.gemini) {
                Gemini.MODELS.gemini.forEach(function (m) {
                    var o = document.createElement('option');
                    o.value = m.id; o.textContent = m.name;
                    if (m.isDefault) o.selected = true;
                    sel.appendChild(o);
                });
            }
            var savedModel = Toolbox.getPref('cb_model');
            if (savedModel && sel) sel.value = savedModel;
            if (sel) sel.addEventListener('change', function () { Toolbox.setPref('cb_model', sel.value); });

            var presetSel = document.getElementById('cbSystemPreset');
            var sysPromptTa = document.getElementById('cbSystemPrompt');
            var savedPreset = Toolbox.getPref('cb_preset');
            if (presetSel && sysPromptTa) {
                if (savedPreset) {
                    presetSel.value = savedPreset;
                    for (var j = 0; j < presets.length; j++) {
                        if (presets[j].id === savedPreset) { sysPromptTa.value = presets[j].text; break; }
                    }
                }
                presetSel.addEventListener('change', function () {
                    var v = presetSel.value;
                    Toolbox.setPref('cb_preset', v);
                    for (var k = 0; k < presets.length; k++) {
                        if (presets[k].id === v) { sysPromptTa.value = presets[k].text; break; }
                    }
                });
            }

            var tempSlider = document.getElementById('cbTemperature');
            var tempValueEl = document.getElementById('cbTempValue');
            var tempInfoBtn = document.getElementById('cbTempInfoBtn');
            var tempInfoCard = document.getElementById('cbTempInfoCard');
            var savedTemp = Toolbox.getPref('cb_temperature');
            if (tempSlider && tempValueEl) {
                if (savedTemp !== undefined) { tempSlider.value = savedTemp; tempValueEl.textContent = savedTemp; }
                tempSlider.addEventListener('input', function () { tempValueEl.textContent = tempSlider.value; Toolbox.setPref('cb_temperature', tempSlider.value); });
            }
            if (tempInfoBtn && tempInfoCard) {
                tempInfoBtn.addEventListener('click', function () { tempInfoCard.classList.toggle('open'); });
            }

            var attachBtn = document.getElementById('cbAttachBtn');
            var fileInput = document.getElementById('cbFileInput');
            var inputArea = document.getElementById('cbInputArea');
            if (attachBtn && fileInput) {
                attachBtn.onclick = function () { fileInput.click(); };
                fileInput.onchange = function () {
                    var files = fileInput.files;
                    for (var i = 0; i < files.length; i++) {
                        if (files[i].type.indexOf('image/') === 0) fileToBase64(files[i]).then(addPendingImage);
                    }
                    fileInput.value = '';
                };
            }
            if (inputArea) {
                inputArea.ondragover = function (e) { e.preventDefault(); inputArea.classList.add('drag-over'); };
                inputArea.ondragleave = function () { inputArea.classList.remove('drag-over'); };
                inputArea.ondrop = function (e) {
                    e.preventDefault(); inputArea.classList.remove('drag-over');
                    var files = e.dataTransfer.files;
                    for (var i = 0; i < files.length; i++) {
                        if (files[i].type.indexOf('image/') === 0) fileToBase64(files[i]).then(addPendingImage);
                    }
                };
            }

            var chatInput = document.getElementById('cbInput');
            if (chatInput) {
                chatInput.addEventListener('keydown', function (e) {
                    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); window._cb.send(); }
                });
                chatInput.addEventListener('paste', function (e) {
                    var items = e.clipboardData && e.clipboardData.items;
                    if (!items) return;
                    for (var i = 0; i < items.length; i++) {
                        if (items[i].type.indexOf('image/') === 0) {
                            e.preventDefault();
                            var file = items[i].getAsFile();
                            if (file) fileToBase64(file).then(addPendingImage);
                        }
                    }
                });
            }

            var micBtn = document.getElementById('cbMicBtn');
            if (micBtn) {
                var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (SpeechRecognition) {
                    var recognition = new SpeechRecognition();
                    recognition.lang = 'ko-KR';
                    recognition.interimResults = true;
                    recognition.continuous = false;
                    var isRecording = false;
                    var finalTranscript = '';
                    micBtn.addEventListener('click', function () {
                        if (isRecording) { recognition.stop(); return; }
                        finalTranscript = '';
                        recognition.start();
                    });
                    recognition.onstart = function () { isRecording = true; micBtn.classList.add('recording'); micBtn.title = '녹음 중... 클릭하여 중지'; };
                    recognition.onend = function () {
                        isRecording = false;
                        micBtn.classList.remove('recording');
                        micBtn.title = '음성 입력';
                        if (finalTranscript && chatInput) {
                            chatInput.value += (chatInput.value ? ' ' : '') + finalTranscript;
                            chatInput.focus();
                        }
                    };
                    recognition.onresult = function (e) {
                        for (var i = e.resultIndex; i < e.results.length; i++) {
                            if (e.results[i].isFinal) finalTranscript += e.results[i][0].transcript;
                        }
                    };
                    recognition.onerror = function (e) {
                        if (e.error !== 'aborted') Toolbox.showToast('음성 인식 오류: ' + e.error, 'error');
                    };
                } else {
                    micBtn.style.display = 'none';
                }
            }

            var index = Session.getSessionsIndex();
            if (index.length === 0) {
                state.currentSessionId = Session.createNewSession('대화 1');
            } else {
                state.currentSessionId = index[index.length - 1].id;
            }
            renderSessionTabs();

            var msgs = document.getElementById('cbMessages');
            if (msgs) {
                var data = Session.loadSession(state.currentSessionId);
                if (data && data.chatHistory && data.chatHistory.length > 0) {
                    state.chatHistory = data.chatHistory;
                    state.conversationSummary = data.conversationSummary || '';
                    state.chatHistory.forEach(function (msg) {
                        var role = msg.role === 'user' ? 'user' : 'bot';
                        var text = (msg.parts && msg.parts[0]) ? msg.parts[0].text : '';
                        if (text) appendMsg(role, text, false);
                    });
                } else {
                    appendMsg('bot', '안녕하세요! 무엇을 도와드릴까요? 😊');
                }
                msgs.scrollTop = msgs.scrollHeight;
            }

            var searchToggle = document.getElementById('cbSearchToggle');
            var searchBar = document.getElementById('cbSearchBar');
            var searchInput = document.getElementById('cbSearchInput');
            var searchNav = document.getElementById('cbSearchNav');
            var searchPrev = document.getElementById('cbSearchPrev');
            var searchNext = document.getElementById('cbSearchNext');
            var searchClose = document.getElementById('cbSearchClose');
            if (searchToggle && searchBar && searchInput && msgs) {
                var searchResults = [];
                var searchIdx = -1;
                function toggleSearch() {
                    var open = searchBar.classList.toggle('open');
                    if (open) searchInput.focus();
                    else { clearHighlights(); searchInput.value = ''; searchNav.textContent = ''; }
                }
                function clearHighlights() {
                    msgs.querySelectorAll('.cb-search-highlight').forEach(function (el) {
                        var parent = el.parentNode;
                        parent.replaceChild(document.createTextNode(el.textContent), el);
                        parent.normalize();
                    });
                    searchResults = [];
                    searchIdx = -1;
                }
                function doSearch() {
                    clearHighlights();
                    var q = searchInput.value.trim();
                    if (!q) { searchNav.textContent = ''; return; }
                    var regex = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
                    var walker = document.createTreeWalker(msgs, NodeFilter.SHOW_TEXT, null, false);
                    var matches = [];
                    while (walker.nextNode()) {
                        var node = walker.currentNode;
                        if (node.parentElement.closest('pre, code, .cb-code-header')) continue;
                        if (regex.test(node.textContent)) matches.push(node);
                        regex.lastIndex = 0;
                    }
                    matches.forEach(function (node) {
                        var text = node.textContent;
                        var parts = text.split(regex);
                        if (parts.length <= 1) return;
                        var frag = document.createDocumentFragment();
                        var m;
                        regex.lastIndex = 0;
                        var lastIdx = 0;
                        while ((m = regex.exec(text)) !== null) {
                            if (m.index > lastIdx) frag.appendChild(document.createTextNode(text.slice(lastIdx, m.index)));
                            var mark = document.createElement('mark');
                            mark.className = 'cb-search-highlight';
                            mark.textContent = m[0];
                            frag.appendChild(mark);
                            lastIdx = regex.lastIndex;
                        }
                        if (lastIdx < text.length) frag.appendChild(document.createTextNode(text.slice(lastIdx)));
                        node.parentNode.replaceChild(frag, node);
                    });
                    searchResults = Array.prototype.slice.call(msgs.querySelectorAll('.cb-search-highlight'));
                    searchIdx = searchResults.length > 0 ? 0 : -1;
                    updateSearchNav();
                }
                function updateSearchNav() {
                    if (searchResults.length === 0) { searchNav.textContent = '결과 없음'; return; }
                    searchNav.textContent = (searchIdx + 1) + ' / ' + searchResults.length;
                    searchResults.forEach(function (el, i) { el.classList.toggle('current', i === searchIdx); });
                    if (searchResults[searchIdx]) searchResults[searchIdx].scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
                function navigate(dir) {
                    if (searchResults.length === 0) return;
                    searchIdx = (searchIdx + dir + searchResults.length) % searchResults.length;
                    updateSearchNav();
                }
                searchToggle.addEventListener('click', toggleSearch);
                searchClose.addEventListener('click', toggleSearch);
                searchInput.addEventListener('input', doSearch);
                searchPrev.addEventListener('click', function () { navigate(-1); });
                searchNext.addEventListener('click', function () { navigate(1); });
                searchInput.addEventListener('keydown', function (e) {
                    if (e.key === 'Enter') { e.preventDefault(); navigate(e.shiftKey ? -1 : 1); }
                    if (e.key === 'Escape') toggleSearch();
                });
            }

            var shortcutsBtn = document.getElementById('cbShortcutsBtn');
            var shortcutsOverlay = document.getElementById('cbShortcutsOverlay');
            if (shortcutsBtn && shortcutsOverlay) {
                shortcutsBtn.addEventListener('click', function () { shortcutsOverlay.classList.toggle('open'); });
                shortcutsOverlay.addEventListener('click', function (e) { if (e.target === shortcutsOverlay) shortcutsOverlay.classList.remove('open'); });
            }

            document.addEventListener('keydown', function (e) {
                var chatEl = document.querySelector('.cb-chat');
                if (!chatEl || chatEl.offsetParent === null) return;
                if (e.ctrlKey && e.key === 'f') {
                    e.preventDefault();
                    var sb = document.getElementById('cbSearchBar');
                    var si = document.getElementById('cbSearchInput');
                    if (sb && si) { sb.classList.add('open'); si.focus(); }
                }
                if (e.ctrlKey && e.key === '/') { e.preventDefault(); shortcutsOverlay.classList.toggle('open'); }
                if (e.ctrlKey && e.key === 'n') {
                    e.preventDefault();
                    if (Session.getSessionsIndex().length >= 10) { Toolbox.showToast('최대 10개 세션까지 생성 가능합니다.', 'error'); return; }
                    var nid = Session.createNewSession('대화 ' + (Session.getSessionsIndex().length + 1));
                    window.Chatbot.Actions.switchSession(nid);
                }
                if (e.key === 'Escape') shortcutsOverlay.classList.remove('open');
            });

            var chatbotPage = document.getElementById('page-chatbot');
            if (chatbotPage) {
                var checkStoryKeywords = function () {
                    try {
                        var raw = sessionStorage.getItem('toolbox_chatbot_story_keywords');
                        if (raw) {
                            sessionStorage.removeItem('toolbox_chatbot_story_keywords');
                            var keywords = JSON.parse(raw);
                            if (Array.isArray(keywords) && keywords.length > 0) {
                                var input = document.getElementById('cbInput');
                                var presetSel = document.getElementById('cbSystemPreset');
                                var sysPromptTa = document.getElementById('cbSystemPrompt');
                                var writer = null;
                                for (var w = 0; w < presets.length; w++) { if (presets[w].id === 'writer') { writer = presets[w]; break; } }
                                if (presetSel && sysPromptTa && writer) {
                                    presetSel.value = 'writer';
                                    sysPromptTa.value = writer.text;
                                    Toolbox.setPref('cb_preset', 'writer');
                                }
                                if (input) {
                                    input.value = '다음 키워드를 포함해서 짧은 이야기를 써줘: ' + keywords.join(', ');
                                    input.focus();
                                    Toolbox.showToast('키워드가 입력되었습니다. 전송 버튼을 누르세요.');
                                }
                            }
                        }
                    } catch (_) {}
                };
                var obs = new MutationObserver(checkStoryKeywords);
                obs.observe(chatbotPage, { attributes: true, attributeFilter: ['class'] });
                if (chatbotPage.classList.contains('active')) checkStoryKeywords();
            }
        });
    }

    window.Chatbot.UI = {
        buildChat: buildChat,
        appendMsg: appendMsg,
        appendStreamMsg: appendStreamMsg,
        finalizeStreamMsg: finalizeStreamMsg,
        updateTokens: updateTokens,
        renderAttachThumbs: renderAttachThumbs,
        renderSessionTabs: renderSessionTabs,
        addPendingImage: addPendingImage,
        fileToBase64: fileToBase64
    };
})();
