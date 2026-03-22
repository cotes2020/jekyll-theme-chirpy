/** Chatbot 액션: send, regenerate, clearChat, import/export, switchSession */
(function () {
    var Session = window.Chatbot.Session;
    var Markdown = window.Chatbot.Markdown;
    var UI = window.Chatbot.UI;
    var state = window.Chatbot.state;

    var currentStreamAbort = null;

    function saveSession() {
        if (!state.currentSessionId) return;
        var toSave = state.chatHistory.map(function (msg) {
            return {
                role: msg.role,
                parts: msg.parts.map(function (p) {
                    if (p.inlineData) return { text: '[image]' };
                    return p;
                })
            };
        });
        Session.saveSession(state.currentSessionId, {
            chatHistory: toSave,
            conversationSummary: state.conversationSummary
        });
    }

    function switchSession(id) {
        saveSession();
        state.currentSessionId = id;
        state.chatHistory = [];
        state.conversationSummary = '';
        state.pendingImages = [];
        UI.renderAttachThumbs();
        var msgs = document.getElementById('cbMessages');
        if (msgs) msgs.innerHTML = '';
        var data = Session.loadSession(id);
        if (data && data.chatHistory && data.chatHistory.length > 0) {
            state.chatHistory = data.chatHistory;
            state.conversationSummary = data.conversationSummary || '';
            state.chatHistory.forEach(function (msg) {
                var role = msg.role === 'user' ? 'user' : 'bot';
                var text = (msg.parts && msg.parts[0]) ? msg.parts[0].text : '';
                if (text) UI.appendMsg(role, text, false);
            });
        } else {
            UI.appendMsg('bot', '안녕하세요! 무엇을 도와드릴까요? 😊');
        }
        if (msgs) msgs.scrollTop = msgs.scrollHeight;
        UI.renderSessionTabs();
    }

    async function doSendOrRegenerate() {
        var useMemory = document.getElementById('cbMemory') && document.getElementById('cbMemory').checked;
        var useWebSearch = document.getElementById('cbWebSearch') && document.getElementById('cbWebSearch').checked;
        var systemPrompt = (document.getElementById('cbSystemPrompt') && document.getElementById('cbSystemPrompt').value) || '';
        if (useMemory) {
            systemPrompt += '\n\n[SYSTEM: MEMORY PROTOCOL]\nTo maintain conversation continuity, you must update the conversation summary with every response.\n\nCurrent Summary:\n' + (state.conversationSummary || 'No summary yet.') + '\n\nOutput Format:\nYou must start your response with a summary block wrapped in {{{ ... }}} braces, followed by your response.\n\nExample:\n{{{User asked about X. I explained Y.}}}\nHere is my actual response...';
        }
        var modelSel = document.getElementById('cbModelSelect');
        var modelId = (modelSel && modelSel.value) || (typeof Gemini !== 'undefined' && Gemini.getDefaultModel && Gemini.getDefaultModel('gemini'));
        var tempInput = document.getElementById('cbTemperature');
        var temperature = tempInput ? parseFloat(tempInput.value) : 0.8;

        var streamEl = UI.appendStreamMsg();
        currentStreamAbort = new AbortController();
        var sendBtn = document.getElementById('cbSendBtn');
        var stopBtn = document.getElementById('cbStopBtn');
        if (sendBtn) sendBtn.style.display = 'none';
        if (stopBtn) stopBtn.style.display = '';

        try {
            var stream = await Gemini.callChatStream(state.chatHistory, systemPrompt, modelId, {
                webSearch: useWebSearch,
                temperature: temperature,
                signal: currentStreamAbort.signal
            });
            var fullText = '';
            var lastUsage = null;
            var renderPending = false;
            for await (var chunk of stream.chunks()) {
                fullText += chunk.text;
                if (chunk.usage) lastUsage = chunk.usage;
                if (!renderPending) {
                    renderPending = true;
                    requestAnimationFrame(function () {
                        renderPending = false;
                        streamEl.div.innerHTML = Markdown.render(fullText) + '<span class="cb-cursor-blink">▌</span>';
                        var msgs = document.getElementById('cbMessages');
                        if (msgs) msgs.scrollTop = msgs.scrollHeight;
                    });
                }
            }
            var responseText = fullText;
            if (useMemory) {
                var m = responseText.match(/\{\{\{(.*?)\}\}\}/s);
                if (m) {
                    state.conversationSummary = m[1].trim();
                    responseText = responseText.replace(/\{\{\{.*?\}\}\}/s, '').trim();
                }
            }
            UI.finalizeStreamMsg(streamEl, responseText);
            UI.updateTokens(lastUsage);
            state.chatHistory.push({ role: 'model', parts: [{ text: responseText }] });
            saveSession();
            if (typeof Toolbox !== 'undefined') Toolbox.recordUsage('chat', (lastUsage && lastUsage.totalTokenCount) || 0);
            if (typeof Mdd !== 'undefined') { Mdd.setMood('happy'); Mdd.say('대답 완료해요!'); }
        } catch (e) {
            if (streamEl && streamEl.wrap && streamEl.wrap.parentNode) streamEl.wrap.remove();
            if (e.message !== '요청이 취소되었습니다.') {
                UI.appendMsg('bot', '오류: ' + e.message, true);
                if (typeof Toolbox !== 'undefined') Toolbox.showToast(e.message || '오류', 'error', e);
                if (typeof Mdd !== 'undefined') { Mdd.setMood('sad'); Mdd.say('에러예요...'); }
            }
            console.error('Chat Error:', e);
        } finally {
            currentStreamAbort = null;
            if (sendBtn) sendBtn.style.display = '';
            if (stopBtn) stopBtn.style.display = 'none';
        }
    }

    window._cb = {
        send: function () {
            var input = document.getElementById('cbInput');
            var text = input && input.value.trim();
            if (!text) return;
            if (typeof Gemini !== 'undefined' && Gemini.requireApiKey && !Gemini.requireApiKey()) return;

            UI.appendMsg('user', text + (state.pendingImages.length ? ' [📎 이미지 ' + state.pendingImages.length + '장]' : ''));
            input.value = '';

            var parts = [{ text: text }];
            state.pendingImages.forEach(function (img) {
                parts.push({ inlineData: { mimeType: img.mimeType, data: img.base64 } });
            });
            state.pendingImages = [];
            UI.renderAttachThumbs();
            state.chatHistory.push({ role: 'user', parts: parts });

            doSendOrRegenerate();
        },

        stopStream: function () {
            if (currentStreamAbort) {
                currentStreamAbort.abort();
                if (typeof Toolbox !== 'undefined') Toolbox.showToast('스트리밍 중지');
            }
        },

        regenerate: function () {
            if (state.chatHistory.length < 2) return;
            var last = state.chatHistory[state.chatHistory.length - 1];
            if (last && last.role === 'model') state.chatHistory.pop();
            var msgs = document.getElementById('cbMessages');
            var wraps = msgs && msgs.querySelectorAll('.cb-msg-wrap');
            if (wraps && wraps.length) wraps[wraps.length - 1].remove();
            saveSession();
            var lastUser = state.chatHistory[state.chatHistory.length - 1];
            if (!lastUser || lastUser.role !== 'user') return;
            doSendOrRegenerate();
        },

        clearChat: function () {
            state.chatHistory = [];
            state.conversationSummary = '';
            state.pendingImages = [];
            UI.renderAttachThumbs();
            if (state.currentSessionId) {
                try { sessionStorage.removeItem(Session.PREFIX + state.currentSessionId); } catch (_) {}
            }
            var msgs = document.getElementById('cbMessages');
            if (msgs) {
                msgs.innerHTML = '';
                UI.appendMsg('bot', '대화가 초기화되었습니다. 무엇을 도와드릴까요?');
            }
            if (typeof Toolbox !== 'undefined') Toolbox.showToast('대화 초기화 완료');
            if (typeof Mdd !== 'undefined') { Mdd.setMood('idle'); Mdd.say('새로 시작이에요!'); }
        },

        importChat: function () {
            var input = document.createElement('input');
            input.type = 'file';
            input.accept = '.txt,.json';
            input.onchange = function () {
                var file = input.files && input.files[0];
                if (!file) return;
                file.text().then(function (text) {
                    try {
                        var imported = [];
                        if (file.name.indexOf('.json') !== -1) {
                            var data = JSON.parse(text);
                            imported = Array.isArray(data) ? data : [];
                            if (!imported.length) throw new Error('잘못된 JSON 형식');
                        } else {
                            var blocks = text.split(/\n\n/).filter(function (b) { return b.trim(); });
                            for (var i = 0; i < blocks.length; i++) {
                                var m = blocks[i].match(/^\[(You|AI)\]\n([\s\S]*)$/);
                                if (m) imported.push({ role: m[1] === 'You' ? 'user' : 'model', parts: [{ text: m[2] }] });
                            }
                        }
                        if (!imported.length) { Toolbox.showToast('가져올 대화를 찾을 수 없습니다.', 'error'); return; }
                        var sessionName = file.name.replace(/\.[^.]+$/, '');
                        var newId = Session.createNewSession(sessionName);
                        state.currentSessionId = newId;
                        state.chatHistory = imported;
                        saveSession();
                        UI.renderSessionTabs();
                        var msgs = document.getElementById('cbMessages');
                        if (msgs) {
                            msgs.innerHTML = '';
                            state.chatHistory.forEach(function (msg) {
                                var role = msg.role === 'user' ? 'user' : 'bot';
                                var t = (msg.parts && msg.parts[0]) ? msg.parts[0].text : '';
                                if (t) UI.appendMsg(role, t, false);
                            });
                            msgs.scrollTop = msgs.scrollHeight;
                        }
                        Toolbox.showToast(imported.length + '개 메시지를 가져왔습니다.');
                    } catch (e) {
                        Toolbox.showToast('가져오기 실패: ' + e.message, 'error');
                    }
                });
            };
            input.click();
        },

        exportChat: function (format) {
            if (format === undefined) format = 'txt';
            if (state.chatHistory.length === 0) {
                Toolbox.showToast('내보낼 대화가 없습니다.', 'error');
                return;
            }
            var date = new Date().toISOString().slice(0, 10);
            var blob, filename;
            if (format === 'json') {
                blob = new Blob([JSON.stringify(state.chatHistory, null, 2)], { type: 'application/json;charset=utf-8' });
                filename = 'chat-export-' + date + '.json';
            } else {
                var lines = state.chatHistory.map(function (m) {
                    var role = m.role === 'user' ? 'You' : 'AI';
                    var text = (m.parts && m.parts[0]) ? m.parts[0].text : '';
                    return '[' + role + ']\n' + text;
                });
                blob = new Blob([lines.join('\n\n')], { type: 'text/plain;charset=utf-8' });
                filename = 'chat-export-' + date + '.txt';
            }
            var a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = filename;
            a.click();
            URL.revokeObjectURL(a.href);
            Toolbox.showToast('대화 내보내기 완료 (' + format.toUpperCase() + ')');
        }
    };

    window.Chatbot.Actions = { switchSession: switchSession, saveSession: saveSession };
})();
