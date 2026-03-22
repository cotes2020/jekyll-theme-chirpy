/**
 * Gemini API 공통 모듈
 *
 * Toolbox 전역에서 사용하는 Gemini/Imagen API 헬퍼.
 * - API 키 관리 (localStorage)
 * - 모델 목록 정의
 * - fetchWithRetry, callText, callChat, callChatStream, callGeminiImage, callImagen
 * - enhancePrompt, buildApiKeyUI
 *
 * 모델 목록 출처: Google AI Studio (aistudio.google.com) 확인 목록 기준.
 * - 문서: https://ai.google.dev/gemini-api/docs/models
 * - 이미지: https://ai.google.dev/gemini-api/docs/image-generation
 * - Imagen: https://ai.google.dev/gemini-api/docs/imagen
 */
const Gemini = (() => {
    /* ===== 모델 정의 (API ListModels 2026-03-18 조회 기준, .cursor/rules/google-ai-models.mdc 참조) ===== */
    const MODELS = {
        /* Gemini 시리즈: 채팅/텍스트 (generateContent 지원) */
        gemini: [
            { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', isDefault: true },
            { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro' },
            { id: 'gemini-2.0-flash', name: 'Gemini 2 Flash' },
            { id: 'gemini-2.0-flash-lite', name: 'Gemini 2 Flash Lite' },
            { id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash Lite' },
            { id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash' },
            { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro' },
            { id: 'gemini-3.1-flash-lite-preview', name: 'Gemini 3.1 Flash Lite' }
        ],
        /* Nano Banana 시리즈: 이미지 생성 (generateContent + responseModalities: IMAGE) */
        geminiImage: [
            { id: 'gemini-2.5-flash-image', name: 'Nano Banana (Gemini 2.5 Flash Image)', isDefault: true },
            { id: 'gemini-3-pro-image-preview', name: 'Nano Banana Pro (Gemini 3 Pro Image)' },
            { id: 'gemini-3.1-flash-image-preview', name: 'Nano Banana 2 (Gemini 3.1 Flash Image)' }
        ],
        /* Imagen 시리즈 (predict 엔드포인트) */
        imagen: [
            { id: 'imagen-4.0-generate-001', name: 'Imagen 4 Generate', isDefault: true },
            { id: 'imagen-4.0-ultra-generate-001', name: 'Imagen 4 Ultra Generate' },
            { id: 'imagen-4.0-fast-generate-001', name: 'Imagen 4 Fast Generate' }
        ]
    };

    const STORAGE_KEY = 'toolbox_gemini_api_key'; // legacy single-key
    const KEYS_STORE_KEY = 'toolbox_gemini_api_keys_v2';

    /* ===== API 키 관리 (다중 프로필) ===== */
    function getKeyStore() {
        // 1) 최신 구조: { activeId, profiles: [{id,name,key}] }
        try {
            const parsed = JSON.parse(localStorage.getItem(KEYS_STORE_KEY) || 'null');
            if (parsed && typeof parsed === 'object') {
                if (Array.isArray(parsed.profiles)) {
                    if (!parsed.profiles.length) {
                        parsed.profiles.push({ id: 'default', name: '기본', key: '' });
                    }
                    parsed.activeId = parsed.activeId || parsed.profiles[0].id;
                    return parsed;
                }
                // 2) 구버전 v2: { active: 'free'|'paid', keys: { free, paid } }
                if (parsed.keys) {
                    const profiles = [];
                    if (parsed.keys.free) profiles.push({ id: 'free', name: '무료', key: parsed.keys.free });
                    if (parsed.keys.paid) profiles.push({ id: 'paid', name: '유료', key: parsed.keys.paid });
                    if (!profiles.length) profiles.push({ id: 'default', name: '기본', key: '' });
                    const activeId = profiles.find(p => p.id === parsed.active)?.id || profiles[0].id;
                    const migrated = { activeId, profiles };
                    localStorage.setItem(KEYS_STORE_KEY, JSON.stringify(migrated));
                    return migrated;
                }
            }
        } catch (_) {}

        // 3) 더 구버전: STORAGE_KEY 단일 키
        const legacy = localStorage.getItem(STORAGE_KEY) || '';
        const baseProfile = legacy
            ? { id: 'default', name: '기본', key: legacy }
            : { id: 'default', name: '기본', key: '' };
        const store = { activeId: baseProfile.id, profiles: [baseProfile] };
        localStorage.setItem(KEYS_STORE_KEY, JSON.stringify(store));
        return store;
    }

    function saveKeyStore(store) {
        localStorage.setItem(KEYS_STORE_KEY, JSON.stringify(store));
    }

    function getProfiles() {
        const store = getKeyStore();
        return store.profiles || [];
    }

    function getActiveProfileId() {
        const store = getKeyStore();
        return store.activeId;
    }

    function setActiveProfileId(id) {
        const store = getKeyStore();
        if (!store.profiles?.some(p => p.id === id)) return;
        store.activeId = id;
        saveKeyStore(store);
    }

    function getActiveProfileName() {
        const store = getKeyStore();
        const p = store.profiles?.find(p => p.id === store.activeId);
        return p ? p.name : '';
    }

    function getApiKey(id = null) {
        const store = getKeyStore();
        const pid = id || store.activeId;
        const p = store.profiles?.find(p => p.id === pid);
        return p ? (p.key || '') : '';
    }

    function setApiKey(key, id = null, nameOverride) {
        const store = getKeyStore();
        let pid = id || store.activeId;
        if (!pid) {
            pid = 'profile_' + Date.now();
            store.profiles = store.profiles || [];
            store.profiles.push({ id: pid, name: nameOverride || '프로필', key: '' });
            store.activeId = pid;
        }
        const idx = (store.profiles || []).findIndex(p => p.id === pid);
        if (idx >= 0) {
            store.profiles[idx].key = key;
            if (nameOverride) store.profiles[idx].name = nameOverride;
        } else {
            store.profiles.push({ id: pid, name: nameOverride || '프로필', key });
        }
        saveKeyStore(store);
    }

    function requireApiKey() {
        const key = getApiKey();
        if (!key) {
            Toolbox.showToast('API 키를 먼저 설정해주세요.', 'error');
            return null;
        }
        return key;
    }

    /* ===== 기본 모델 ===== */
    function getDefaultModel(provider = 'gemini') {
        const models = MODELS[provider];
        if (!models || models.length === 0) return '';
        const def = models.find(m => m.isDefault);
        return def ? def.id : models[0].id;
    }

    /* ===== API 요청/응답 히스토리 (디버깅용, 메모리만) ===== */
    const apiHistory = [];
    const API_HISTORY_MAX = 20;

    function recordApiCall(entry) {
        const copy = { ...entry, ts: new Date().toISOString() };
        if (copy.responseBody && copy.responseBody.candidates) {
            copy.responseBody = JSON.parse(JSON.stringify(copy.responseBody));
            copy.responseBody.candidates?.forEach(c => {
                c.content?.parts?.forEach(p => {
                    if (p.inlineData?.data) p.inlineData.data = `[base64 ${p.inlineData.data.length} chars]`;
                });
            });
        }
        if (copy.responseBody?.predictions) {
            copy.responseBody = JSON.parse(JSON.stringify(copy.responseBody));
            copy.responseBody.predictions?.forEach(p => {
                if (p.bytesBase64Encoded) p.bytesBase64Encoded = `[base64 ${p.bytesBase64Encoded.length} chars]`;
            });
        }
        if (copy.url) copy.url = maskUrl(copy.url);
        apiHistory.unshift(copy);
        if (apiHistory.length > API_HISTORY_MAX) apiHistory.pop();
    }

    function getApiHistory() { return [...apiHistory]; }
    function clearApiHistory() { apiHistory.length = 0; }

    /* ===== HTTP 헬퍼 ===== */
    function maskKey(key) {
        if (!key || key.length < 8) return '••••';
        return key.slice(0, 6) + '...' + key.slice(-4);
    }

    function maskUrl(url) {
        try {
            const u = new URL(url);
            const k = u.searchParams.get('key');
            if (k) u.searchParams.set('key', maskKey(k));
            return u.toString();
        } catch (_) { return url.replace(/key=[^&]+/, 'key=••••'); }
    }

    async function fetchWithRetry(url, body, options = {}) {
        try {
            const fetchOpts = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            };
            if (options.signal) fetchOpts.signal = options.signal;
            const response = await fetch(url, fetchOpts);

            if (!response.ok) {
                let errorDetails = '상세 정보 없음';
                let fullError = null;
                try {
                    fullError = await response.json();
                    errorDetails = fullError.error?.message || JSON.stringify(fullError);
                } catch (e) {
                    errorDetails = await response.text();
                }

                if (body && (url.includes('generateContent') || url.includes('predict'))) {
                    const type = url.includes('predict') ? 'imagen' : 'geminiImage';
                    recordApiCall({
                        type,
                        url,
                        method: 'POST',
                        requestBody: body,
                        responseBody: fullError || { error: errorDetails },
                        status: response.status
                    });
                }

                let userMessage = errorDetails;
                if (response.status === 401) userMessage = 'API 키 인증 실패. 올바른 API 키를 확인해주세요.';
                else if (response.status === 403) userMessage = 'API 접근이 거부되었습니다.';
                else if (response.status === 429) userMessage = 'API 요청 한도 초과. 잠시 후 다시 시도해주세요.';
                else if (response.status === 500) userMessage = 'Google API 서버 오류. 잠시 후 다시 시도해주세요.';

                console.warn(`[Gemini API] ${response.status} ${url ? maskUrl(url) : ''}`);
                throw new Error(`[HTTP ${response.status}] ${userMessage}`);
            }

            return response;
        } catch (e) {
            if (e.name === 'AbortError') throw new Error('요청이 취소되었습니다.');
            if (e.message.includes('Failed to fetch') || e.message.includes('NetworkError')) {
                throw new Error('네트워크 연결 오류. 인터넷 연결을 확인해주세요.');
            }
            throw e;
        }
    }

    /* ===== 텍스트 생성 ===== */
    async function callText(userText, systemPrompt, modelId) {
        const key = requireApiKey();
        if (!key) return null;

        const model = modelId || getDefaultModel('gemini');
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;
        const body = {
            contents: [{ parts: [{ text: userText }] }],
            systemInstruction: { parts: [{ text: systemPrompt }] },
            generationConfig: { maxOutputTokens: 8192 }
        };

        const res = await fetchWithRetry(url, body);
        const data = await res.json();

        if (!data.candidates || !data.candidates[0]) {
            throw new Error('응답에 결과가 없습니다: ' + JSON.stringify(data));
        }

        return {
            text: data.candidates[0].content.parts[0].text,
            usage: data.usageMetadata
        };
    }

    /* ===== 챗 (멀티턴) ===== */
    async function callChat(history, systemPrompt, modelId, options = {}) {
        const key = requireApiKey();
        if (!key) return null;

        const model = modelId || getDefaultModel('gemini');
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;
        const body = {
            contents: history,
            systemInstruction: { parts: [{ text: systemPrompt }] },
            generationConfig: {
                temperature: options.temperature || 0.8,
                maxOutputTokens: options.maxTokens || 8192
            }
        };

        if (options.webSearch) {
            body.tools = [{ googleSearch: {} }];
        }

        const res = await fetchWithRetry(url, body);
        const data = await res.json();

        if (!data.candidates || !data.candidates[0]?.content) {
            throw new Error('응답에 결과가 없습니다.');
        }

        return {
            text: data.candidates[0].content.parts[0].text,
            usage: data.usageMetadata
        };
    }

    /* ===== 챗 스트리밍 ===== */
    async function callChatStream(history, systemPrompt, modelId, options = {}) {
        const key = requireApiKey();
        if (!key) return null;

        const model = modelId || getDefaultModel('gemini');
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse&key=${key}`;
        const body = {
            contents: history,
            systemInstruction: { parts: [{ text: systemPrompt }] },
            generationConfig: {
                temperature: options.temperature || 0.8,
                maxOutputTokens: options.maxTokens || 8192
            }
        };
        if (options.webSearch) body.tools = [{ googleSearch: {} }];

        const fetchOpts = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        };
        if (options.signal) fetchOpts.signal = options.signal;

        const response = await fetch(url, fetchOpts);
        if (!response.ok) {
            let msg = `[HTTP ${response.status}]`;
            try { const e = await response.json(); msg += ' ' + (e.error?.message || JSON.stringify(e)); } catch (_) {}
            throw new Error(msg);
        }

        return {
            async *chunks() {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const jsonStr = line.slice(6).trim();
                        if (!jsonStr || jsonStr === '[DONE]') continue;
                        try {
                            const parsed = JSON.parse(jsonStr);
                            const text = parsed.candidates?.[0]?.content?.parts?.[0]?.text;
                            if (text) yield { text, usage: parsed.usageMetadata || null };
                        } catch (_) {}
                    }
                }
            }
        };
    }

    /* ===== Gemini 이미지 생성 (NanoBanana) ===== */
    /**
     * options: { signal, aspectRatio, safetyThreshold }
     * - aspectRatio: '1:1', '16:9', '9:16', '3:4', '4:3' 등
     * - safetyThreshold: 'OFF' | 'BLOCK_NONE' | 'BLOCK_ONLY_HIGH' | 'BLOCK_MEDIUM_AND_ABOVE' | 'BLOCK_LOW_AND_ABOVE'
     */
    async function callGeminiImage(prompt, modelId, options = {}) {
        const key = requireApiKey();
        if (!key) throw new Error('API 키가 설정되지 않았습니다.');

        const model = modelId || getDefaultModel('geminiImage');
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;

        const genConfig = {
            maxOutputTokens: 8192,
            responseModalities: ['TEXT', 'IMAGE']
        };
        if (options.aspectRatio) {
            genConfig.imageConfig = { aspectRatio: options.aspectRatio };
        }

        const threshold = options.safetyThreshold || 'BLOCK_ONLY_HIGH';
        const body = {
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: genConfig,
            safetySettings: [
                { category: 'HARM_CATEGORY_HARASSMENT', threshold },
                { category: 'HARM_CATEGORY_HATE_SPEECH', threshold },
                { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold },
                { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold },
                { category: 'HARM_CATEGORY_CIVIC_INTEGRITY', threshold }
            ]
        };

        const res = await fetchWithRetry(url, body, { signal: options.signal });
        const data = await res.json();

        recordApiCall({
            type: 'geminiImage',
            url,
            method: 'POST',
            requestBody: body,
            responseBody: data,
            status: res.status
        });

        if (data.error) {
            const msg = data.error.message || data.error.status || 'API 오류';
            throw new Error(msg);
        }
        if (data.candidates && data.candidates[0]?.finishReason === 'SAFETY') {
            throw new Error('안전 필터에 의해 차단되었습니다.');
        }
        if (!data.candidates || data.candidates.length === 0) {
            throw new Error('이미지 데이터가 없습니다. (API가 candidates를 반환하지 않음)');
        }
        const cand = data.candidates[0];
        if (!cand.content || !cand.content.parts?.length) {
            const reason = cand.finishReason || 'content 없음';
            throw new Error(`이미지 데이터가 없습니다. (finishReason: ${reason})`);
        }

        const imageData = cand.content.parts.find(p => p.inlineData);
        if (!imageData) {
            const hasText = cand.content.parts.some(p => p.text);
            throw new Error(hasText
                ? '이미지 대신 텍스트만 반환되었습니다. 이 모델은 이미지 생성이 제한되거나, 프롬프트/안전 필터로 인해 이미지가 생성되지 않았을 수 있습니다.'
                : '이미지를 찾을 수 없습니다.');
        }

        return {
            dataUrl: `data:image/png;base64,${imageData.inlineData.data}`,
            usage: data.usageMetadata
        };
    }

    /* ===== Imagen 이미지 생성 ===== */
    /**
     * options: { signal, aspectRatio, safetyFilterLevel, negativePrompt, personGeneration }
     * - aspectRatio: '1:1'(기본), '16:9', '9:16', '3:4', '4:3'
     * - safetyFilterLevel: 'off' | 'block_none' | 'block_only_high' | 'block_medium_and_above' | 'block_low_and_above'
     * - negativePrompt: 제외할 요소 텍스트
     * - personGeneration: 'allow_adult'(기본) | 'allow_all' | 'dont_allow'
     */
    async function callImagen(prompt, modelId, count = 1, options = {}) {
        const key = requireApiKey();
        if (!key) throw new Error('API 키가 설정되지 않았습니다.');

        const model = modelId || getDefaultModel('imagen');
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:predict?key=${key}`;

        const params = { sampleCount: count };
        if (options.aspectRatio) params.aspectRatio = options.aspectRatio;
        if (options.safetyFilterLevel) params.safetyFilterLevel = options.safetyFilterLevel;
        if (options.personGeneration) params.personGeneration = options.personGeneration;

        const instance = { prompt };
        if (options.negativePrompt) instance.negativePrompt = options.negativePrompt;

        const body = { instances: [instance], parameters: params };

        const res = await fetchWithRetry(url, body, { signal: options.signal });
        const data = await res.json();

        recordApiCall({
            type: 'imagen',
            url,
            method: 'POST',
            requestBody: body,
            responseBody: data,
            status: res.status
        });

        if (!data.predictions || data.predictions.length === 0) {
            throw new Error('이미지 생성 결과가 없습니다.');
        }

        return data.predictions.map(p => {
            if (!p.bytesBase64Encoded) throw new Error('base64 이미지 데이터 없음');
            return `data:image/png;base64,${p.bytesBase64Encoded}`;
        });
    }

    /* ===== 공통 API 키 UI 빌더 (프로필 리스트) ===== */
    const AI_STUDIO_API_KEY_URL = 'https://aistudio.google.com/app/apikey';

    function buildApiKeyUI(idPrefix) {
        const store = getKeyStore();
        const profiles = store.profiles || [];
        const activeId = store.activeId;
        const html = `
            <div class="field-group">
                <label class="field-label">🔑 Gemini API 키 프로필</label>
                <div id="${idPrefix}ApiProfileList" style="display:flex;flex-direction:column;gap:6px;margin-bottom:8px;"></div>
                <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">
                    <button type="button" class="btn btn-ghost" id="${idPrefix}ApiAdd" style="font-size:var(--font-size-xs);padding:4px 10px;">+ 프로필 추가</button>
                    <span style="font-size:var(--font-size-2xs);color:var(--text-tertiary);">여러 개의 키를 저장해두고 상황에 따라 활성 프로필을 선택할 수 있습니다.</span>
                </div>
                <div style="font-size:var(--font-size-2xs);margin-top:4px;color:var(--text-tertiary);">
                    Google AI Studio에서 발급: <a href="${AI_STUDIO_API_KEY_URL}" target="_blank" rel="noopener" style="color:var(--accent);">API 키 발급 페이지</a>
                </div>
                <div style="font-size:var(--font-size-2xs);margin-top:4px;color:#facc15;">
                    ⚠️ API 키는 브라우저 localStorage에 평문 저장됩니다. 공용 PC에서는 사용 후 반드시 삭제하세요.
                </div>
            </div>`;
        return {
            html,
            init(container) {
                const root = container || document;
                const listEl = root.querySelector ? root.querySelector(`#${idPrefix}ApiProfileList`) : document.getElementById(`${idPrefix}ApiProfileList`);
                const addBtn = root.querySelector ? root.querySelector(`#${idPrefix}ApiAdd`) : document.getElementById(`${idPrefix}ApiAdd`);

                function render() {
                    const store = getKeyStore();
                    const profiles = store.profiles || [];
                    const activeId = store.activeId;
                    if (!listEl) return;
                    listEl.innerHTML = profiles.map(p => {
                        const hasKey = !!p.key;
                        return `
                            <div class="api-prof-row" data-id="${p.id}" style="display:flex;align-items:center;gap:6px;padding:4px 6px;border:1px solid var(--border);border-radius:6px;background:${p.id === activeId ? 'var(--accent-subtle)' : 'transparent'};">
                                <input type="text" class="api-prof-name" value="${Toolbox.escapeHtml(p.name || '')}" placeholder="프로필 이름" style="flex:1;min-width:80px;font-size:var(--font-size-xs);padding:4px 6px;border-radius:4px;border:1px solid var(--border);background:var(--bg-primary);color:var(--text-primary);">
                                <span class="api-prof-status" style="font-size:var(--font-size-2xs);color:${hasKey ? 'var(--success)' : 'var(--text-tertiary)'};">${hasKey ? '키 저장됨' : '키 없음'}</span>
                                <button type="button" class="btn btn-ghost api-prof-active" data-role="active" style="font-size:var(--font-size-2xs);padding:3px 8px;${p.id === activeId ? 'color:var(--accent);' : ''}">${p.id === activeId ? '✓ 활성' : '활성화'}</button>
                                <button type="button" class="btn btn-ghost api-prof-edit" data-role="edit" style="font-size:var(--font-size-2xs);padding:3px 8px;">키 변경</button>
                                <button type="button" class="btn btn-ghost api-prof-del" data-role="delete" style="font-size:var(--font-size-2xs);padding:3px 8px;color:var(--error);">삭제</button>
                            </div>`;
                    }).join('');
                }

                render();

                if (addBtn) {
                    addBtn.addEventListener('click', () => {
                        const store = getKeyStore();
                        const profiles = store.profiles || [];
                        const newId = 'profile_' + Date.now();
                        const newName = '프로필 ' + (profiles.length + 1);
                        profiles.push({ id: newId, name: newName, key: '' });
                        store.profiles = profiles;
                        store.activeId = newId;
                        saveKeyStore(store);
                        window.dispatchEvent(new CustomEvent('gemini-active-profile-changed'));
                        render();
                    });
                }

                if (listEl) {
                    listEl.addEventListener('input', (e) => {
                        const target = e.target;
                        if (!(target instanceof HTMLInputElement) || !target.classList.contains('api-prof-name')) return;
                        const row = target.closest('.api-prof-row');
                        const id = row?.getAttribute('data-id');
                        if (!id) return;
                        const store = getKeyStore();
                        const idx = (store.profiles || []).findIndex(p => p.id === id);
                        if (idx >= 0) {
                            store.profiles[idx].name = target.value.trim() || '프로필';
                            saveKeyStore(store);
                        }
                    });

                    listEl.addEventListener('click', (e) => {
                        const btn = e.target.closest('button');
                        if (!btn) return;
                        const role = btn.getAttribute('data-role');
                        const row = btn.closest('.api-prof-row');
                        const id = row?.getAttribute('data-id');
                        if (!id) return;
                        const store = getKeyStore();
                        const profiles = store.profiles || [];
                        const idx = profiles.findIndex(p => p.id === id);
                        if (idx < 0) return;

                        if (role === 'active') {
                            store.activeId = id;
                            saveKeyStore(store);
                            window.dispatchEvent(new CustomEvent('gemini-active-profile-changed'));
                            Toolbox.showToast('활성 프로필이 변경되었습니다.');
                            render();
                        } else if (role === 'edit') {
                            const current = profiles[idx];
                            const val = prompt('새 API 키를 입력하세요 (빈 값 입력 시 키 삭제)', current.key ? '••••••••••' : '');
                            if (val === null) return;
                            const trimmed = val.trim();
                            if (!trimmed || trimmed === '••••••••••') {
                                profiles[idx].key = '';
                                saveKeyStore(store);
                                Toolbox.showToast('키가 삭제되었습니다.');
                                render();
                                return;
                            }
                            if (!trimmed.startsWith('AIza')) {
                                Toolbox.showToast('⚠️ 유효하지 않은 키 형식입니다.', 'error');
                                return;
                            }
                            profiles[idx].key = trimmed;
                            saveKeyStore(store);
                            Toolbox.showToast('API 키 저장 완료');
                            render();
                        } else if (role === 'delete') {
                            if (profiles.length === 1) {
                                Toolbox.showToast('최소 1개의 프로필은 필요합니다.', 'error');
                                return;
                            }
                            if (!confirm('이 프로필을 삭제하시겠습니까?')) return;
                            profiles.splice(idx, 1);
                            if (store.activeId === id) {
                                store.activeId = profiles[0]?.id || '';
                                window.dispatchEvent(new CustomEvent('gemini-active-profile-changed'));
                            }
                            store.profiles = profiles;
                            saveKeyStore(store);
                            Toolbox.showToast('프로필이 삭제되었습니다.');
                            render();
                        }
                    });
                }
            }
        };
    }

    /* ===== 프롬프트 향상 ===== */
    async function enhancePrompt(originalPrompt) {
        const systemPrompt = `You are a prompt enhancement expert for AI image generation.
Improve the given prompt to produce higher quality, more detailed, and more visually striking images.
Keep the same subject and intent, but add details about lighting, composition, art style, colors, and atmosphere.
Reply ONLY with the enhanced prompt in English, nothing else. Do not add any explanation.`;

        const result = await callText(originalPrompt, systemPrompt);
        if (!result) throw new Error('프롬프트 향상 실패');
        return result.text.trim();
    }

    /* ===== Public API ===== */
    return {
        MODELS,
        getApiKey, setApiKey, getProfiles, getActiveProfileId, setActiveProfileId, getActiveProfileName, requireApiKey, getDefaultModel,
        fetchWithRetry, callText, callChat, callChatStream, callGeminiImage, callImagen,
        getApiHistory, clearApiHistory,
        buildApiKeyUI, enhancePrompt
    };
})();

/**
 * ImageDB — IndexedDB 기반 이미지 라이브러리 공유 모듈.
 * imagegen, imagelib 등 여러 위젯에서 공통으로 사용.
 */
const ImageDB = (() => {
    const DB_NAME = 'toolbox_imagegen';
    const STORE_NAME = 'images';
    const DB_VERSION = 1;

    function open() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(DB_NAME, DB_VERSION);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(STORE_NAME)) {
                    const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                }
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }

    function notify(action, detail) {
        window.dispatchEvent(new CustomEvent('imagedb-change', { detail: { action, ...detail } }));
    }

    async function save(item) {
        const db = await open();
        try {
            await new Promise((resolve, reject) => {
                const tx = db.transaction(STORE_NAME, 'readwrite');
                tx.objectStore(STORE_NAME).put(item);
                tx.oncomplete = () => resolve();
                tx.onerror = () => reject(tx.error);
            });
            notify('save', { id: item.id });
        } finally { db.close(); }
    }

    async function getAll() {
        const db = await open();
        try {
            return await new Promise((resolve, reject) => {
                const tx = db.transaction(STORE_NAME, 'readonly');
                const req = tx.objectStore(STORE_NAME).index('timestamp').getAll();
                req.onsuccess = () => resolve(req.result.reverse());
                req.onerror = () => reject(req.error);
            });
        } finally { db.close(); }
    }

    async function remove(id) {
        const db = await open();
        try {
            await new Promise((resolve, reject) => {
                const tx = db.transaction(STORE_NAME, 'readwrite');
                tx.objectStore(STORE_NAME).delete(id);
                tx.oncomplete = () => resolve();
                tx.onerror = () => reject(tx.error);
            });
            notify('remove', { id });
        } finally { db.close(); }
    }

    async function clear() {
        const db = await open();
        try {
            await new Promise((resolve, reject) => {
                const tx = db.transaction(STORE_NAME, 'readwrite');
                tx.objectStore(STORE_NAME).clear();
                tx.oncomplete = () => resolve();
                tx.onerror = () => reject(tx.error);
            });
            notify('clear');
        } finally { db.close(); }
    }

    return { save, getAll, remove, clear };
})();
