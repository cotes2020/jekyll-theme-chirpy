/**
 * Gemini API 공통 모듈
 *
 * Toolbox 전역에서 사용하는 Gemini/Imagen API 헬퍼.
 * - API 키 관리 (localStorage)
 * - 모델 목록 정의
 * - fetchWithRetry, callText, callChat, callChatStream, callGeminiImage (선택: referenceImage), callImagen
 * - enhancePrompt, buildApiKeyUI
 * - Vertex AI용 Google Cloud API 키 (별도 localStorage, AI Studio 키와 독립)
 *
 * 모델 목록 출처: Google AI Studio (aistudio.google.com) 확인 목록 기준.
 * - 문서: https://ai.google.dev/gemini-api/docs/models
 * - 이미지: https://ai.google.dev/gemini-api/docs/image-generation
 * - Imagen: https://ai.google.dev/gemini-api/docs/imagen
 *
 * 모델 ID·목록 SSOT: packages/karmolab-ai (KarmoLabAI)
 */
// @ts-nocheck — large API surface; narrow types incrementally later (Toolbox/Gemini shapes)
import { MODEL_CATALOG as MODELS, getDefaultModelId } from 'karmolab-ai';

const Gemini = (() => {
    const STORAGE_KEY = 'toolbox_gemini_api_key'; // legacy single-key
    const KEYS_STORE_KEY = 'toolbox_gemini_api_keys_v2';
    /** Vertex AI (Google Cloud API 키 / Express 모드 등) — AI Studio 키와 별도 */
    const VERTEX_API_KEY_STORAGE = 'toolbox_vertex_api_key';

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

    function getVertexApiKey() {
        try {
            return localStorage.getItem(VERTEX_API_KEY_STORAGE) || '';
        } catch (_) {
            return '';
        }
    }

    function setVertexApiKey(key) {
        try {
            if (key) localStorage.setItem(VERTEX_API_KEY_STORAGE, key);
            else localStorage.removeItem(VERTEX_API_KEY_STORAGE);
            window.dispatchEvent(new CustomEvent('vertex-api-key-changed'));
        } catch (_) {}
    }

    function requireVertexApiKey() {
        const key = getVertexApiKey();
        if (!key) {
            Toolbox.showToast('Vertex AI API 키를 설정에서 먼저 입력해주세요.', 'error');
            return null;
        }
        return key;
    }

    /* ===== 기본 모델 (catalog: karmolab-ai) ===== */
    function getDefaultModel(provider = 'gemini') {
        return getDefaultModelId(provider);
    }

    /* ===== API 요청/응답 히스토리 (디버깅용, 메모리만) ===== */
    const apiHistory = [];
    const API_HISTORY_MAX = 20;

    function recordApiCall(entry) {
        const copy = { ...entry, ts: new Date().toISOString() };
        if (copy.requestBody && typeof copy.requestBody === 'object') {
            try {
                copy.requestBody = JSON.parse(JSON.stringify(copy.requestBody));
                copy.requestBody.contents?.forEach(c => {
                    c.parts?.forEach(p => {
                        if (p.inlineData?.data) {
                            const len = String(p.inlineData.data).length;
                            p.inlineData.data = `[base64 ${len} chars]`;
                        }
                    });
                });
            } catch (_) {}
        }
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
     * options: { signal, aspectRatio, safetyThreshold, referenceImage, referenceMimeType }
     * - aspectRatio: '1:1', '16:9', '9:16', '3:4', '4:3' 등
     * - safetyThreshold: 'OFF' | 'BLOCK_NONE' | 'BLOCK_ONLY_HIGH' | 'BLOCK_MEDIUM_AND_ABOVE' | 'BLOCK_LOW_AND_ABOVE'
     * - referenceImage: data URL 또는 순수 base64(편집·업스케일 등 입력 이미지)
     * - referenceMimeType: 기본 image/png
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

        const parts = [{ text: prompt }];
        if (options.referenceImage) {
            let b64 = options.referenceImage;
            if (typeof b64 === 'string' && b64.includes(',')) b64 = b64.split(',')[1];
            if (typeof b64 !== 'string' || !b64.length) throw new Error('참조 이미지(base64)가 비어 있습니다.');
            parts.push({
                inlineData: {
                    mimeType: options.referenceMimeType || 'image/png',
                    data: b64
                }
            });
        }

        const threshold = options.safetyThreshold || 'BLOCK_ONLY_HIGH';
        const body = {
            contents: [{ parts }],
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

    /* ===== Vertex Gemini 이미지 생성 (generateContent on Vertex) ===== */
    /**
     * Vertex AI Gemini 이미지 생성.
     * - 인증: Vertex용 Google Cloud API 키 (toolbox_vertex_api_key)
     * - 엔드포인트: https://aiplatform.googleapis.com/v1/publishers/google/models/${model}:generateContent?key=...
     * - Imagen 이미지는 callVertexImagen(:predict) 사용.
     */
    async function callVertexGeminiImage(prompt, modelId, options = {}) {
        const key = requireVertexApiKey();
        if (!key) throw new Error('Vertex API 키가 설정되지 않았습니다.');

        const model = modelId || getDefaultModel('geminiImage');
        const projectId = (options.projectId || '').trim();
        if (!projectId) {
            throw new Error('Vertex Gemini 이미지에는 GCP 프로젝트 ID가 필요합니다. 위젯에 프로젝트 ID를 입력하세요.');
        }
        const location = (options.location || 'us-central1').trim() || 'us-central1';

        // Vertex AI REST: projects.locations.publishers.models.generateContent
        // https://aiplatform.googleapis.com/v1/{model}:generateContent
        // where {model} = projects/{project}/locations/{location}/publishers/google/models/{modelId}
        const url =
            `https://${encodeURIComponent(location)}-aiplatform.googleapis.com/v1/projects/${encodeURIComponent(projectId)}` +
            `/locations/${encodeURIComponent(location)}/publishers/google/models/${encodeURIComponent(model)}:generateContent?key=${key}`;

        const genConfig = {
            maxOutputTokens: 8192,
            responseModalities: ['TEXT', 'IMAGE']
        };
        if (options.aspectRatio) {
            genConfig.imageConfig = { aspectRatio: options.aspectRatio };
        }

        const parts = [{ text: prompt }];
        if (options.referenceImage) {
            let b64 = options.referenceImage;
            if (typeof b64 === 'string' && b64.includes(',')) b64 = b64.split(',')[1];
            if (typeof b64 !== 'string' || !b64.length) throw new Error('참조 이미지(base64)가 비어 있습니다.');
            parts.push({
                inlineData: {
                    mimeType: options.referenceMimeType || 'image/png',
                    data: b64
                }
            });
        }

        const threshold = options.safetyThreshold || 'BLOCK_ONLY_HIGH';
        const body = {
            contents: [{ parts }],
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
            type: 'vertexGeminiImage',
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
            usage: data.usageMetadata || null
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

    /** Vertex predict 응답의 이미지 base64 추출 (JSON 또는 protobuf Struct JSON) */
    function extractImagenPredictionBase64(p) {
        if (!p) return null;
        if (p.bytesBase64Encoded) return p.bytesBase64Encoded;
        const fields = p.structValue?.fields;
        const b64 = fields?.bytesBase64Encoded?.stringValue;
        return b64 || null;
    }

    /**
     * Vertex AI Imagen — Prediction API `:predict` (scripts/generate-image.cjs 와 동일 계열)
     * - 인증: Vertex용 Google Cloud API 키
     * - 엔드포인트: https://{location}-aiplatform.googleapis.com/v1/projects/.../models/...:predict?key=...
     * options: { signal, projectId, location, aspectRatio, safetyFilterLevel, negativePrompt, personGeneration }
     */
    async function callVertexImagen(prompt, modelId, count = 1, options = {}) {
        const key = requireVertexApiKey();
        if (!key) throw new Error('Vertex API 키가 설정되지 않았습니다.');

        const projectId = (options.projectId || '').trim();
        if (!projectId) {
            throw new Error('Vertex Imagen에는 GCP 프로젝트 ID가 필요합니다. 위젯에 프로젝트 ID를 입력하세요.');
        }

        const location = (options.location || 'us-central1').trim() || 'us-central1';
        const model = modelId || getDefaultModel('imagen');

        const url =
            `https://${encodeURIComponent(location)}-aiplatform.googleapis.com/v1/projects/${encodeURIComponent(projectId)}` +
            `/locations/${encodeURIComponent(location)}/publishers/google/models/${encodeURIComponent(model)}:predict?key=${key}`;

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
            type: 'vertexImagen',
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
        if (!data.predictions || data.predictions.length === 0) {
            throw new Error('이미지 생성 결과가 없습니다.');
        }

        return data.predictions.map((p) => {
            const b64 = extractImagenPredictionBase64(p);
            if (!b64) throw new Error('base64 이미지 데이터 없음 (필터 차단 또는 응답 형식 불일치)');
            return `data:image/png;base64,${b64}`;
        });
    }

    /* ===== 공통 API 키 UI 빌더 (프로필 리스트) ===== */
    const AI_STUDIO_API_KEY_URL = 'https://aistudio.google.com/app/apikey';
    const VERTEX_API_KEY_HELP_URL = 'https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys';

    function buildApiKeyUI(idPrefix) {
        const store = getKeyStore();
        const profiles = store.profiles || [];
        const activeId = store.activeId;
        const html = `
            <div class="field-group">
                <label class="field-label">🔑 Google AI Studio (Gemini API) — 프로필</label>
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
            </div>
            <div class="field-group">
                <label class="field-label">☁️ Vertex AI (Google Cloud API 키)</label>
                <p style="font-size:var(--font-size-2xs);color:var(--text-tertiary);margin:0 0 8px 0;">
                    AI Studio 키와는 <strong>별도</strong>입니다. Vertex / Express 모드 등에서 발급한 Google Cloud API 키를 넣습니다. (접두사는 <code>AIza</code>가 아닐 수 있어 형식 검증은 하지 않습니다.)
                </p>
                <div id="${idPrefix}VertexKeyRow" style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:8px;border:1px solid var(--border);border-radius:6px;margin-bottom:8px;">
                    <span id="${idPrefix}VertexKeyStatus" style="font-size:var(--font-size-2xs);flex:1;min-width:120px;"></span>
                    <button type="button" class="btn btn-ghost" id="${idPrefix}VertexKeyEdit" style="font-size:var(--font-size-2xs);padding:3px 10px;">키 변경</button>
                    <button type="button" class="btn btn-ghost" id="${idPrefix}VertexKeyClear" style="font-size:var(--font-size-2xs);padding:3px 10px;color:var(--error);">삭제</button>
                </div>
                <div style="font-size:var(--font-size-2xs);color:var(--text-tertiary);">
                    안내: <a href="${VERTEX_API_KEY_HELP_URL}" target="_blank" rel="noopener" style="color:var(--accent);">Vertex AI — API 키</a>
                </div>
                <div style="display:flex;flex-direction:column;gap:8px;margin-top:12px;">
                    <div>
                        <label class="field-label" for="${idPrefix}VertexProjectId" style="font-size:var(--font-size-xs);">GCP 프로젝트 ID</label>
                        <input type="text" id="${idPrefix}VertexProjectId" class="settings-control" style="width:100%;box-sizing:border-box;" placeholder="my-gcp-project-id" autocomplete="off">
                    </div>
                    <div>
                        <label class="field-label" for="${idPrefix}VertexLocation" style="font-size:var(--font-size-xs);">리전</label>
                        <input type="text" id="${idPrefix}VertexLocation" class="settings-control" style="width:100%;box-sizing:border-box;" placeholder="us-central1" autocomplete="off">
                    </div>
                </div>
                <p style="font-size:var(--font-size-2xs);color:var(--text-tertiary);margin:8px 0 0 0;line-height:1.4;">
                    Vertex 호출(이미지 생성 등)에 공통으로 쓰입니다. 이미지 생성 위젯의 입력란과 같은 값을 공유합니다.
                </p>
            </div>`;
        return {
            html,
            init(container) {
                const root = container || document;
                const listEl = root.querySelector ? root.querySelector(`#${idPrefix}ApiProfileList`) : document.getElementById(`${idPrefix}ApiProfileList`);
                const addBtn = root.querySelector ? root.querySelector(`#${idPrefix}ApiAdd`) : document.getElementById(`${idPrefix}ApiAdd`);

                const vertexStatusEl = root.querySelector(`#${idPrefix}VertexKeyStatus`);
                const vertexEditBtn = root.querySelector(`#${idPrefix}VertexKeyEdit`);
                const vertexClearBtn = root.querySelector(`#${idPrefix}VertexKeyClear`);
                const vertexProjectEl = root.querySelector(`#${idPrefix}VertexProjectId`);
                const vertexLocationEl = root.querySelector(`#${idPrefix}VertexLocation`);

                const VERTEX_PREF_PROJECT = 'ig_vertex_project_id';
                const VERTEX_PREF_LOCATION = 'ig_vertex_location';

                function syncVertexContextInputs() {
                    if (vertexProjectEl instanceof HTMLInputElement) {
                        const v = Toolbox.getPref(VERTEX_PREF_PROJECT) || '';
                        vertexProjectEl.value = typeof v === 'string' ? v : '';
                    }
                    if (vertexLocationEl instanceof HTMLInputElement) {
                        const v = Toolbox.getPref(VERTEX_PREF_LOCATION) || 'us-central1';
                        vertexLocationEl.value = (typeof v === 'string' && v.trim()) ? v.trim() : 'us-central1';
                    }
                }

                function renderVertexRow() {
                    if (!vertexStatusEl) return;
                    const has = !!getVertexApiKey();
                    vertexStatusEl.textContent = has ? '키 저장됨' : '키 없음';
                    vertexStatusEl.style.color = has ? 'var(--success)' : 'var(--text-tertiary)';
                }

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
                renderVertexRow();
                syncVertexContextInputs();

                if (vertexProjectEl instanceof HTMLInputElement) {
                    vertexProjectEl.addEventListener('change', () => {
                        Toolbox.setPref(VERTEX_PREF_PROJECT, vertexProjectEl.value.trim());
                        window.dispatchEvent(new CustomEvent('vertex-context-changed'));
                    });
                }
                if (vertexLocationEl instanceof HTMLInputElement) {
                    vertexLocationEl.addEventListener('change', () => {
                        const loc = vertexLocationEl.value.trim() || 'us-central1';
                        Toolbox.setPref(VERTEX_PREF_LOCATION, loc);
                        window.dispatchEvent(new CustomEvent('vertex-context-changed'));
                    });
                }

                if (vertexEditBtn) {
                    vertexEditBtn.addEventListener('click', () => {
                        const current = getVertexApiKey();
                        const val = prompt(
                            'Vertex AI용 Google Cloud API 키를 입력하세요 (빈 값이면 취소)',
                            current ? '••••••••••' : ''
                        );
                        if (val === null) return;
                        const trimmed = val.trim();
                        if (!trimmed || trimmed === '••••••••••') {
                            if (!current) return;
                            setVertexApiKey('');
                            Toolbox.showToast('Vertex API 키가 삭제되었습니다.');
                            renderVertexRow();
                            return;
                        }
                        setVertexApiKey(trimmed);
                        Toolbox.showToast('Vertex API 키 저장 완료');
                        renderVertexRow();
                    });
                }
                if (vertexClearBtn) {
                    vertexClearBtn.addEventListener('click', () => {
                        if (!getVertexApiKey()) return;
                        if (!confirm('저장된 Vertex API 키를 삭제할까요?')) return;
                        setVertexApiKey('');
                        Toolbox.showToast('Vertex API 키가 삭제되었습니다.');
                        renderVertexRow();
                    });
                }

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
        getApiKey, setApiKey, getProfiles, getActiveProfileId, setActiveProfileId, getActiveProfileName, requireApiKey,
        getVertexApiKey, setVertexApiKey, requireVertexApiKey,
        getDefaultModel,
        fetchWithRetry, callText, callChat, callChatStream, callGeminiImage, callImagen, callVertexImagen,
        callVertexGeminiImage,
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
