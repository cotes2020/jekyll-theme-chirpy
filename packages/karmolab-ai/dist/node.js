"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.resolveAiStudioTextModelId = resolveAiStudioTextModelId;
exports.createAiStudioTextModel = createAiStudioTextModel;
exports.generateAiStudioText = generateAiStudioText;
exports.generateVertexText = generateVertexText;
exports.parseGenerativeSurfaceFromEnv = parseGenerativeSurfaceFromEnv;
exports.generateBlobTextFromEnvWithOptions = generateBlobTextFromEnvWithOptions;
exports.tryCreateGenerativeTextFromEnv = tryCreateGenerativeTextFromEnv;
exports.generativeEnvHint = generativeEnvHint;
/**
 * Node 전용: AI Studio(`@google/generative-ai`) 또는 Vertex REST(`fetch`)로 텍스트 생성.
 * 브라우저 번들에 포함하지 말 것 — `import 'karmolab-ai/node'`.
 */
const generative_ai_1 = require("@google/generative-ai");
const index_1 = require("./index");
function resolveAiStudioTextModelId(modelFromEnv) {
    const t = modelFromEnv?.trim();
    return t || index_1.DEFAULT_TEXT_MODEL_ID;
}
/** AI Studio API 키 + 선택적 모델 오버라이드로 텍스트용 GenerativeModel */
function createAiStudioTextModel(apiKey, modelId) {
    const genAI = new generative_ai_1.GoogleGenerativeAI(apiKey.trim());
    return genAI.getGenerativeModel({ model: resolveAiStudioTextModelId(modelId) });
}
/** 단일 문자열 프롬프트 → 응답 텍스트 (AI Studio) */
async function generateAiStudioText(opts) {
    const model = createAiStudioTextModel(opts.apiKey, opts.modelId);
    const ro = opts.signal ? { signal: opts.signal } : undefined;
    const res = await model.generateContent(opts.prompt, ro);
    return res.response.text();
}
/** Vertex Publisher `generateContent` (API 키 인증, 브라우저 `gemini.ts`와 동일 REST 형태) */
async function generateVertexText(opts) {
    const model = resolveAiStudioTextModelId(opts.modelId);
    const loc = (opts.location?.trim() || index_1.DEFAULT_VERTEX_LOCATION).trim() || index_1.DEFAULT_VERTEX_LOCATION;
    const url = (0, index_1.buildVertexPublisherModelUrl)({
        projectId: opts.projectId.trim(),
        location: loc,
        modelId: model,
        method: 'generateContent',
        apiKey: opts.apiKey.trim(),
    });
    const body = {
        contents: [{ role: 'user', parts: [{ text: opts.userText }] }],
        generationConfig: { maxOutputTokens: 8192 },
    };
    const sys = opts.systemInstruction?.trim();
    if (sys) {
        body.systemInstruction = { parts: [{ text: sys }] };
    }
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: opts.signal,
    });
    const raw = await res.text();
    let data;
    try {
        data = JSON.parse(raw);
    }
    catch {
        throw new Error(`Vertex 응답 파싱 실패 HTTP ${res.status}: ${raw.slice(0, 400)}`);
    }
    if (!res.ok) {
        throw new Error(data.error?.message || data.error?.status || `Vertex HTTP ${res.status}: ${raw.slice(0, 400)}`);
    }
    if (data.error) {
        throw new Error(data.error.message || data.error.status || 'Vertex API 오류');
    }
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
    if (text == null || text === '') {
        throw new Error('응답에 텍스트가 없습니다: ' + JSON.stringify(data).slice(0, 500));
    }
    return text;
}
function readSurfaceRaw(env) {
    return (env.KARMOLAB_AI_SURFACE?.trim() ||
        env.GEMINI_SURFACE?.trim() ||
        env.GOOGLE_GEN_SURFACE?.trim() ||
        '');
}
/**
 * `vertex` | `vertex_ai` | `gcp_vertex` → Vertex, 그 외·비어 있음 → AI Studio.
 */
function parseGenerativeSurfaceFromEnv(env = process.env) {
    const s = readSurfaceRaw(env).toLowerCase().replace(/-/g, '_');
    if (s === 'vertex' || s === 'vertex_ai' || s === 'gcp_vertex')
        return 'vertex';
    return 'aiStudio';
}
/**
 * 시스템+맥락+질문을 한 문자열로 묶어 보낼 때(AI Studio `generateContent` / Vertex `generateContent` REST).
 * `surface: inherit` 이면 `KARMOLAB_AI_SURFACE` 등과 동일 규칙.
 */
async function generateBlobTextFromEnvWithOptions(env, blobPrompt, options = {}) {
    const surfaceChoice = options.surface ?? 'inherit';
    const surface = surfaceChoice === 'inherit' ? parseGenerativeSurfaceFromEnv(env) : surfaceChoice;
    const modelOverride = options.modelId != null ? String(options.modelId).trim() : '';
    const effectiveModelId = resolveAiStudioTextModelId(modelOverride ? modelOverride : env.GEMINI_MODEL);
    if (surface === 'vertex') {
        const apiKey = env.VERTEX_API_KEY?.trim();
        const projectId = env.VERTEX_PROJECT_ID?.trim();
        if (!apiKey || !projectId) {
            throw new Error('Vertex API: .env에 VERTEX_API_KEY와 VERTEX_PROJECT_ID가 필요합니다.');
        }
        const location = env.VERTEX_LOCATION?.trim() || undefined;
        const text = await generateVertexText({
            apiKey,
            projectId,
            location,
            modelId: effectiveModelId,
            userText: blobPrompt,
            signal: options.signal,
        });
        return { text, surface: 'vertex', modelId: effectiveModelId };
    }
    const apiKey = env.GEMINI_API_KEY?.trim();
    if (!apiKey) {
        throw new Error('AI Studio API: .env에 GEMINI_API_KEY가 필요합니다.');
    }
    const text = await generateAiStudioText({
        apiKey,
        modelId: effectiveModelId,
        prompt: blobPrompt,
        signal: options.signal,
    });
    return { text, surface: 'aiStudio', modelId: effectiveModelId };
}
function tryCreateGenerativeTextFromEnv(env = process.env) {
    const surface = parseGenerativeSurfaceFromEnv(env);
    if (surface === 'vertex') {
        const apiKey = env.VERTEX_API_KEY?.trim();
        const projectId = env.VERTEX_PROJECT_ID?.trim();
        if (!apiKey || !projectId)
            return null;
        const location = env.VERTEX_LOCATION?.trim() || undefined;
        const modelId = env.GEMINI_MODEL?.trim() || null;
        return {
            surface: 'vertex',
            generateFromPrompt(prompt, signal) {
                return generateVertexText({
                    apiKey,
                    projectId,
                    location,
                    modelId,
                    userText: prompt,
                    signal,
                });
            },
        };
    }
    const apiKey = env.GEMINI_API_KEY?.trim();
    if (!apiKey)
        return null;
    const modelId = env.GEMINI_MODEL?.trim() || null;
    return {
        surface: 'aiStudio',
        generateFromPrompt(prompt, signal) {
            return generateAiStudioText({ apiKey, modelId, prompt, signal });
        },
    };
}
/** `tryCreateGenerativeTextFromEnv`가 `null`일 때 안내용 */
function generativeEnvHint(env = process.env) {
    if (parseGenerativeSurfaceFromEnv(env) === 'vertex') {
        return 'Vertex 모드: .env에 VERTEX_API_KEY, VERTEX_PROJECT_ID 가 필요합니다. (선택: VERTEX_LOCATION, GEMINI_MODEL)';
    }
    return 'AI Studio 모드: .env에 GEMINI_API_KEY 가 필요합니다. (선택: GEMINI_MODEL, 또는 KARMOLAB_AI_SURFACE=vertex 로 전환)';
}
