"use strict";
/**
 * KarmoLabAI — Google Generative AI 공통 계약 (AI Studio + Vertex AI).
 * 브라우저/Node 공통: 모델 카탈로그, REST URL 조립, 문서·기본 리전 등. fetch·키 저장 없음.
 * Node에서 `@google/generative-ai` 호출까지 맞출 때는 서브패스 `karmolab-ai/node` 참고.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_GEMINI_TEXT_MODEL_ID = exports.DEFAULT_TEXT_MODEL_ID = exports.MODEL_CATALOG = exports.ENV_GOOGLE_AI = exports.DOC_URL_VERTEX_API_KEYS = exports.DOC_URL_AI_STUDIO_API_KEY = exports.DEFAULT_VERTEX_LOCATION = exports.AI_STUDIO_GENERATIVE_BASE = exports.AI_STUDIO_GENERATIVE_HOST = void 0;
exports.buildAiStudioGenerateContentUrl = buildAiStudioGenerateContentUrl;
exports.buildAiStudioStreamGenerateContentUrl = buildAiStudioStreamGenerateContentUrl;
exports.buildAiStudioPredictUrl = buildAiStudioPredictUrl;
exports.buildVertexPublisherModelUrl = buildVertexPublisherModelUrl;
exports.getDefaultModelId = getDefaultModelId;
// ─── AI Studio (Generative Language API, API 키 = AI Studio / Google AI) ─
exports.AI_STUDIO_GENERATIVE_HOST = 'generativelanguage.googleapis.com';
exports.AI_STUDIO_GENERATIVE_BASE = `https://${exports.AI_STUDIO_GENERATIVE_HOST}/v1beta`;
function buildAiStudioGenerateContentUrl(modelId, apiKey) {
    return `${exports.AI_STUDIO_GENERATIVE_BASE}/models/${encodeURIComponent(modelId)}:generateContent?key=${encodeURIComponent(apiKey)}`;
}
function buildAiStudioStreamGenerateContentUrl(modelId, apiKey) {
    return `${exports.AI_STUDIO_GENERATIVE_BASE}/models/${encodeURIComponent(modelId)}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`;
}
/** Imagen 등 `:predict` RPC (AI Studio) */
function buildAiStudioPredictUrl(modelId, apiKey) {
    return `${exports.AI_STUDIO_GENERATIVE_BASE}/models/${encodeURIComponent(modelId)}:predict?key=${encodeURIComponent(apiKey)}`;
}
// ─── Vertex AI (regional aiplatform, GCP 프로젝트 + 리전 + API 키) ───────
exports.DEFAULT_VERTEX_LOCATION = 'us-central1';
/**
 * Vertex: `projects/.../locations/.../publishers/google/models/{modelId}:{method}`
 * `streamGenerateContent` → `?alt=sse` (AI Studio와 동일 패턴)
 * @see https://cloud.google.com/vertex-ai/docs/reference/rest
 */
function buildVertexPublisherModelUrl(opts) {
    const loc = (opts.location || exports.DEFAULT_VERTEX_LOCATION).trim() || exports.DEFAULT_VERTEX_LOCATION;
    const pid = encodeURIComponent(opts.projectId.trim());
    const locEnc = encodeURIComponent(loc);
    const mid = encodeURIComponent(opts.modelId);
    const key = encodeURIComponent(opts.apiKey);
    const q = opts.method === 'streamGenerateContent' ? `?alt=sse&key=${key}` : `?key=${key}`;
    return (`https://${locEnc}-aiplatform.googleapis.com/v1/projects/${pid}` +
        `/locations/${locEnc}/publishers/google/models/${mid}:${opts.method}${q}`);
}
// ─── 문서 / 온보딩 URL ───────────────────────────────────────────────────
exports.DOC_URL_AI_STUDIO_API_KEY = 'https://aistudio.google.com/app/apikey';
exports.DOC_URL_VERTEX_API_KEYS = 'https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys';
/** 스크립트·봇 env 이름 (참고용, 런타임 읽기 없음) */
exports.ENV_GOOGLE_AI = {
    /** AI Studio 스타일 API 키 (욘봇·카카오 스크립트 등) */
    apiKey: 'GEMINI_API_KEY',
    modelOverride: 'GEMINI_MODEL',
    /** `aiStudio`(기본) 또는 `vertex` — `KARMOLAB_AI_SURFACE` 우선, 없으면 `GEMINI_SURFACE` */
    surfacePrimary: 'KARMOLAB_AI_SURFACE',
    surfaceAlt: 'GEMINI_SURFACE',
    vertexApiKey: 'VERTEX_API_KEY',
    vertexProjectId: 'VERTEX_PROJECT_ID',
    vertexLocation: 'VERTEX_LOCATION',
};
exports.MODEL_CATALOG = {
    gemini: [
        { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', isDefault: true },
        { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro' },
        { id: 'gemini-2.0-flash', name: 'Gemini 2 Flash' },
        { id: 'gemini-2.0-flash-lite', name: 'Gemini 2 Flash Lite' },
        { id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash Lite' },
        { id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash' },
        { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro' },
        { id: 'gemini-3.1-flash-lite-preview', name: 'Gemini 3.1 Flash Lite' },
    ],
    geminiImage: [
        { id: 'gemini-2.5-flash-image', name: 'Nano Banana (Gemini 2.5 Flash Image)', isDefault: true },
        { id: 'gemini-3-pro-image-preview', name: 'Nano Banana Pro (Gemini 3 Pro Image)' },
        { id: 'gemini-3.1-flash-image-preview', name: 'Nano Banana 2 (Gemini 3.1 Flash Image)' },
    ],
    imagen: [
        { id: 'imagen-4.0-generate-001', name: 'Imagen 4 Generate', isDefault: true },
        { id: 'imagen-4.0-ultra-generate-001', name: 'Imagen 4 Ultra Generate' },
        { id: 'imagen-4.0-fast-generate-001', name: 'Imagen 4 Fast Generate' },
    ],
};
function getDefaultModelId(provider) {
    const models = exports.MODEL_CATALOG[provider];
    if (!models?.length)
        return '';
    const def = models.find((m) => m.isDefault);
    return def ? def.id : models[0].id;
}
/** 텍스트 generateContent 기본 모델 (AI Studio·Vertex 동일 모델 ID 문자열) */
exports.DEFAULT_TEXT_MODEL_ID = getDefaultModelId('gemini');
/** @deprecated `DEFAULT_TEXT_MODEL_ID` 사용 권장 */
exports.DEFAULT_GEMINI_TEXT_MODEL_ID = exports.DEFAULT_TEXT_MODEL_ID;
