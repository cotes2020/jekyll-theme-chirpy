"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.resolveAiStudioTextModelId = resolveAiStudioTextModelId;
exports.createAiStudioTextModel = createAiStudioTextModel;
exports.generateAiStudioText = generateAiStudioText;
/**
 * Node 전용: AI Studio(Generative Language API) 텍스트 호출을 `@google/generative-ai`로 수행.
 * 브라우저·KarmoLab 번들에는 포함하지 말 것 — `import 'karmolab-ai/node'`.
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
/** 단일 문자열 프롬프트 → 응답 텍스트 */
async function generateAiStudioText(opts) {
    const model = createAiStudioTextModel(opts.apiKey, opts.modelId);
    const res = await model.generateContent(opts.prompt);
    return res.response.text();
}
