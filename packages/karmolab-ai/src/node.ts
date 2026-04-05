/**
 * Node 전용: AI Studio(Generative Language API) 텍스트 호출을 `@google/generative-ai`로 수행.
 * 브라우저·KarmoLab 번들에는 포함하지 말 것 — `import 'karmolab-ai/node'`.
 */
import { GoogleGenerativeAI } from '@google/generative-ai';
import { DEFAULT_TEXT_MODEL_ID } from './index';

export function resolveAiStudioTextModelId(modelFromEnv?: string | null): string {
  const t = modelFromEnv?.trim();
  return t || DEFAULT_TEXT_MODEL_ID;
}

/** AI Studio API 키 + 선택적 모델 오버라이드로 텍스트용 GenerativeModel */
export function createAiStudioTextModel(apiKey: string, modelId?: string | null) {
  const genAI = new GoogleGenerativeAI(apiKey.trim());
  return genAI.getGenerativeModel({ model: resolveAiStudioTextModelId(modelId) });
}

/** 단일 문자열 프롬프트 → 응답 텍스트 */
export async function generateAiStudioText(opts: {
  apiKey: string;
  modelId?: string | null;
  prompt: string;
}): Promise<string> {
  const model = createAiStudioTextModel(opts.apiKey, opts.modelId);
  const res = await model.generateContent(opts.prompt);
  return res.response.text();
}
