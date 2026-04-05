/**
 * Node 전용: AI Studio(`@google/generative-ai`) 또는 Vertex REST(`fetch`)로 텍스트 생성.
 * 브라우저 번들에 포함하지 말 것 — `import 'karmolab-ai/node'`.
 */
import { GoogleGenerativeAI } from '@google/generative-ai';
import {
  type GoogleGenerativeSurface,
  buildVertexPublisherModelUrl,
  DEFAULT_VERTEX_LOCATION,
  DEFAULT_TEXT_MODEL_ID,
} from './index';

export type { GoogleGenerativeSurface };

export function resolveAiStudioTextModelId(modelFromEnv?: string | null): string {
  const t = modelFromEnv?.trim();
  return t || DEFAULT_TEXT_MODEL_ID;
}

/** AI Studio API 키 + 선택적 모델 오버라이드로 텍스트용 GenerativeModel */
export function createAiStudioTextModel(apiKey: string, modelId?: string | null) {
  const genAI = new GoogleGenerativeAI(apiKey.trim());
  return genAI.getGenerativeModel({ model: resolveAiStudioTextModelId(modelId) });
}

/** 단일 문자열 프롬프트 → 응답 텍스트 (AI Studio) */
export async function generateAiStudioText(opts: {
  apiKey: string;
  modelId?: string | null;
  prompt: string;
  signal?: AbortSignal;
}): Promise<string> {
  const model = createAiStudioTextModel(opts.apiKey, opts.modelId);
  const ro = opts.signal ? { signal: opts.signal } : undefined;
  const res = await model.generateContent(opts.prompt, ro);
  return res.response.text();
}

type VertexJson = {
  error?: { message?: string; status?: string };
  candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
};

/** Vertex Publisher `generateContent` (API 키 인증, 브라우저 `gemini.ts`와 동일 REST 형태) */
export async function generateVertexText(opts: {
  apiKey: string;
  projectId: string;
  location?: string | null;
  modelId?: string | null;
  userText: string;
  systemInstruction?: string | null;
  signal?: AbortSignal;
}): Promise<string> {
  const model = resolveAiStudioTextModelId(opts.modelId);
  const loc = (opts.location?.trim() || DEFAULT_VERTEX_LOCATION).trim() || DEFAULT_VERTEX_LOCATION;
  const url = buildVertexPublisherModelUrl({
    projectId: opts.projectId.trim(),
    location: loc,
    modelId: model,
    method: 'generateContent',
    apiKey: opts.apiKey.trim(),
  });
  const body: Record<string, unknown> = {
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
  let data: VertexJson;
  try {
    data = JSON.parse(raw) as VertexJson;
  } catch {
    throw new Error(`Vertex 응답 파싱 실패 HTTP ${res.status}: ${raw.slice(0, 400)}`);
  }
  if (!res.ok) {
    throw new Error(
      data.error?.message || data.error?.status || `Vertex HTTP ${res.status}: ${raw.slice(0, 400)}`,
    );
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

function readSurfaceRaw(env: NodeJS.ProcessEnv): string {
  return (
    env.KARMOLAB_AI_SURFACE?.trim() ||
    env.GEMINI_SURFACE?.trim() ||
    env.GOOGLE_GEN_SURFACE?.trim() ||
    ''
  );
}

/**
 * `vertex` | `vertex_ai` | `gcp_vertex` → Vertex, 그 외·비어 있음 → AI Studio.
 */
export function parseGenerativeSurfaceFromEnv(env: NodeJS.ProcessEnv = process.env): GoogleGenerativeSurface {
  const s = readSurfaceRaw(env).toLowerCase().replace(/-/g, '_');
  if (s === 'vertex' || s === 'vertex_ai' || s === 'gcp_vertex') return 'vertex';
  return 'aiStudio';
}

export type GenerativeTextClient = {
  surface: GoogleGenerativeSurface;
  /** 단일 사용자 프롬프트(또는 시스템+사용자를 한 덩어리로 넣은 문자열) */
  generateFromPrompt: (prompt: string, signal?: AbortSignal) => Promise<string>;
};

/**
 * `.env` 기준으로 호출 가능한 텍스트 클라이언트를 만듦. 자격이 없으면 `null`.
 *
 * - **AI Studio (기본):** `GEMINI_API_KEY` 필수, `GEMINI_MODEL` 선택
 * - **Vertex:** `KARMOLAB_AI_SURFACE=vertex`(또는 `GEMINI_SURFACE`) + `VERTEX_API_KEY`, `VERTEX_PROJECT_ID` 필수, `VERTEX_LOCATION`·`GEMINI_MODEL` 선택
 */
export function tryCreateGenerativeTextFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): GenerativeTextClient | null {
  const surface = parseGenerativeSurfaceFromEnv(env);
  if (surface === 'vertex') {
    const apiKey = env.VERTEX_API_KEY?.trim();
    const projectId = env.VERTEX_PROJECT_ID?.trim();
    if (!apiKey || !projectId) return null;
    const location = env.VERTEX_LOCATION?.trim() || undefined;
    const modelId = env.GEMINI_MODEL?.trim() || null;
    return {
      surface: 'vertex',
      generateFromPrompt(prompt: string, signal?: AbortSignal) {
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
  if (!apiKey) return null;
  const modelId = env.GEMINI_MODEL?.trim() || null;
  return {
    surface: 'aiStudio',
    generateFromPrompt(prompt: string, signal?: AbortSignal) {
      return generateAiStudioText({ apiKey, modelId, prompt, signal });
    },
  };
}

/** `tryCreateGenerativeTextFromEnv`가 `null`일 때 안내용 */
export function generativeEnvHint(env: NodeJS.ProcessEnv = process.env): string {
  if (parseGenerativeSurfaceFromEnv(env) === 'vertex') {
    return 'Vertex 모드: .env에 VERTEX_API_KEY, VERTEX_PROJECT_ID 가 필요합니다. (선택: VERTEX_LOCATION, GEMINI_MODEL)';
  }
  return 'AI Studio 모드: .env에 GEMINI_API_KEY 가 필요합니다. (선택: GEMINI_MODEL, 또는 KARMOLAB_AI_SURFACE=vertex 로 전환)';
}
