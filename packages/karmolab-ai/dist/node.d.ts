import { type GoogleGenerativeSurface } from './index';
export type { GoogleGenerativeSurface };
export declare function resolveAiStudioTextModelId(modelFromEnv?: string | null): string;
/** AI Studio API 키 + 선택적 모델 오버라이드로 텍스트용 GenerativeModel */
export declare function createAiStudioTextModel(apiKey: string, modelId?: string | null): import("@google/generative-ai").GenerativeModel;
/** 단일 문자열 프롬프트 → 응답 텍스트 (AI Studio) */
export declare function generateAiStudioText(opts: {
    apiKey: string;
    modelId?: string | null;
    prompt: string;
    signal?: AbortSignal;
}): Promise<string>;
/** Vertex Publisher `generateContent` (API 키 인증, 브라우저 `gemini.ts`와 동일 REST 형태) */
export declare function generateVertexText(opts: {
    apiKey: string;
    projectId: string;
    location?: string | null;
    modelId?: string | null;
    userText: string;
    systemInstruction?: string | null;
    signal?: AbortSignal;
}): Promise<string>;
/**
 * `vertex` | `vertex_ai` | `gcp_vertex` → Vertex, 그 외·비어 있음 → AI Studio.
 */
export declare function parseGenerativeSurfaceFromEnv(env?: NodeJS.ProcessEnv): GoogleGenerativeSurface;
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
export declare function tryCreateGenerativeTextFromEnv(env?: NodeJS.ProcessEnv): GenerativeTextClient | null;
/** `tryCreateGenerativeTextFromEnv`가 `null`일 때 안내용 */
export declare function generativeEnvHint(env?: NodeJS.ProcessEnv): string;
