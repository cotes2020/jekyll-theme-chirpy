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
/** `/yawn` 슬래시: `.env` 기본 vs `aiStudio` / `vertex` 강제 */
export type GenerativeSurfaceOverride = 'inherit' | 'aiStudio' | 'vertex';
/**
 * 시스템+맥락+질문을 한 문자열로 묶어 보낼 때(AI Studio `generateContent` / Vertex `generateContent` REST).
 * `surface: inherit` 이면 `KARMOLAB_AI_SURFACE` 등과 동일 규칙.
 */
export declare function generateBlobTextFromEnvWithOptions(env: NodeJS.ProcessEnv, blobPrompt: string, options?: {
    surface?: GenerativeSurfaceOverride;
    modelId?: string | null;
    signal?: AbortSignal;
}): Promise<{
    text: string;
    surface: GoogleGenerativeSurface;
    modelId: string;
}>;
export declare function tryCreateGenerativeTextFromEnv(env?: NodeJS.ProcessEnv): GenerativeTextClient | null;
/** `tryCreateGenerativeTextFromEnv`가 `null`일 때 안내용 */
export declare function generativeEnvHint(env?: NodeJS.ProcessEnv): string;
/**
 * 로컬에 설치된 `claude` CLI (`claude --print`)로 텍스트 생성.
 * Claude Max 구독으로 인증된 환경에서 API 키 없이 사용 가능.
 *
 * 환경 변수:
 *   CLAUDE_CLI_COMMAND  : CLI 실행 파일 이름 (기본: claude)
 *   CLAUDE_CLI_TIMEOUT_MS : 타임아웃 ms (기본: 60000)
 */
export declare function generateClaudeCliText(opts: {
    prompt: string;
    timeoutMs?: number;
    /** 에이전트 모드: cwd 지정 시 파일 읽기/편집/명령 실행 가능 */
    cwd?: string;
}): Promise<string>;
export type AssistantAiProvider = 'gemini' | 'claude-cli';
export declare function resolveAssistantProvider(env?: NodeJS.ProcessEnv): AssistantAiProvider;
/**
 * ASSISTANT_AI_PROVIDER 에 따라 Gemini 또는 Claude CLI로 텍스트 생성.
 * assistant-handler, memory-service 등에서 공통으로 사용.
 *
 * claude-cli 프로바이더일 때 env.ASSISTANT_AGENT_REPO_PATH 가 설정돼 있으면
 * 해당 경로를 cwd로 설정해 에이전트 모드(파일 읽기/편집/명령 실행)로 실행.
 */
export declare function generateAssistantText(env: NodeJS.ProcessEnv, prompt: string, opts?: {
    timeoutMs?: number;
}): Promise<{
    text: string;
    provider: AssistantAiProvider;
}>;
