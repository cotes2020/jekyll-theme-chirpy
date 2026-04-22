/**
 * KarmoLabAI — Google Generative AI 공통 계약 (AI Studio + Vertex AI).
 * 브라우저/Node 공통: 모델 카탈로그, REST URL 조립, 문서·기본 리전 등. fetch·키 저장 없음.
 * Node에서 `@google/generative-ai` 호출까지 맞출 때는 서브패스 `karmolab-ai/node` 참고.
 */
export type GoogleGenerativeSurface = 'aiStudio' | 'vertex';
export declare const AI_STUDIO_GENERATIVE_HOST = "generativelanguage.googleapis.com";
export declare const AI_STUDIO_GENERATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta";
export declare function buildAiStudioGenerateContentUrl(modelId: string, apiKey: string): string;
export declare function buildAiStudioStreamGenerateContentUrl(modelId: string, apiKey: string): string;
/** Imagen 등 `:predict` RPC (AI Studio) */
export declare function buildAiStudioPredictUrl(modelId: string, apiKey: string): string;
export declare const DEFAULT_VERTEX_LOCATION = "us-central1";
/**
 * Vertex: `projects/.../locations/.../publishers/google/models/{modelId}:{method}`
 * `streamGenerateContent` → `?alt=sse` (AI Studio와 동일 패턴)
 * @see https://cloud.google.com/vertex-ai/docs/reference/rest
 */
export declare function buildVertexPublisherModelUrl(opts: {
    projectId: string;
    location?: string;
    modelId: string;
    method: 'generateContent' | 'streamGenerateContent' | 'predict';
    apiKey: string;
}): string;
export declare const DOC_URL_AI_STUDIO_API_KEY = "https://aistudio.google.com/app/apikey";
export declare const DOC_URL_VERTEX_API_KEYS = "https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys";
/** 스크립트·봇 env 이름 (참고용, 런타임 읽기 없음) */
export declare const ENV_GOOGLE_AI: {
    /** AI Studio 스타일 API 키 (욘봇·카카오 스크립트 등) */
    readonly apiKey: "GEMINI_API_KEY";
    readonly modelOverride: "GEMINI_MODEL";
    /** `aiStudio`(기본) 또는 `vertex` — `KARMOLAB_AI_SURFACE` 우선, 없으면 `GEMINI_SURFACE` */
    readonly surfacePrimary: "KARMOLAB_AI_SURFACE";
    readonly surfaceAlt: "GEMINI_SURFACE";
    readonly vertexApiKey: "VERTEX_API_KEY";
    readonly vertexProjectId: "VERTEX_PROJECT_ID";
    readonly vertexLocation: "VERTEX_LOCATION";
};
export type ModelProvider = 'gemini' | 'geminiImage' | 'imagen' | 'embedding';
export interface ModelEntry {
    id: string;
    name: string;
    isDefault?: boolean;
}
export declare const MODEL_CATALOG: Record<ModelProvider, ModelEntry[]>;
export declare function getDefaultModelId(provider: ModelProvider): string;
/** 텍스트 generateContent 기본 모델 (AI Studio·Vertex 동일 모델 ID 문자열) */
export declare const DEFAULT_TEXT_MODEL_ID: string;
