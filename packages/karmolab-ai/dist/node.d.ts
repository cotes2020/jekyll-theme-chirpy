export declare function resolveAiStudioTextModelId(modelFromEnv?: string | null): string;
/** AI Studio API 키 + 선택적 모델 오버라이드로 텍스트용 GenerativeModel */
export declare function createAiStudioTextModel(apiKey: string, modelId?: string | null): import("@google/generative-ai").GenerativeModel;
/** 단일 문자열 프롬프트 → 응답 텍스트 */
export declare function generateAiStudioText(opts: {
    apiKey: string;
    modelId?: string | null;
    prompt: string;
}): Promise<string>;
